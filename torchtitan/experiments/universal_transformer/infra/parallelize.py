import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, SequenceParallel,
    parallelize_module, PrepareModuleInput,
)

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.infra.parallelize import apply_compile, apply_fsdp, apply_ddp
from torchtitan.tools.logging import logger


def _maybe_compile_model(model: nn.Module, job_config):
    """
    Compile the model in a way that's compatible with both UT (no .layers)
    and classic stacks (with .layers). Prefer the repo helper only when it
    won't crash; otherwise fall back to torch.compile.
    """
    if not getattr(job_config.compile, "enable", False):
        return model
    # Try the repo helper only if it can operate on this model shape
    if hasattr(model, "layers"):
        try:
            out = apply_compile(model)  # some versions take (model)
            return out if out is not None else model
        except TypeError:
            out = apply_compile(model, job_config)  # others take (model, job_config)
            return out if out is not None else model
    # UT fallback: compile whole module
    dynamic = bool(getattr(job_config.compile, "dynamic", False))
    return torch.compile(model, dynamic=dynamic)


def _apply_fsdp_flexible(model: nn.Module, parallel_dims: ParallelDims, job_config: JobConfig):
    """
    Apply FSDP to Universal Transformer with pre/shared_block/post structure.
    Based on Llama3 apply_fsdp but adapted for UniversalLlama architecture.
    """
    if parallel_dims.fsdp_enabled:
        from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
        
        world_mesh = parallel_dims.world_mesh
        
        # Determine dp mesh names
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
            
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]
        
        # Get dtypes from job config (exactly like Llama3)
        param_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_param] 
        reduce_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce]
        
        # Set up FSDP config (same as Llama3)
        fsdp_config = {
            "mesh": dp_mesh,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
            ),
        }
        
        if job_config.training.enable_cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()
            
        # Map reshard policy
        reshard_after_forward_policy_map = {
            "default": True,
            "always": True, 
            "never": False,
        }
        reshard_after_forward = reshard_after_forward_policy_map.get(
            job_config.parallelism.fsdp_reshard_after_forward, True
        )
        
        # Apply FSDP to embedding layer (same as Llama3)
        if hasattr(model, "tok_embeddings"):
            fully_shard(
                model.tok_embeddings,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        
        # Apply FSDP to pre layers (like model.layers in Llama3)
        if hasattr(model, "pre"):
            for layer_id, transformer_block in model.pre.items():
                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
        
        # Apply FSDP to shared block
        if hasattr(model, "shared_block"):
            fully_shard(
                model.shared_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
            
        # Apply FSDP to post layers 
        if hasattr(model, "post"):
            for layer_id, transformer_block in model.post.items():
                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
        
        # Apply FSDP to norm and output (same as Llama3, but don't reshard for optimization)
        if hasattr(model, "norm") and model.norm is not None:
            fully_shard(model.norm, **fsdp_config, reshard_after_forward=False)
        if hasattr(model, "output") and model.output is not None:
            fully_shard(model.output, **fsdp_config, reshard_after_forward=False)
        
        # Apply FSDP to the root model
        fully_shard(model, **fsdp_config, reshard_after_forward=reshard_after_forward)
        
        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to Universal Transformer")
        else:
            logger.info("Applied FSDP to Universal Transformer")
            
    elif parallel_dims.dp_replicate_enabled:
        # Use DDP instead
        world_mesh = parallel_dims.world_mesh
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=job_config.compile.enable,
            enable_compiled_autograd=job_config.parallelism.enable_compiled_autograd,
        )
        logger.info("Applied DDP to Universal Transformer")
    
    return model


def _apply_ddp_flexible(model: nn.Module, parallel_dims: ParallelDims, job_config: JobConfig):
    """
    Apply DDP using the exact same logic as Llama3.
    Reference: lines 125-130 in llama3/infra/parallelize.py
    """
    if parallel_dims.dp_replicate_enabled:
        world_mesh = parallel_dims.world_mesh
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=job_config.compile.enable,
            enable_compiled_autograd=job_config.parallelism.enable_compiled_autograd,
        )
        logger.info("Applied DDP to the model")
    return model

# ---- portable checkpoint wrapper shim ---------------------------------------
def _get_checkpoint_wrapper():
    """
    Returns f(module, preserve_rng_state=False, early_stop=False) -> wrapped_module.
    Tries: (1) torch.distributed.algorithms._checkpoint.checkpoint_wrapper (if present)
           (2) torch.utils.checkpoint.checkpoint_wrapper
           (3) minimal local wrapper using torch.utils.checkpoint.checkpoint
    """
    # (1) Prefer the DP-friendly algo wrapper if your build has it
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # type: ignore
            checkpoint_wrapper as _algo_ckpt_wrap,
            CheckpointImpl,
        )
        def _wrap(m, preserve_rng_state=False, early_stop=False):
            # NOTE: this API does NOT accept `use_reentrant`; select via CheckpointImpl
            return _algo_ckpt_wrap(
                m,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                preserve_rng_state=preserve_rng_state,
            )
        return _wrap
    except Exception:
        pass
    # (2) utils.checkpoint.checkpoint_wrapper (exists in many stable builds)
    try:
        from torch.utils.checkpoint import checkpoint_wrapper as _utils_ckpt_wrap  # type: ignore
        def _wrap(m, preserve_rng_state=False, early_stop=False):
            # prefer non-reentrant for better compile compatibility
            return _utils_ckpt_wrap(m, preserve_rng_state=preserve_rng_state, use_reentrant=False)
        return _wrap
    except Exception:
        pass
    # (3) last resort: create a tiny wrapper that calls checkpoint() in forward
    from torch.utils.checkpoint import checkpoint as _utils_checkpoint
    class _SimpleCheckpointWrapper(nn.Module):
        def __init__(self, mod, preserve_rng_state=False):
            super().__init__()
            # register so params stay visible to FSDP/DDP and DCP can unwrap
            self.module = mod
            self._checkpoint_wrapped_module = mod
            self._preserve = preserve_rng_state

        def forward(self, *args, **kwargs):
            fn = self.module
            return _utils_checkpoint(
                fn, *args, use_reentrant=False, preserve_rng_state=self._preserve, **kwargs
            )

        # keep it dumb; FSDP will wrap *outside* of this now
            
    def _wrap(m, preserve_rng_state=False, early_stop=False):
        return _SimpleCheckpointWrapper(m, preserve_rng_state=preserve_rng_state)
    return _wrap
_ckpt_wrap = _get_checkpoint_wrapper()


def parallelize_ut(model: nn.Module, parallel_dims: ParallelDims, job_config: JobConfig):
    # --- TP sharding (regex hits inside any nested UT block) ---
    if parallel_dims.tp_enabled:
        mesh_tp = DeviceMesh(parallel_dims.world_mesh.device_type,
                             parallel_dims.world_mesh.get_dim_groups(["tp"]))
        parallelize_module(
            module=model,
            device_mesh=mesh_tp,
            parallelize_plan={
                ".*attention.wqkv": ColwiseParallel(),
                ".*attention.wo": RowwiseParallel(),
                ".*feed_forward.w1": ColwiseParallel(),
                ".*feed_forward.w3": ColwiseParallel(),
                ".*feed_forward.w2": RowwiseParallel(),
            },
            prepare_input=PrepareModuleInput(
                input_layouts={"input_ids": Replicate()},
                desired_input_layouts={"input_ids": Replicate()},
            ),
            sequence_parallel=SequenceParallel(),
        )
        maybe_enable_async_tp(job_config, parallel_dims)

    # --- compile, then FSDP, then checkpoint, then DDP ---
    model = _maybe_compile_model(model, job_config)                # compile first
    fsdp_try = _apply_fsdp_flexible(model, parallel_dims, job_config)   # shard next
    if fsdp_try is None:
        logger.warning("UT: FSDP helper signature mismatch; proceeding with DDP-only (no sharding).")
        fsdpd = model
    else:
        fsdpd = fsdp_try
    # Now apply activation checkpoint wrappers OUTSIDE FSDP (your agent's fix)
    ac_cfg = job_config.activation_checkpoint
    if getattr(ac_cfg, "mode", "off").lower() != "off":
        if hasattr(fsdpd, "pre"):
            for k, blk in fsdpd.pre.items():
                fsdpd.pre[k] = _ckpt_wrap(blk, preserve_rng_state=False, early_stop=ac_cfg.early_stop)
        if hasattr(fsdpd, "shared_block"):
            fsdpd.shared_block = _ckpt_wrap(fsdpd.shared_block, preserve_rng_state=False, early_stop=ac_cfg.early_stop)
        if hasattr(fsdpd, "post"):
            for k, blk in fsdpd.post.items():
                fsdpd.post[k] = _ckpt_wrap(blk, preserve_rng_state=False, early_stop=ac_cfg.early_stop)
        logger.info("UT: activation checkpointing ENABLED (wrapped outside FSDP)")
    else:
        logger.info("UT: activation checkpointing DISABLED")
    model = _apply_ddp_flexible(fsdpd, parallel_dims, job_config)
    return model