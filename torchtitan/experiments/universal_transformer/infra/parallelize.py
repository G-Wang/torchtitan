import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, SequenceParallel,
    parallelize_module, PrepareModuleInput,
)

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.infra.parallelize import apply_compile, apply_fsdp, apply_ddp
from torchtitan.tools.logging import logger

# ---- portable checkpoint wrapper shim ---------------------------------------
def _get_checkpoint_wrapper():
    """
    Returns a function f(module, preserve_rng_state=False, early_stop=False) -> wrapped_module
    Tries (1) composable wrapper, (2) torch.utils.checkpoint.checkpoint_wrapper,
    then (3) a minimal local wrapper using torch.utils.checkpoint.checkpoint.
    """
    # (1) new composable API (PyTorch 2.3+ in many builds)
    try:
        from torch.distributed._composable.checkpoint import checkpoint_wrapper as _ptd_ckpt_wrap  # type: ignore
        def _wrap(m, preserve_rng_state=False, early_stop=False):
            # map args as needed; ignore early_stop if not supported
            return _ptd_ckpt_wrap(m, preserve_rng_state=preserve_rng_state)
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
            self._checkpoint_wrapped_module = mod  # duck-typing compatibility
            self._preserve = preserve_rng_state
        def forward(self, *args, **kwargs):
            fn = self._checkpoint_wrapped_module
            return _utils_checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=self._preserve, **kwargs)
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

    # --- Activation Checkpointing ---
    ac_cfg = job_config.activation_checkpoint
    # Use core AC only if the model exposes a standard .layers stack.
    if hasattr(model, "layers"):
        apply_ac(
            model,
            ac_cfg,
            model_compile_enabled=job_config.compile.enable,
            use_flex_attn=getattr(model.model_args, "use_flex_attn", False),
        )
    # Wrap UT's pre/shared/post blocks locally when AC is enabled.
    if getattr(ac_cfg, "mode", "off").lower() != "off":
        if hasattr(model, "pre"):
            for k, blk in model.pre.items():
                model.pre[k] = _ckpt_wrap(
                    blk, preserve_rng_state=False, early_stop=ac_cfg.early_stop
                )
        if hasattr(model, "shared_block"):
            model.shared_block = _ckpt_wrap(
                model.shared_block, preserve_rng_state=False, early_stop=ac_cfg.early_stop
            )
        if hasattr(model, "post"):
            for k, blk in model.post.items():
                model.post[k] = _ckpt_wrap(
                    blk, preserve_rng_state=False, early_stop=ac_cfg.early_stop
                )
        logger.info("UT: activation checkpointing ENABLED on pre/shared/post")
    else:
        logger.info("UT: activation checkpointing DISABLED")

    # --- compile & FSDP & DDP (reusing Llama3 helpers) ---
    apply_compile(model, job_config)
    model = apply_fsdp(model, parallel_dims, job_config)
    model = apply_ddp(model, parallel_dims, job_config)
    return model