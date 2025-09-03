import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, SequenceParallel,
    parallelize_module, PrepareModuleInput,
)
from torch.distributed._composable.checkpoint import checkpoint_wrapper as ptd_checkpoint_wrapper

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.infra.parallelize import apply_compile, apply_fsdp, apply_ddp
from torchtitan.tools.logging import logger


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

    # --- Activation Checkpointing: wrap pre / shared / post explicitly ---
    apply_ac(model, job_config.activation_checkpoint,
             model_compile_enabled=job_config.compile.enable,
             use_flex_attn=getattr(model.model_args, "use_flex_attn", False))
    if hasattr(model, "pre"):
        for k, blk in model.pre.items():
            model.pre[k] = ptd_checkpoint_wrapper(
                blk, preserve_rng_state=False,
                early_stop=job_config.activation_checkpoint.early_stop
            )
    if hasattr(model, "shared_block"):
        model.shared_block = ptd_checkpoint_wrapper(
            model.shared_block, preserve_rng_state=False,
            early_stop=job_config.activation_checkpoint.early_stop
        )
    if hasattr(model, "post"):
        for k, blk in model.post.items():
            model.post[k] = ptd_checkpoint_wrapper(
                blk, preserve_rng_state=False,
                early_stop=job_config.activation_checkpoint.early_stop
            )
    logger.info("UT: applied activation checkpointing to pre/shared/post")

    # --- compile & FSDP & DDP (reusing Llama3 helpers) ---
    apply_compile(model, job_config)
    model = apply_fsdp(model, parallel_dims, job_config)
    model = apply_ddp(model, parallel_dims, job_config)
    return model