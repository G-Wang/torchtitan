from typing import List, Tuple
import torch.nn as nn
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.train_spec import _PipelineSchedule

def pipeline_ut(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device=None,
    model_args=None,
    parallelize_fn=None,
    loss_fn=None,
) -> Tuple[_PipelineSchedule, List[nn.Module], bool, bool]:
    if parallel_dims.pp_enabled:
        raise RuntimeError("UniversalTransformer v1: set pipeline_parallel_degree=1.")
    return "NoPP", [model], True, True