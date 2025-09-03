from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_ut
from .infra.pipeline import pipeline_ut
from .model.model import UniversalLlama, UniversalArgs

# Ready-made flavors (you can tweak in the TOML too)
ut_configs = {
    # "8B-ish" compute profile with tied middle; params << 8B since K is compute-only.
    "8B": UniversalArgs(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, rope_theta=500000,
                        pre_layers=2, shared_depth=6, post_layers=2, use_depth_embedding=True),
    # ~1B parameter target (fits 4×RTX 4090 comfortably; see TOML below)
    # dim=3072, heads=24 (head_dim=128), effective layers = 2 + 6 + 2 = 10
    "1B": UniversalArgs(dim=3072, n_layers=10, n_heads=24, n_kv_heads=8, rope_theta=500000,
                        pre_layers=2, shared_depth=6, post_layers=2, use_depth_embedding=True),
}

register_train_spec(
    TrainSpec(
        name="llama3_ut",
        model_cls=UniversalLlama,
        model_args=ut_configs,
        parallelize_fn=parallelize_ut,
        pipelining_fn=pipeline_ut,  # v1: NoPP
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)