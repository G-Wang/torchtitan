# Universal Transformer

This experiment implements Universal Transformers with configurable pre/post layers and shared middle layers.

## Architecture

The Universal Transformer follows this structure:
```
Input → Pre-Layers → Universal Layers (repeated depth times) → Post-Layers → Output
```

### Key Features:
- **Fixed Pre-layers**: Distinct preprocessing transformer blocks
- **Shared Universal Layer**: Single transformer block reused multiple times
- **Fixed Post-layers**: Distinct postprocessing transformer blocks  
- **Depth Embeddings**: Learned positional encoding for each repetition step
- **Parameter Efficiency**: Fewer unique parameters through weight sharing

## Configurations

### 8B Variant
- `dim=4096, heads=32, n_kv_heads=8`
- `pre_layers=2, shared_depth=6, post_layers=2`
- Effective compute: 10 layers, but much fewer parameters than 10 distinct blocks

### 1B Variant  
- `dim=3072, heads=24, n_kv_heads=8`
- `pre_layers=2, shared_depth=6, post_layers=2`
- Optimized for 4×RTX 4090 training

## Usage

```bash
# 8B-ish UT demo
CONFIG_FILE="torchtitan/experiments/universal_transformer/train_configs/llama3_ut_8b.toml" ./run_train.sh

# 1B-equivalent UT on 4×4090
CONFIG_FILE="torchtitan/experiments/universal_transformer/train_configs/llama3_ut_1b.toml" ./run_train.sh
```

## Implementation Notes

- Reuses Llama3 TransformerBlock components for compatibility
- Full distributed training support (TP/AC/FSDP/DDP)
- Explicit activation checkpointing of pre/shared/post blocks
- Compile-friendly with proper buffer management
- Stable FQNs via ModuleDict for checkpointing compatibility
- V1 implementation disables Pipeline Parallelism (can be added later)