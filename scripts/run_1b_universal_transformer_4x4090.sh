#!/usr/bin/env bash
set -euo pipefail

# Where to put the tokenizer files locally:
HF_ASSETS_DIR="${HF_ASSETS_DIR:-./hf_assets/deepseek-v3}"
mkdir -p "${HF_ASSETS_DIR}"

echo "==> Downloading DeepSeek-V3 tokenizer into ${HF_ASSETS_DIR}"
# The bundled helper supports pulling just the tokenizer from HF.
# Example repo_id accepted by the script: deepseek-ai/DeepSeek-V3
python scripts/download_hf_assets.py \
  --repo_id deepseek-ai/DeepSeek-V3 \
  --assets tokenizer \
  --local_dir "${HF_ASSETS_DIR}"
# (Pass --hf_token ... if your HF account requires it) 

echo "==> Starting Universal Transformer 1B training on 4x GPUs"
CONFIG_FILE="./torchtitan/experiments/universal_transformer/train_configs/llama3_ut_1b.toml" \
NGPU=4 \
./run_train.sh

# Uncomment to try the 8B-ish Universal Transformer config instead:
# CONFIG_FILE="./torchtitan/experiments/universal_transformer/train_configs/llama3_ut_8b.toml" \
# NGPU=8 \
# ./run_train.sh