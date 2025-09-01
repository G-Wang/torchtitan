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

echo "==> Starting Llama-3 1B training on 4x GPUs"
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_1b_deepseek.toml" \
NGPU=4 \
./run_train.sh

# Uncomment to try the Llama-4 1B dense config instead:
# CONFIG_FILE="./torchtitan/experiments/llama4/train_configs/llama4_1b_deepseek.toml" \
# NGPU=4 \
# ./run_train.sh