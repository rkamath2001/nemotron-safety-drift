#!/usr/bin/env bash
# Three isolated environments, because vLLM, garak and the training stack pin conflicting transformers versions.
#   .venv-serve : vLLM + pipeline (generation, judging, analysis)
#   .venv-garak : garak scanner
#   .venv-train : LoRA training + merge
# Requires: CUDA 12.x GPU with >= 80 GB, ~250 GB free disk, a Hugging Face token with access to
# meta-llama/Llama-3.1-8B-Instruct (the guard's base model is gated).
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv -q --python 3.12 .venv-serve
uv pip install -q --python .venv-serve -r requirements/serve.txt

uv venv -q --python 3.12 .venv-garak
uv pip install -q --python .venv-garak -r requirements/garak.txt

uv venv -q --python 3.12 .venv-train
uv pip install -q --python .venv-train -r requirements/train.txt
# Fused Mamba kernels: without them the hybrid layers fall back to a very slow PyTorch path. Builds take a while.
uv pip install -q --python .venv-train --no-build-isolation mamba-ssm causal-conv1d || \
  echo "WARNING: mamba kernels failed to build; training will run but slowly. See README 'Training fallback'."

.venv-serve/bin/huggingface-cli whoami >/dev/null 2>&1 || .venv-serve/bin/huggingface-cli login
echo "setup complete. Next: ./run.sh"
