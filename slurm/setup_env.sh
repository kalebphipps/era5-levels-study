#!/usr/bin/env bash
# One-time environment setup on HoreKa (venv, no container).
#
#   export WS=$(ws_find levels)            # a workspace, NOT $HOME (quota)
#   export BEAST_DIR=$HOME/beast           # your local beast checkout
#   bash slurm/setup_env.sh
#
# Installs a CUDA-matched PyTorch, then editable-installs beast AND its jigsaw
# submodule, then this study package. NOTE: jigsaw is a git submodule of beast
# (libs/jigsaw) and is imported as a top-level `jigsaw` package; it is NOT listed
# in beast's pyproject, so `pip install -e beast` alone does not provide it. We
# init the submodule and install it explicitly. torch_blue IS a git dependency in
# beast's pyproject and installs automatically.
set -euo pipefail

: "${WS:?Set WS, e.g. export WS=\$(ws_find levels)}"
: "${BEAST_DIR:?Set BEAST_DIR to your beast checkout, e.g. export BEAST_DIR=\$HOME/beast}"

VENV_DIR="$WS/venv"
TORCH_CUDA="${TORCH_CUDA:-cu124}"          # match the H100/H200 node driver (nvidia-smi)
PYTHON_BIN="${PYTHON_BIN:-python3}"

module purge 2>/dev/null || true
# module load devel/python/3.12            # adjust to an available module if needed

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# Make sure the jigsaw submodule is checked out in the beast tree.
git -C "$BEAST_DIR" submodule update --init --recursive

# PyTorch first (CUDA wheel), then beast (editable, brings torch_blue), then the
# jigsaw submodule (editable, top-level `jigsaw` import), then this study repo.
pip install --upgrade torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
pip install -e "$BEAST_DIR"
pip install -e "$BEAST_DIR/libs/jigsaw"
pip install -e "$(cd "$(dirname "$0")/.." && pwd)"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
for mod in ("jigsaw", "beast", "era5_levels.config"):
    try:
        import importlib
        importlib.import_module(mod)
        print(f"{mod} OK")
    except Exception as e:
        print(f"{mod} import issue:", e)
PY
echo "Activate later with:  source $VENV_DIR/bin/activate"
