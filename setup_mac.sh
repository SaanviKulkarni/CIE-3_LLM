#!/usr/bin/env bash
# One-time setup for macOS (Apple Silicon) with Python 3.10.
# Usage:  bash setup_mac.sh        (run from the project folder)
set -euo pipefail
cd "$(dirname "$0")"

PY=$(command -v python3.10 || true)
if [ -z "$PY" ]; then
  echo "python3.10 not found. Install it with:  brew install python@3.10"
  exit 1
fi
echo "Using $PY ($($PY --version))"

if [ -d .venv ] && ! .venv/bin/python --version >/dev/null 2>&1; then
  echo "Existing .venv is broken (no working python) - recreating it."
  rm -rf .venv
fi
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python --version | grep -q "3.10" || { echo "venv is not Python 3.10"; exit 1; }

python -m pip install "pip<24.1"
python -m pip install torch

# fairseq (required by IndicXlit) from official source, cloned OUTSIDE the project folder.
FAIRSEQ_DIR="${FAIRSEQ_DIR:-$HOME/fairseq}"
if ! python -c "import fairseq" 2>/dev/null; then
  [ -d "$FAIRSEQ_DIR" ] || git clone https://github.com/facebookresearch/fairseq.git "$FAIRSEQ_DIR"
  python -m pip install --editable "$FAIRSEQ_DIR"
fi

python -m pip install -r requirements.txt
python -m pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

echo
echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  huggingface-cli login     # after accepting the IndicTrans2 gate on its model page"
echo "  python tests/probe_models.py versions"
