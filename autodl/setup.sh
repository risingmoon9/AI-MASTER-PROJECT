#!/usr/bin/env bash
set -euo pipefail
HQA_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HQA_PROJECT_ROOT"
HQA_PYTHON="${HQA_PYTHON:-python3}"
"$HQA_PYTHON" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10-3.12"'
if [[ ! -d .venv ]]; then
    "$HQA_PYTHON" -m venv .venv
fi
source autodl/env.sh
python -m pip install --upgrade pip setuptools wheel
TORCH_CUDA="${TORCH_CUDA:-cu118}"
case "$TORCH_CUDA" in
    cu118|cu126|cu128|cpu) ;;
    *) echo "TORCH_CUDA must be cu118, cu126, cu128, or cpu" >&2; exit 2 ;;
esac
python -m pip install "torch==2.7.1+$TORCH_CUDA" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
mkdir -p runs/audit
python -m pip freeze > runs/audit/installed_packages.txt
python - <<'PY'
import json
import platform
from pathlib import Path
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(32, 32, device=device, requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
report = {'python': platform.python_version(), 'torch': torch.__version__,
          'cuda': torch.version.cuda, 'device': device,
          'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None}
Path('runs/audit/setup_environment.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
echo "Setup complete. Import data, then run: python scripts/preflight.py"
