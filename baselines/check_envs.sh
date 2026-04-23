#!/bin/bash
# Check that all required libraries import correctly in each baseline env.

BASELINES_DIR="$(cd "$(dirname "$0")" && pwd)"

get_python() {
    conda info --envs 2>/dev/null | awk -v env="$1" '$1==env{print $NF"/bin/python3"}'
}

# check_env NAME ENV_NAME LIBS_PY [PREAMBLE_PY]
# PREAMBLE_PY: optional Python code injected before the import checks (e.g. compatibility shims).
check_env() {
    local name=$1
    local env=$2
    local PYTHON
    PYTHON=$(get_python "$env")

    echo "--- [$name] env: $env ---"

    if [ ! -x "$PYTHON" ]; then
        echo "  FAIL: env not found (run ${env}_setup.sh first)"
        echo ""
        return
    fi
    echo "  Python: $PYTHON"

    "$PYTHON" - <<PYEOF
import sys
${4:-}

libs = $3

ok, fail = [], []
for lib, import_str in libs:
    try:
        exec(import_str)
        ok.append(lib)
    except Exception as e:
        fail.append((lib, str(e).split(chr(10))[0]))

if ok:
    print("  OK:   " + ", ".join(ok))
for lib, err in fail:
    print(f"  FAIL: {lib} — {err}")
PYEOF
    echo ""
}

# ── GeoDiff ──────────────────────────────────────────────────────────────────
GEODIFF_REPO="$BASELINES_DIR/GeoDiff"
check_env "GeoDiff" "geodiff" "[
    ('torch',           'import torch; assert torch.cuda.is_available(), \"CUDA not available\"'),
    ('torch_geometric', 'import torch_geometric'),
    ('torch_scatter',   'import torch_scatter'),
    ('rdkit',           'from rdkit import Chem'),
    ('easydict',        'from easydict import EasyDict'),
    ('yaml',            'import yaml'),
    ('geodiff_model',   'import sys; sys.path.insert(0, \"$GEODIFF_REPO\"); from models.epsnet import get_model'),
]"

# ── GeoMol ───────────────────────────────────────────────────────────────────
GEOMOL_REPO="$BASELINES_DIR/GeoMol"
check_env "GeoMol" "geomol" "[
    ('torch',           'import torch; assert torch.cuda.is_available(), \"CUDA not available\"'),
    ('torch_geometric', 'import torch_geometric'),
    ('torch_scatter',   'import torch_scatter'),
    ('numpy',           'import numpy'),
    ('rdkit',           'from rdkit import Chem'),
    ('yaml',            'import yaml'),
    ('pot',             'import ot'),
    ('geomol_model',    'import sys; sys.path.insert(0, \"$GEOMOL_REPO\"); from model.model import GeoMol'),
]"

# ── MCF ──────────────────────────────────────────────────────────────────────
MCF_REPO="$BASELINES_DIR/ml-mcf"
check_env "MCF" "mcf" "[
    ('torch',           'import torch; assert torch.cuda.is_available(), \"CUDA not available\"'),
    ('lightning',       'import lightning'),
    ('omegaconf',       'from omegaconf import OmegaConf'),
    ('numpy',           'import numpy'),
    ('einops',          'import einops'),
    ('xformers',        'import xformers'),
    ('mcf_arch',        'import sys; sys.path.insert(0, \"$MCF_REPO\"); from models.architectures import PerceiverIO'),
]"

# ── Torsional Diffusion ───────────────────────────────────────────────────────
# Preamble: inject DiagnosticOptions shim before any import check runs.
# torch_geometric (and code that imports it) fails on PyTorch 2.6+ because
# torch.onnx._internal.exporter.DiagnosticOptions was removed. Setting the
# attribute on the already-imported module object makes 'from ... import
# DiagnosticOptions' succeed for all subsequent exec() calls in this process.
TORDIFF_REPO="$BASELINES_DIR/torsional-diffusion"
TORDIFF_PREAMBLE='import torch.onnx._internal.exporter as _onnx_exp
if not hasattr(_onnx_exp, "DiagnosticOptions"):
    _onnx_exp.DiagnosticOptions = type("DiagnosticOptions", (object,), {})'
check_env "TorsionalDiff" "torsional_diffusion" "[
    ('torch',           'import torch; assert torch.cuda.is_available(), \"CUDA not available\"'),
    ('torch_geometric', 'import torch_geometric'),
    ('torch_scatter',   'import torch_scatter'),
    ('numpy',           'import numpy'),
    ('rdkit',           'from rdkit import Chem'),
    ('e3nn',            'import e3nn'),
    ('yaml',            'import yaml'),
    ('tordiff_model',   'import sys; sys.path.insert(0, \"$TORDIFF_REPO\"); from utils.utils import get_model'),
]" "$TORDIFF_PREAMBLE"

# ── ETFlow ───────────────────────────────────────────────────────────────────
ETFLOW_REPO="$BASELINES_DIR/ETFlow"
check_env "ETFlow" "etflow" "[
    ('torch',           'import torch; assert torch.cuda.is_available(), \"CUDA not available\"'),
    ('torch_geometric', 'import torch_geometric'),
    ('numpy',           'import numpy'),
    ('rdkit',           'from rdkit import Chem'),
    ('lightning',       'import lightning'),
    ('datamol',         'import datamol'),
    ('etflow_model',    'import sys; sys.path.insert(0, \"$ETFLOW_REPO\"); from etflow.models.model import BaseFlow'),
]"
