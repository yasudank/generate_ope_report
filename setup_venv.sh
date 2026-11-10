#!/bin/bash
set -e

# setup_venv.sh: Create Python virtual environment and install dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "=========================================================="
echo " Setting up Python virtual environment for OPE Reporter"
echo "=========================================================="

# Check Python version
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "Error: Python 3 was not found in PATH."
    exit 1
fi

PY_VER=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python: $($PYTHON_CMD --version) at $(which $PYTHON_CMD)"

# Create virtual environment if not already present
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    $PYTHON_CMD -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at ${VENV_DIR}."
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade pip and install requirements
echo "Installing / upgrading dependencies..."
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "=========================================================="
echo " Setup complete!"
echo " To activate the virtual environment, run:"
echo "   source .venv/bin/activate"
echo ""
echo " To run the report generator:"
echo "   python generate_ope_report.py <ope_file> -d <pfs_designs_dir>"
echo "=========================================================="

