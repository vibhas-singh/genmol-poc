#!/usr/bin/env bash
set -euo pipefail

START_DIR="$(pwd)"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="genmol"

echo "Starting setup from: $START_DIR"

# ----------------------------------------
# 1. Install Miniconda if not installed
# ----------------------------------------
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "Miniconda not found. Installing..."

    INSTALLER="$HOME/miniconda.sh"

    if command -v wget >/dev/null 2>&1; then
        wget -q \
            https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            -O "$INSTALLER"
    elif command -v curl >/dev/null 2>&1; then
        curl -L \
            https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            -o "$INSTALLER"
    else
        echo "ERROR: Neither wget nor curl is available."
        exit 1
    fi

    bash "$INSTALLER" -b -p "$CONDA_DIR"
    rm -f "$INSTALLER"

    echo "Miniconda installed at: $CONDA_DIR"
else
    echo "Miniconda already installed."
fi

# ----------------------------------------
# 2. Load Conda
# ----------------------------------------
source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "Conda version:"
conda --version

# Make conda available in future bash sessions
conda init bash >/dev/null 2>&1 || true

# ----------------------------------------
# 3. Accept Anaconda Terms of Service
# ----------------------------------------
echo "Accepting Anaconda Terms of Service..."

conda tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/main

conda tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/r

# ----------------------------------------
# 4. Go to GenMol repository
# ----------------------------------------
cd "$START_DIR/genmol"

echo "Working directory: $(pwd)"

# ----------------------------------------
# 5. Create Conda environment
# ----------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating Conda environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# ----------------------------------------
# 6. Activate environment
# ----------------------------------------
conda activate "$ENV_NAME"

echo "Using Python:"
which python
python --version

# ----------------------------------------
# 7. Install GenMol dependencies
# ----------------------------------------
echo "Installing requirements..."
pip install -r env/requirements.txt

echo "Installing GenMol..."
pip install -e .

echo "Installing scikit-learn..."
pip install scikit-learn==1.2.2
pip install "setuptools<82" # Fix W&B dependency
pip install fcd_torch


# ----------------------------------------
# 8. Return to original/base directory
# ----------------------------------------
cd "$START_DIR"

echo
echo "========================================"
echo "GenMol setup complete."
echo "Conda environment: $ENV_NAME"
echo "Returned to: $(pwd)"
echo "========================================"
echo
echo "On a GPU node run:"
echo "source ~/miniconda3/etc/profile.d/conda.sh"
echo "conda activate genmol"