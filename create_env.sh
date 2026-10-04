# Ubuntu 22.04/24.04 on WSL

sudo apt-get update
sudo apt-get install -y \
    libglfw3 \
    libglew-dev \
    libgl1 \
    libglx-mesa0 \
    libosmesa6-dev \
    libegl1 \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd0 \
    build-essential \
    cmake \
    git \
    wget \
    curl

conda create -y -n lerobot python=3.12

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate lerobot

echo "==> Installing native cmake + egl-probe via conda-forge"

# libero requires egl-probe and hf-egl-probe, where both do not install because of libero's pyproject.toml
# so installing them on the side like this was the solution
conda install -y -c conda-forge cmake=3.31.6 egl-probe
pip install --no-build-isolation hf-egl-probe

LEROBOT_DIR="${LEROBOT_DIR:-$HOME/lerobot}"

git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"

cd "$LEROBOT_DIR"
pip install -e ".[smolvla]"
pip install -e ".[libero]"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

pip install mediapipe
pip install opencv-python

echo "Done! Run 'conda activate lerobot && source env.sh' to activate the env"
echo "Run 'make test' to see if everything is correctly installed and to download the SmolVLA model too"