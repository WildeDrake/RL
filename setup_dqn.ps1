# Descargar:
# https://www.python.org/downloads/release/python-3110

# Para ejecutar script, abrir PowerShell y ejecutar:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .\setup_dqn.ps1

# Crear entorno virtual con Python 3.11
python -m venv env
.\env\Scripts\Activate.ps1

# Actualizar pip
python -m pip install --upgrade pip setuptools wheel

# Instalar PyTorch (CUDA 12.1 o superior)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar Gymnasium con soporte Atari (ALE)
pip install gymnasium[atari,accept-rom-license]

# Instalar utilidades comunes
pip install numpy pillow matplotlib tensorboard tqdm opencv-python

# Verificar instalaciones
python -c "import torch, gymnasium, numpy; print('Torch:', torch.__version__, torch.version.cuda); print('Gymnasium:', gymnasium.__version__); print('Numpy:', numpy.__version__)"
