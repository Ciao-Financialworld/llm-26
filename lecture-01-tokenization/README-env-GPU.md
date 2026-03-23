# GPU Environment Setup for NVIDIA 5090 (Blackwell Architecture)

> **Note:** This is a more robust alternative to the original `environment-gpu.yml` for setting up the course environment.
> It avoids conda dependency resolution issues by keeping the conda package list minimal and installing PyTorch separately via pip.
> If you are **not** using a Blackwell-architecture GPU (e.g. RTX 5090), simply replace the PyTorch install command in Step 2 with the appropriate CUDA version from [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).

## Why this modified setup?

The original `environment-gpu.yml` uses conda to install PyTorch with CUDA, which has two problems:

1. **Conda often hangs** when resolving the full dependency tree.
2. **NVIDIA 5090 (Blackwell / SM 120)** requires CUDA 12.8+ and a PyTorch build that supports the new architecture. The default conda PyTorch package does not support this yet.

This modified approach uses conda only for basic Python packages, then installs PyTorch manually via pip with the `cu128` (CUDA 12.8) wheel — which properly supports the Blackwell architecture.

## Setup Steps

### 1. Create the conda environment

```bash
conda env create -f environment-gpu.yml
conda activate llm-26-gpu
```

### 2. Install PyTorch with CUDA 12.8 support (for 5090)

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

### 3. Download spaCy language models

```bash
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

### 4. Install Ollama CLI (for local LLM inference)

The `ollama-python` package in the conda environment is only the Python client library.
To actually run local models (e.g. `ollama serve`, `ollama run llama3`), you need to install the Ollama CLI separately.

#### Option A: With sudo privileges

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This installs Ollama system-wide to `/usr/local` and sets up a systemd service.

#### Option B: Without sudo privileges (manual install)

Download the release binary and install it to your home directory:

```bash
mkdir -p ~/opt/ollama ~/bin
curl -L -o /tmp/ollama-linux-amd64.tar.zst \
  https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
tar -C ~/opt/ollama -xf /tmp/ollama-linux-amd64.tar.zst
ln -sf ~/opt/ollama/bin/ollama ~/bin/ollama
rm -f /tmp/ollama-linux-amd64.tar.zst
```

Then add `~/bin` to your PATH (only needed once):

```bash
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

To start the Ollama server manually (since there is no systemd service in this mode):

```bash
ollama serve    # run in a separate terminal, keep it running
```

## Verification

Run the following commands to verify that everything is set up correctly:

**Check PyTorch and CUDA:**

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Expected output (example):

```
torch= 2.x.x+cu128
cuda= 12.8
available= True
gpu= NVIDIA GeForce RTX 5090
```

**Check spaCy models:**

```bash
python -c "import spacy; spacy.load('en_core_web_sm'); print('en ok')"
python -c "import spacy; spacy.load('zh_core_web_sm'); print('zh ok')"
```

**Check Ollama:**

```bash
ollama --version
python -c "import ollama; print('ollama-python ok')"
```

## Tested Environment

- GPU: NVIDIA GeForce RTX 5090
- Architecture: Blackwell (SM 120)
- CUDA: 12.8
- Python: 3.12
- OS: Ubuntu Linux
