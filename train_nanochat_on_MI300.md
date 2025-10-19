---

## 🧠 How to Train NanoChat on AMD MI300 (DigitalOcean)

<img width="2630" height="1950" alt="nanochat" src="https://github.com/user-attachments/assets/9948ac02-39c4-4ba9-9435-a3dd0062578a" />

### 1️⃣ Create a GPU Droplet

Go to **[AMD DigitalOcean](https://amd.digitalocean.com/)** and create a new **GPU Droplet** with the **AMD MI300** GPU.
Choose **Ubuntu 22.04** as the operating system.

---

### 2️⃣ Set Up the Environment

```bash
# Clone NanoChat
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# Install dependencies
sudo apt update
sudo apt install -y python3.12-venv build-essential pkg-config libssl-dev

# Create and activate virtual environment
python3 -m venv venv
source ./venv/bin/activate
```

---

### 3️⃣ Install ROCm PyTorch and Packages

```bash
# Install PyTorch with ROCm 7.0
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0

# Upgrade pip tools
pip install --upgrade pip wheel setuptools

# Install Python dependencies
pip install numpy tqdm datasets sentencepiece wandb tokenizers pytest
```

---

### 4️⃣ Install Rust and Build RustBPE

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build helper
pip install maturin

# Compile Rust-based tokenizer
maturin develop --release --manifest-path rustbpe/Cargo.toml
```

---

### 5️⃣ Prepare Dataset and Tokenizer

```bash
# Generate a small sample dataset
python3 -m nanochat.dataset -n 2

# Tokenize data
python3 -m nanochat.tokenizer

# Train tokenizer on up to 2B characters
python3 -m scripts.tok_train --max_chars=2000000000
```

---

### 6️⃣ Verify the Build

```bash
# Run tests for RustBPE
python3 -m pytest tests/test_rustbpe.py -v -s
```

---

### 7️⃣ Train the Model

```bash
# Start training
python3 -m scripts.base_train --depth=12 --device_batch_size=1 --max_seq_len=1024
```

---

### ✅ Notes

* Check GPU usage with:

  ```bash
  rocm-smi
  ```
* Adjust parameters such as `--depth`, `--max_seq_len`, and `--device_batch_size` based on GPU memory.

---

### 🧩 Next Steps

1. **Train on MI300X ×8 GPUs**

   * Use PyTorch’s distributed mode or `torchrun` for multi-GPU scaling.
   * Example:

     ```bash
     torchrun --nproc_per_node=8 scripts/base_train.py --depth=12 --max_seq_len=1024
     ```

2. **Enable AMD Advanced Features**

   * Experiment with **Primus** and **Primus-Turbo** optimizations (for kernel fusion, memory tiling, and scheduling).
   * Monitor improvements in utilization and throughput.

---

