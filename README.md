# 🎨 FLUX Kontext + LoRA Studio

An advanced AI-powered image editing application built with FLUX Kontext diffusion model, featuring intelligent reference modes and LoRA style support.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

### 🧠 Smart Reference Mode
Automatically detects transformation type from your prompt and applies the appropriate enhancement:

- **👗 Clothing Swap**: "Dress the woman in this outfit" → Auto-detects clothing replacement
- **🤲 Object Placement**: "Make him hold this item" → Auto-detects object placement  
- **🌅 Background Change**: "Change background to this scene" → Auto-detects environment swap
- **💇 Appearance Modification**: "Give her this hairstyle" → Auto-detects appearance change
- **🕺 Pose Transfer**: "Make her pose like this" → Auto-detects pose transfer
- **🎨 Style Transfer**: "Apply this artistic style" → Auto-detects style transfer

### 🔧 Additional Features
- **LoRA Support**: Dynamic loading of custom style LoRAs
- **Memory Optimized**: Smart GPU memory management with CPU offloading
- **Multiple Combination Modes**: Single, Side-by-side, Reference
- **Performance Controls**: Adjustable image sizes (512px-1536px)
- **Responsive UI**: Built with Gradio for easy use

## 🚀 Installation

### Prerequisites
- **Python**: 3.10+ 
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 24GB+ system RAM
- **Storage**: 64GB+ free space for model downloads
- **CUDA**: 12.4 (newest), 12.1, 11.8, or CPU-only support

### 🔍 Check Your CUDA Version
```bash
# Method 1: Check NVIDIA driver info
nvidia-smi

# Method 2: Check CUDA toolkit version
nvcc --version

# Method 3: Check current PyTorch CUDA version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')"
```

**Common CUDA Versions:**
- **CUDA 12.4** → `cu124` (Newest, best performance) ⭐
- **CUDA 12.1** → `cu121` (Good performance, stable)  
- **CUDA 11.8** → `cu118` (Older, widely compatible)
- **No GPU** → `cpu` (Slower, but works without NVIDIA GPU)

---

## 📦 Option 1: Conda Installation (Recommended)

### Step 1: Clone Repository
```bash
git clone https://github.com/ShmuelRonen/flux-kontext-lora-studio.git
cd flux-kontext-lora-studio
```

### Step 2: Create Conda Environment
```bash
# Create a new conda environment
conda create -n kontext python=3.10

# Activate environment
conda activate kontext
```

### Step 3: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# STEP 1: Install PyTorch with CUDA support (choose your CUDA version)
# For CUDA 12.4 (newest, best performance):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# STEP 2: Install remaining requirements
pip install -r requirements.txt
```

### Step 4: Run Application
```bash
python app.py
```

---

## 🐍 Option 2: Virtual Environment (venv)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/flux-kontext-lora-studio.git
cd flux-kontext-lora-studio
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# STEP 1: Install PyTorch with CUDA support (choose your CUDA version)
# Check your CUDA version first: nvidia-smi

# For CUDA 12.4 (newest, best performance):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older systems):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (no GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# STEP 2: Install remaining requirements
pip install -r requirements.txt
```

**🔍 Check Your CUDA Version:**
```bash
# Check CUDA version
nvidia-smi

# Or check current PyTorch CUDA version (after activating environment)
conda activate kontext
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')"
```

### Step 4: Run Application
```bash
python app.py
```

---

## 🎯 Quick Start Guide

### Basic Usage
1. **Launch** the application: 
   ```bash
   conda activate kontext  # or activate your venv
   python app.py
   ```
2. **Open** your browser to the displayed URL (usually `http://127.0.0.1:7860`)
3. **Upload** your main image
4. **Upload** reference image (optional)
5. **Select** combination mode
6. **Write** your prompt using smart keywords
7. **Click** "Process Images"

### Smart Prompt Examples

#### Clothing Transfer
```
"Dress the woman in this elegant gown"
"Change her outfit to this casual style"
```

#### Object Placement
```
"Make him hold this guitar"
"Give her this bouquet of flowers"
```

#### Background Change
```
"Change the background to this cityscape"
"Place her in this forest setting"
```

#### Style Transfer
```
"Apply this painting style to the photo"
"Transform this into watercolor art"
```

## 🎨 LoRA Management

### Setup LoRA Files
1. Create `AVAILABLE_LORAS/` folder in your project directory
2. Add your `.safetensors` LoRA files to this folder
3. Click the refresh button (🔄) in the UI
4. Select LoRA from dropdown and adjust strength (0.0-2.0)

### Finding LoRAs
- [Civitai](https://civitai.com/) - Community LoRAs
- [Hugging Face](https://huggingface.co/models?other=lora) - Official LoRAs

## ⚙️ Performance Settings

### Image Size vs Speed
- **512px**: Ultra fast (1-2 min) ⚡ - Good for testing
- **768px**: Balanced (2-3 min) ⚖️ - Recommended  
- **1024px+**: High quality (4-6 min) 🏆 - Best results

### Advanced Parameters
- **Guidance Scale**: 2.5-4.0 (higher = stronger prompt following)
- **Steps**: 15-25 (more steps = better quality, but slower)
- **Seed**: Set for reproducible results

## 🔧 Troubleshooting

### GPU Memory Issues
```bash
# If you get CUDA out of memory errors:
# 1. Reduce max image size to 512px
# 2. Close other GPU applications
# 3. Restart the application
```

### Model Download Issues
- Ensure stable internet connection
- Check Hugging Face Hub access: `huggingface-cli login`
- Verify ~8GB free storage space

### LoRA Loading Errors
- Verify files are in `.safetensors` format
- Check file permissions and paths
- Ensure LoRA is compatible with FLUX architecture

### Installation Issues
```bash
# If diffusers installation fails:
pip install git+https://github.com/huggingface/diffusers.git --force-reinstall

# Optional: xformers for better memory efficiency (may cause PyTorch conflicts)
# Only install if you experience memory issues:
# pip install xformers --index-url https://download.pytorch.org/whl/cu118
```

### Memory Optimization
The application includes built-in memory optimizations:
- **Sequential CPU offloading** - Automatically enabled
- **VAE slicing** - Reduces memory usage
- **Attention slicing** - Prevents OOM errors
- **Smart memory management** - Clears cache between operations

**Note**: xformers is not included by default to avoid PyTorch conflicts.

## 📁 Project Structure

```
flux-kontext-lora-studio/
├── app.py                    # Main application
├── requirements.txt          # pip dependencies
├── start.bat                # Windows startup script
├── AVAILABLE_LORAS/         # LoRA files directory
│   └── your_lora.safetensors
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT license
└── README.md               # This file
```

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature-amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature-amazing-feature`
5. **Open** a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[FLUX.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)** by Black Forest Labs
- **[Diffusers](https://github.com/huggingface/diffusers)** by Hugging Face  
- **[Gradio](https://gradio.app/)** for the amazing UI framework
- **[Spaces](https://huggingface.co/spaces)** for GPU acceleration support

## 📞 Support & Community

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/yourusername/flux-kontext-lora-studio/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/yourusername/flux-kontext-lora-studio/discussions)
- **📚 Documentation**: Check code comments and this README
- **🎨 Share Results**: Tag us in your creations!

## ⭐ Star History

If this project helped you, please consider giving it a star! ⭐

---

<div align="center">
<strong>Made with ❤️ for the AI art community</strong>
<br>
<em>Transform images with the power of AI and creativity</em>
</div>

---

<div align="center">
<strong>Made with ❤️ for the AI art community</strong>
<br>
<em>Transform images with the power of AI and creativity</em>
</div>
