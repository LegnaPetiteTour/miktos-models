# 🤖 Miktos AI Models - Model Management & Optimization

> **Centralized AI model management for Miktos** - Download, organize, optimize, and distribute AI models for 3D creation workflows. Supports Stable Diffusion, FLUX, ControlNet, LoRA, and custom models.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Models](https://img.shields.io/badge/Models-50+-green.svg)](./models)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Compatible-yellow.svg)](https://huggingface.co/)

## 🚀 Overview

Miktos AI Models provides a unified system for managing AI models used in 3D creation workflows:

- **Model Registry**: Curated list of compatible models
- **Automatic Download**: One-command model installation
- **Optimization Tools**: Convert and optimize models for faster inference
- **Version Control**: Track model versions and updates
- **LoRA Management**: Organize and apply custom LoRA models
- **Distribution System**: Share models with the community

## 📚 Supported Model Types

### 🎨 Image Generation
- **Stable Diffusion XL**: High-quality image generation
- **FLUX**: Next-generation diffusion models
- **SD 1.5**: Legacy support for older workflows
- **Custom Checkpoints**: Community-trained models

### 🎮 Control Models
- **ControlNet**: Guided generation with various conditions
- **IP-Adapter**: Image prompt adaptation
- **T2I-Adapter**: Text-to-image control
- **OpenPose**: Human pose control

### 🎯 Specialized Models
- **LoRA**: Low-rank adaptations for style/concept
- **LyCORIS**: Advanced LoRA variants
- **Textual Inversion**: Custom embeddings
- **Hypernetworks**: Dynamic network modifications

### 🏗️ 3D Generation
- **Shap-E**: Text/image to 3D mesh
- **Point-E**: Point cloud generation
- **Zero123**: Single image to 3D
- **DreamGaussian**: Fast 3D generation

## 🛠️ Installation

### Quick Start
```bash
# Clone the repository
git clone https://github.com/LegnaPetiteTour/miktos-models.git
cd miktos-models

# Install model manager
pip install -e .

# Download essential models
python download_models.py --preset essential

# Or download specific models
python download_models.py --model stable-diffusion-xl
```

### Model Presets
```bash
# Essential models for basic workflows (~10GB)
python download_models.py --preset essential

# Professional set with advanced models (~50GB)
python download_models.py --preset professional

# Complete collection (~150GB)
python download_models.py --preset complete

# Custom selection
python download_models.py --config my_models.yaml
```

## 📁 Directory Structure

```
miktos-models/
├── registry/           # Model registry and metadata
│   ├── official.yaml   # Official model list
│   ├── community.yaml  # Community models
│   └── custom.yaml     # Your custom models
│
├── models/             # Model storage
│   ├── stable-diffusion/
│   │   ├── sdxl/       # SDXL models
│   │   ├── sd15/       # SD 1.5 models
│   │   └── custom/     # Custom checkpoints
│   ├── controlnet/     # Control models
│   ├── lora/           # LoRA models
│   ├── embeddings/     # Textual inversions
│   └── 3d/             # 3D generation models
│
├── tools/              # Model management tools
│   ├── download.py     # Model downloader
│   ├── optimize.py     # Model optimizer
│   ├── convert.py      # Format converter
│   └── validate.py     # Model validator
│
└── configs/            # Configuration files
    ├── presets/        # Model preset definitions
    └── optimization/   # Optimization configs
```

## 🎯 Model Registry

### Browse Available Models
```python
from miktos_models import ModelRegistry

registry = ModelRegistry()

# List all available models
models = registry.list_models()

# Search by category
sd_models = registry.search(category="stable-diffusion")

# Get model info
model_info = registry.get_model("stable-diffusion-xl-base")
print(f"Size: {model_info.size}")
print(f"Requirements: {model_info.requirements}")
```

### Featured Models

#### 🌟 Stable Diffusion XL Base
- **Size**: 6.94GB
- **Use**: High-quality 1024x1024 generation
- **RAM**: 8GB+ VRAM recommended

#### 🌟 FLUX.1 Dev
- **Size**: 23.8GB
- **Use**: Next-gen image generation
- **RAM**: 24GB+ VRAM recommended

#### 🌟 ControlNet Depth
- **Size**: 1.45GB
- **Use**: Depth-guided generation
- **Pairs with**: Any SD model

#### 🌟 Realistic Vision LoRA
- **Size**: 144MB
- **Use**: Photorealistic enhancement
- **Strength**: 0.6-0.8 recommended

## 🔧 Model Management

### Download Models
```python
from miktos_models import ModelManager

manager = ModelManager()

# Download single model
manager.download("stable-diffusion-xl-base")

# Download with progress callback
def progress(current, total):
    print(f"Progress: {current}/{total} MB")

manager.download("flux-dev", progress_callback=progress)

# Batch download
models = ["controlnet-canny", "controlnet-depth", "ip-adapter"]
manager.download_batch(models)
```

### Optimize Models
```python
# Convert to optimized format
manager.optimize_model(
    model="stable-diffusion-xl",
    optimization="onnx",  # or "tensorrt", "coreml"
    precision="fp16"      # or "int8", "fp32"
)

# Quantize for smaller size
manager.quantize_model(
    model="stable-diffusion-xl",
    bits=8,  # 8-bit quantization
    calibration_data="calibration_images/"
)
```

## 🎨 LoRA Management

### Organize LoRAs
```python
from miktos_models import LoRAManager

lora_manager = LoRAManager()

# Install LoRA
lora_manager.install(
    path="path/to/lora.safetensors",
    category="style",
    tags=["anime", "colorful"]
)

# Search LoRAs
anime_loras = lora_manager.search(tags=["anime"])

# Apply multiple LoRAs
lora_manager.create_preset(
    name="my_style",
    loras=[
        ("anime_style", 0.7),
        ("detailed_eyes", 0.5),
        ("vivid_colors", 0.3)
    ]
)
```

---

**Miktos AI Models** - Your Gateway to AI-Powered Creation 🚀

*One command. Endless possibilities.*