# MetricEPI: Distance-Aware Deep Learning for Enhancer-Promoter Interaction Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A distance-aware deep learning model for predicting enhancer-promoter interactions (EPIs) using multi-scale genomic features.

## Key Features

- **Distance-Aware Attention**: Novel attention mechanism that explicitly incorporates genomic distance into the attention computation
- **Multi-Scale Distance Encoding**: Captures interaction patterns at different genomic scales
- **Contrastive Learning with Distance Constraints**: Distance-guided contrastive learning for better representation learning
- **Multi-Scale CNN Feature Extraction**: Extracts features from epigenetic signals at multiple resolutions

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/MetricEPI.git
cd MetricEPI

# Install as package
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (recommended for GPU acceleration)

## Quick Start

### 1. Train with Provided GM12878 Dataset

```bash
python train.py
```

### 2. Python API Usage

```python
from metricepi import (
    create_distance_aware_model_v2,
    create_trainer_v2,
    get_train_val_dataloaders,
)

# Load data
train_loader, val_loader = get_train_val_dataloaders(
    data_dir='data/GM12878/processed',
    batch_size=32,
    val_ratio=0.2
)

# Create model
model = create_distance_aware_model_v2(
    hidden_dim=128,
    num_heads=4,
    num_layers=2,
    device='cuda'
)

# Train
trainer = create_trainer_v2(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=1e-4,
    device='cuda'
)

trainer.train(epochs=100)
```

### 3. Make Predictions

```python
import torch
from metricepi import create_distance_aware_model_v2

# Load model
model = create_distance_aware_model_v2(device='cuda')
checkpoint = torch.load('outputs/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    enhancer = torch.randn(1, 3000, 10).cuda()  # (batch, length, channels)
    promoter = torch.randn(1, 2000, 10).cuda()
    distance = torch.tensor([50000.0]).cuda()   # 50kb

    proba = model.predict_proba(enhancer, promoter, distance)
    print(f"Interaction probability: {proba.item():.4f}")
```

## Data Format

### Preprocessed Data Structure

```
data/
└── GM12878/
    └── processed/
        ├── metadata.json      # Dataset metadata
        ├── shard_0000.h5     # HDF5 data shards
        ├── shard_0001.h5
        └── ...
```

### HDF5 File Contents

| Field | Shape | Description |
|-------|-------|-------------|
| `enhancer` | `(N, 3000, 10)` | Epigenetic signals at enhancer regions |
| `promoter` | `(N, 2000, 10)` | Epigenetic signals at promoter regions |
| `label` | `(N,)` | Binary labels (1: interaction, 0: no interaction) |
| `distance` | `(N,)` | Genomic distance in base pairs |

### Input Features (10 channels)

| Channel | Feature | Description |
|---------|---------|-------------|
| 4-9 | Epigenetic marks | CTCF, DNase, H3K27ac, H3K4me1, H3K4me3, phastCons |

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MSML-EPI Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Enhancer E ∈ R^{L_e×C}    Promoter P ∈ R^{L_p×C}                  │
│         │                          │                                │
│         ▼                          ▼                                │
│  ┌─────────────────────────────────────────────────┐               │
│  │           Multi-Scale Encoder (共享权重)         │               │
│  │  ┌─────────────────────────────────────────────┐│               │
│  │  │ Multi-Scale CNN (k=3,5,7,11) + Transformer  ││               │
│  │  └─────────────────────────────────────────────┘│               │
│  └─────────────────────────────────────────────────┘               │
│         │                          │                                │
│         ▼                          ▼                                │
│      E' ∈ R^{L'_e×D}          P' ∈ R^{L'_p×D}                      │
│         │                          │                                │
│         └──────────┬───────────────┘                                │
│                    │                                                │
│                    ▼                                                │
│         Distance d = |pos_e - pos_p|                                │
│                    │                                                │
│                    ▼                                                │
│  ┌─────────────────────────────────────────────────┐               │
│  │         Distance Encoder                         │               │
│  │  Multi-Scale Distance Encoding + Decay Weight    │               │
│  └─────────────────────────────────────────────────┘               │
│                    │                                                │
│                    ▼                                                │
│           distance_feat ∈ R^D                                       │
│                    │                                                │
│                    ▼                                                │
│  ┌─────────────────────────────────────────────────┐               │
│  │      Distance-Aware Cross Attention              │               │
│  │  ┌─────────────────┐   ┌─────────────────┐      │               │
│  │  │ E→P Attention   │   │ P→E Attention   │      │               │
│  │  │ + distance bias │   │ + distance bias │      │               │
│  │  └─────────────────┘   └─────────────────┘      │               │
│  └─────────────────────────────────────────────────┘               │
│                    │                                                │
│                    ▼                                                │
│           interaction_feat ∈ R^D                                    │
│                    │                                                │
│                    ▼                                                │
│  ┌─────────────────────────────────────────────────┐               │
│  │         Classification Head                      │               │
│  │         MLP → P(interaction)                      │               │
│  └─────────────────────────────────────────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Innovations

1. **Distance-Aware Attention**: Distance bias is directly added to attention scores, not simply concatenated as a feature
2. **Multi-Scale Distance Encoding**: Different scales capture different biological interaction patterns
3. **Contrastive Learning**: Distance-guided contrastive loss improves representation learning

## Training Options

```bash
python train.py \
    --data-dir data/GM12878/processed \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --hidden-dim 128 \
    --num-heads 4 \
    --num-layers 2 \
    --output ./outputs/experiment1
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | data/GM12878/processed | Path to processed data |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--hidden-dim` | 128 | Hidden dimension |
| `--num-heads` | 4 | Attention heads |
| `--num-layers` | 2 | Transformer layers |
| `--use-contrastive` | True | Enable contrastive learning |
| `--patience` | 15 | Early stopping patience |

## Project Structure

```
MetricEPI/
├── metricepi/                 # Python package
│   ├── __init__.py
│   ├── model.py              # Model architecture
│   ├── trainer.py            # Training logic
│   └── data_loader.py        # Data loading utilities
├── data/                      # Data directory
│   └── GM12878/
│       └── processed/        # Preprocessed HDF5 files
├── train.py                   # Training script
├── examples/
│   └── quick_start.py        # Example usage
├── configs/
│   └── default.yaml          # Configuration
├── tests/
│   └── test_model.py         # Unit tests
├── README.md
├── REPRODUCIBILITY.md        # Reproducibility guide
├── requirements.txt
└── setup.py
```

## Extending the Model

### Adding New Cell Types

To use your own EPI data, prepare:

1. **EPI pair CSV file** with columns:
   - `enhancer_chr`, `enhancer_start`, `enhancer_end`
   - `promoter_chr`, `promoter_start`, `promoter_end`
   - `label` (0 or 1)
   - `distance` (optional)

2. **Epigenetic signal matrix** (N, L, 10) for enhancer and promoter regions

3. Save as HDF5 files following the format in `data/GM12878/processed/`

## Citation

If you use this code in your research, please cite:

```bibtex
~ to do
```

## Contact

For questions or issues, please open a GitHub issue.
