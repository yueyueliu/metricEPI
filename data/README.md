# Data Directory

This directory contains preprocessed datasets for training MetricEPI models.

## GM12878 Dataset

The GM12878 dataset is provided for reproducing the results in the paper.

### Directory Structure

```
GM12878/
├── processed/                    # Preprocessed HDF5 shards (training data)
│   ├── metadata.json            # Dataset metadata
│   ├── shard_0000.h5           # Data shard 0
│   ├── shard_0001.h5           # Data shard 1
│   └── ...
└── GM12878.with_lengths.csv     # Original EPI pair annotations
```

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total samples | 44,313 |
| Enhancer length | 3,000 bp |
| Promoter length | 2,000 bp |
| Channels | 10 |
| Number of shards | 9 |

### CSV File Format

The `GM12878.with_lengths.csv` contains the original EPI pair annotations:

| Column | Description |
|--------|-------------|
| `enhancer_chrom` | Enhancer chromosome |
| `enhancer_start` | Enhancer start position |
| `enhancer_end` | Enhancer end position |
| `promoter_chrom` | Promoter chromosome |
| `promoter_start` | Promoter start position |
| `promoter_end` | Promoter end position |
| `label` | Interaction label (0/1) |
| `enhancer_length` | Enhancer sequence length |
| `promoter_length` | Promoter sequence length |
| `enhancer_promoter_center_distance` | Distance between centers |

### HDF5 Data Format

Each HDF5 shard contains:

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `enhancer` | `(N, 3000, 10)` | float32 | Epigenetic signals at enhancer |
| `promoter` | `(N, 2000, 10)` | float32 | Epigenetic signals at promoter |
| `label` | `(N,)` | int32 | Binary label (0/1) |
| `distance` | `(N,)` | float32 | Genomic distance (bp) |

### Features (10 channels)

| Index | Feature | Description |
|:------|---------|-------------|
| 0-3   | DNA       | DNA sequence              |
| 4     | CTCF      | CTCF ChIP-seq signal      |
| 5     | DNase     | DNase I hypersensitivity  |
| 6     | H3K27ac   | H3K27ac ChIP-seq signal   |
| 7     | H3K4me1   | H3K4me1 ChIP-seq signal   |
| 8     | H3K4me3   | H3K4me3 ChIP-seq signal   |
| 9     | phastCons | Evolutionary conservation |

### Normalization

Signals are normalized using log1p + min-max scaling:
```
signal = log1p(raw_signal)
signal = signal / log1p(max_val)
```

Maximum values for each feature are recorded in `processed/metadata.json`.

## Using the Data

```bash
# Train with GM12878 dataset
python train.py --data-dir data/GM12878/processed

# Or in Python
from metricepi import get_train_val_dataloaders

train_loader, val_loader = get_train_val_dataloaders(
    data_dir='data/GM12878/processed',
    batch_size=32
)
```

## Data Sources

### Epigenetic Data
- ENCODE Consortium (GM12878 cell line)
- Reference genome: hg19

### EPI Annotations
EPI pairs were derived from published Hi-C and ChIA-PET datasets.
