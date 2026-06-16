#!/usr/bin/env python3
"""
Training script for MetricEPI

Usage:
    # Train with default GM12878 dataset
    python train.py

    # Train with custom data directory
    python train.py --data-dir path/to/processed/data

    # Train with custom parameters
    python train.py --epochs 100 --batch-size 32 --lr 1e-4
"""

import argparse
import os
import sys
import json
import random
from datetime import datetime

import numpy as np
import torch

from metricepi import (
    create_distance_aware_model_v2,
    create_trainer_v2,
    get_train_val_dataloaders,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MetricEPI model for enhancer-promoter interaction prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/GM12878/processed',
                        help='Path to sharded data directory')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--cache-size', type=int, default=2,
                        help='Number of shards to cache in memory')

    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler type')

    # Contrastive learning
    parser.add_argument('--use-contrastive', action='store_true', default=True,
                        help='Use contrastive learning loss')
    parser.add_argument('--no-contrastive', action='store_false', dest='use_contrastive',
                        help='Disable contrastive learning')
    parser.add_argument('--contrastive-weight', type=float, default=0.3,
                        help='Weight for contrastive loss')

    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output arguments
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save arguments
    args_path = os.path.join(output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("MetricEPI Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {args.data_dir}")

    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_split,
        num_workers=args.num_workers,
        cache_size=args.cache_size,
        seed=args.seed
    )

    # Load metadata
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\nDataset info:")
    print(f"  - Total samples: {metadata['total_samples']:,}")
    print(f"  - Enhancer length: {metadata['enhancer_length']} bp")
    print(f"  - Promoter length: {metadata['promoter_length']} bp")
    print(f"  - Channels: {metadata['num_channels']}")
    print(f"  - Features: {', '.join(metadata['feature_names'])}")

    print(f"\nTraining configuration:")
    print(f"  - Device: {args.device}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Hidden dim: {args.hidden_dim}")
    print(f"  - Num heads: {args.num_heads}")
    print(f"  - Num layers: {args.num_layers}")
    print(f"  - Contrastive learning: {args.use_contrastive}")
    print(f"  - Output: {output_dir}")
    print("=" * 60)

    # Create model
    print("\nCreating model...")
    model = create_distance_aware_model_v2(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=args.device
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Create trainer
    trainer = create_trainer_v2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        output_dir=output_dir,
        patience=args.patience,
        scheduler_type=args.scheduler,
        use_contrastive=args.use_contrastive,
        contrastive_weight=args.contrastive_weight
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_model(args.resume)

    # Train
    print("\nStarting training...")
    history = trainer.train(epochs=args.epochs, save_every=args.save_every)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation ROC-AUC: {trainer.best_val_roc_auc:.4f}")
    print(f"Model saved to: {trainer.best_model_path}")
    print("=" * 60)

    return history


if __name__ == '__main__':
    main()
