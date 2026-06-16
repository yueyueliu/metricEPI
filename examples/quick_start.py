#!/usr/bin/env python3
"""
Quick Start Example for MetricEPI

This example demonstrates how to:
1. Load preprocessed EPI data
2. Create a distance-aware model
3. Train the model
4. Make predictions

Usage:
    python examples/quick_start.py
"""

import os
import sys
import torch

# Add project root to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metricepi import (
    DistanceAwareEPIModelV2,
    create_distance_aware_model_v2,
    TrainerV2,
    create_trainer_v2,
    ShardedEPIDataset,
    get_train_val_dataloaders,
)


def example_model_forward():
    """Example 1: Basic model forward pass with synthetic data."""
    print("=" * 60)
    print("Example 1: Model Forward Pass (Synthetic Data)")
    print("=" * 60)

    # Create model
    model = create_distance_aware_model_v2(
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        device='cpu'
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Create synthetic input
    batch_size = 4
    enhancer_length = 3000
    promoter_length = 2000
    num_channels = 10

    enhancer = torch.randn(batch_size, enhancer_length, num_channels)
    promoter = torch.randn(batch_size, promoter_length, num_channels)
    distance = torch.tensor([1000.0, 50000.0, 200000.0, 800000.0])
    labels = torch.tensor([1, 1, 0, 0])

    # Forward pass
    result = model(enhancer, promoter, distance, labels=labels, return_loss=True)

    print(f"\nInput shapes:")
    print(f"  enhancer: {enhancer.shape}")
    print(f"  promoter: {promoter.shape}")
    print(f"  distance: {distance.shape}")

    print(f"\nOutput:")
    print(f"  logits: {result['logits'].shape}")
    print(f"  proba: {result['proba'].squeeze().tolist()}")

    print(f"\nDistance encoding:")
    print(f"  decay_weight: {result['decay_weight'].tolist()}")
    print(f"  scale_idx: {result['scale_idx'].tolist()}")
    print(f"  (Scale meaning: 0=<10kb, 1=10-100kb, 2=100kb-1Mb, 3=>1Mb)")

    if 'loss_dict' in result:
        print(f"\nContrastive Learning Loss:")
        print(f"  total: {result['loss_dict']['total'].item():.4f}")
        print(f"  contrast: {result['loss_dict']['contrast'].item():.4f}")
        print(f"  distance: {result['loss_dict']['distance'].item():.4f}")

    print("\nExample 1 completed!\n")


def example_load_data():
    """Example 2: Load and inspect preprocessed data."""
    print("=" * 60)
    print("Example 2: Load Preprocessed Data")
    print("=" * 60)

    data_dir = 'data/GM12878/processed'

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please ensure the GM12878 dataset is available.")
        return None, None

    # Create data loaders
    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=data_dir,
        batch_size=16,
        val_ratio=0.2,
        num_workers=0,
        cache_size=2,
        seed=42
    )

    # Inspect a batch
    print(f"\nDataset loaded successfully!")
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Val samples: {len(val_loader.dataset):,}")

    # Get a batch
    for batch in train_loader:
        if len(batch) == 4:
            enhancer, promoter, labels, distances = batch
        else:
            enhancer, promoter, labels = batch
            distances = None

        print(f"\nBatch shape:")
        print(f"  enhancer: {enhancer.shape}")
        print(f"  promoter: {promoter.shape}")
        print(f"  labels: {labels.shape}")
        if distances is not None:
            print(f"  distances: {distances.shape}")
            print(f"  distance range: [{distances.min():.0f}, {distances.max():.0f}] bp")
        break

    print("\nExample 2 completed!\n")
    return train_loader, val_loader


def example_train_model(train_loader, val_loader):
    """Example 3: Train the model."""
    if train_loader is None:
        print("Skipping Example 3: No data available")
        return

    print("=" * 60)
    print("Example 3: Train Model (2 epochs demo)")
    print("=" * 60)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model
    model = create_distance_aware_model_v2(
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        device=device
    )

    # Create trainer
    trainer = create_trainer_v2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-4,
        weight_decay=1e-4,
        device=device,
        output_dir='./outputs/quick_start',
        patience=10,
        use_contrastive=True
    )

    # Train for a few epochs (demo)
    print("\nStarting training (2 epochs for demo)...")
    trainer.train(epochs=2, save_every=1)

    print(f"\nBest validation ROC-AUC: {trainer.best_val_roc_auc:.4f}")
    print("\nExample 3 completed!\n")


def example_prediction():
    """Example 4: Make predictions with a trained model."""
    print("=" * 60)
    print("Example 4: Make Predictions")
    print("=" * 60)

    # Check for saved model
    model_path = './outputs/quick_start/best_model.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    model = create_distance_aware_model_v2(device=device)

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No saved model found, using untrained model for demonstration")

    model.eval()

    # Create test input
    batch_size = 5
    enhancer = torch.randn(batch_size, 3000, 10)
    promoter = torch.randn(batch_size, 2000, 10)
    distances = torch.tensor([5000.0, 50000.0, 150000.0, 500000.0, 1000000.0])

    # Predict
    with torch.no_grad():
        proba = model.predict_proba(enhancer.to(device), promoter.to(device), distances.to(device))

    print("\nPrediction results:")
    print(f"{'Distance (kb)':<15} {'Probability':<12} {'Scale'}")
    print("-" * 45)

    scale_labels = ['<10kb', '10-100kb', '100kb-1Mb', '>1Mb']
    for i, (d, p) in enumerate(zip(distances.tolist(), proba.squeeze().tolist())):
        if d < 10000:
            scale = scale_labels[0]
        elif d < 100000:
            scale = scale_labels[1]
        elif d < 1000000:
            scale = scale_labels[2]
        else:
            scale = scale_labels[3]

        print(f"{d/1000:<15.1f} {p:<12.4f} {scale}")

    print("\nExample 4 completed!\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MetricEPI Quick Start Examples")
    print("=" * 60 + "\n")

    # Example 1: Model forward pass
    example_model_forward()

    # Example 2: Load data
    train_loader, val_loader = example_load_data()

    # Example 3: Train model
    example_train_model(train_loader, val_loader)

    # Example 4: Make predictions
    example_prediction()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Train full model: python train.py --data-dir data/GM12878/processed")
    print("2. See README.md for full documentation")
    print("3. Explore model architecture in metricepi/model.py")


if __name__ == '__main__':
    main()