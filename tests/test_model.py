#!/usr/bin/env python3
"""
Unit tests for MetricEPI model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from metricepi import DistanceAwareEPIModelV2, create_distance_aware_model_v2


def test_model_creation():
    """Test model can be created"""
    model = create_distance_aware_model_v2(
        hidden_dim=64,
        num_heads=2,
        num_layers=1,
        device='cpu'
    )
    assert model is not None


def test_forward_pass():
    """Test forward pass works correctly"""
    model = create_distance_aware_model_v2(
        hidden_dim=64,
        num_heads=2,
        num_layers=1,
        device='cpu'
    )

    batch_size = 4
    enhancer = torch.randn(batch_size, 3000, 10)
    promoter = torch.randn(batch_size, 2000, 10)
    distance = torch.rand(batch_size) * 100000

    result = model(enhancer, promoter, distance)

    assert result['logits'].shape == (batch_size, 1)
    assert result['proba'].shape == (batch_size, 1)
    assert torch.all(result['proba'] >= 0) and torch.all(result['proba'] <= 1)


def test_forward_with_labels():
    """Test forward pass with labels returns loss"""
    model = create_distance_aware_model_v2(
        hidden_dim=64,
        num_heads=2,
        num_layers=1,
        device='cpu'
    )

    batch_size = 4
    enhancer = torch.randn(batch_size, 3000, 10)
    promoter = torch.randn(batch_size, 2000, 10)
    distance = torch.rand(batch_size) * 100000
    labels = torch.tensor([1, 0, 1, 0])

    result = model(enhancer, promoter, distance, labels=labels, return_loss=True)

    assert 'loss_dict' in result
    assert 'total' in result['loss_dict']
    assert 'contrast' in result['loss_dict']


def test_gradient_flow():
    """Test gradients flow correctly"""
    model = create_distance_aware_model_v2(
        hidden_dim=64,
        num_heads=2,
        num_layers=1,
        device='cpu'
    )

    enhancer = torch.randn(2, 3000, 10, requires_grad=True)
    promoter = torch.randn(2, 2000, 10, requires_grad=True)
    distance = torch.rand(2) * 100000

    result = model(enhancer, promoter, distance)
    loss = result['logits'].sum()
    loss.backward()

    assert enhancer.grad is not None
    assert promoter.grad is not None


def test_distance_encoding():
    """Test distance encoding produces correct scales"""
    model = create_distance_aware_model_v2(device='cpu')

    # Test different distance scales
    distances = torch.tensor([5000.0, 50000.0, 500000.0, 2000000.0])
    enhancer = torch.randn(4, 3000, 10)
    promoter = torch.randn(4, 2000, 10)

    result = model(enhancer, promoter, distances)

    # Check scale indices
    # Scale 0: <10kb, Scale 1: 10-100kb, Scale 2: 100kb-1Mb, Scale 3: >1Mb
    expected_scales = [0, 1, 2, 3]
    assert result['scale_idx'].tolist() == expected_scales


def test_predict_proba():
    """Test predict_proba method"""
    model = create_distance_aware_model_v2(device='cpu')
    model.eval()

    enhancer = torch.randn(2, 3000, 10)
    promoter = torch.randn(2, 2000, 10)
    distance = torch.tensor([10000.0, 100000.0])

    with torch.no_grad():
        proba = model.predict_proba(enhancer, promoter, distance)

    assert proba.shape == (2, 1)
    assert torch.all(proba >= 0) and torch.all(proba <= 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
