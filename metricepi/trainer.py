#!/usr/bin/env python3
"""
Trainer V2 for Distance-Aware EPI Model V2

Features:
- Contrastive learning loss
- Multi-loss weighting
- Focal Loss
- Hard example mining
- V2 model forward pass interface
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    raise ImportError("Please install scikit-learn: pip install scikit-learn")

from tqdm import tqdm


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where p_t = p if y=1, else 1-p
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N,) raw logits
            targets: (N,) binary targets
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)

        # p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # α_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal term
        focal_term = (1 - p_t) ** self.gamma

        # Cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Focal loss
        loss = alpha_t * focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class EarlyStopping:
    """Early stopping based on validation loss"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


class TrainerV2:
    """
    Trainer V2 for Distance-Aware EPI Model V2

    Features:
    - Contrastive learning loss
    - Multi-loss weighted fusion
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        output_dir: str = './outputs',
        patience: int = 10,
        use_contrastive: bool = True,
        contrastive_weight: float = 0.3,  # Weight for contrastive loss
        bce_weight: float = 0.7,  # Weight for BCE loss
        pos_weight: Optional[float] = None,  # Positive sample weight for class imbalance
        use_focal: bool = False,  # Whether to use Focal Loss
        focal_alpha: float = 0.25,  # Focal Loss alpha parameter
        focal_gamma: float = 2.0,  # Focal Loss gamma parameter
        use_ohem: bool = False,  # Whether to use Online Hard Example Mining
        ohem_ratio: float = 0.7  # OHEM keep ratio
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_ohem = use_ohem
        self.ohem_ratio = ohem_ratio

        os.makedirs(output_dir, exist_ok=True)

        # Loss function selection
        if use_focal:
            self.bce_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')
            print(f"Using Focal Loss with alpha={focal_alpha}, gamma={focal_gamma}")
        else:
            # BCE loss (with class weight)
            if pos_weight is not None:
                self.bce_criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([pos_weight], device=device),
                    reduction='none'
                )
                print(f"Using weighted BCE loss with pos_weight={pos_weight:.2f}")
            else:
                self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none')

        self.bce_criterion_mean = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device) if pos_weight is not None else None
        ) if not use_focal else None

        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode='max')

        # Training history
        self.history = {
            'train_loss': [],
            'train_bce_loss': [],
            'train_contrast_loss': [],
            'train_distance_loss': [],
            'val_loss': [],
            'train_roc_auc': [],
            'val_roc_auc': [],
            'train_prc_auc': [],
            'val_prc_auc': [],
            'lr': []
        }

        self.best_val_roc_auc = 0.0
        self.best_model_path = os.path.join(output_dir, 'best_model.pt')

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()

        total_loss = 0.0
        total_bce_loss = 0.0
        total_contrast_loss = 0.0
        total_distance_loss = 0.0
        all_labels = []
        all_probs = []

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            if len(batch) == 4:
                enhancer, promoter, labels, distances = batch
                distances = distances.to(self.device)
            else:
                enhancer, promoter, labels = batch
                distances = None

            enhancer = enhancer.to(self.device)
            promoter = promoter.to(self.device)
            labels = labels.float().to(self.device)

            self.optimizer.zero_grad()

            # V2 model forward pass
            if hasattr(self.model, 'use_contrastive'):
                # V2 model returns dict
                result = self.model(
                    enhancer, promoter, distances,
                    labels=labels,
                    return_loss=self.use_contrastive
                )
                logits = result['logits'].squeeze(-1)
                proba = result['proba'].squeeze(-1)

                # Compute classification loss (per sample)
                bce_loss_per_sample = self.bce_criterion(logits, labels)

                # Online Hard Example Mining (OHEM)
                if self.use_ohem:
                    sorted_loss, _ = torch.sort(bce_loss_per_sample, descending=True)
                    keep_num = max(1, int(len(sorted_loss) * self.ohem_ratio))
                    bce_loss = sorted_loss[:keep_num].mean()
                else:
                    bce_loss = bce_loss_per_sample.mean()

                # Contrastive learning loss
                if self.use_contrastive and 'loss_dict' in result:
                    contrast_loss = result['loss_dict']['contrast']
                    distance_loss = result['loss_dict']['distance']
                    loss = self.bce_weight * bce_loss + self.contrastive_weight * contrast_loss
                else:
                    contrast_loss = torch.tensor(0.0, device=self.device)
                    distance_loss = torch.tensor(0.0, device=self.device)
                    loss = bce_loss
            else:
                # Legacy model
                if hasattr(self.model, 'distance_encoder') and distances is not None:
                    logits = self.model(enhancer, promoter, distances).squeeze(-1)
                else:
                    logits = self.model(enhancer, promoter).squeeze(-1)

                # Compute classification loss (per sample)
                bce_loss_per_sample = self.bce_criterion(logits, labels)

                # OHEM
                if self.use_ohem:
                    sorted_loss, _ = torch.sort(bce_loss_per_sample, descending=True)
                    keep_num = max(1, int(len(sorted_loss) * self.ohem_ratio))
                    loss = sorted_loss[:keep_num].mean()
                else:
                    loss = bce_loss_per_sample.mean()

                bce_loss = loss
                contrast_loss = torch.tensor(0.0, device=self.device)
                distance_loss = torch.tensor(0.0, device=self.device)
                proba = torch.sigmoid(logits)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Record
            total_loss += loss.item() * len(labels)
            total_bce_loss += bce_loss.item() * len(labels)
            total_contrast_loss += contrast_loss.item() * len(labels)
            total_distance_loss += distance_loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(proba.detach().cpu().numpy())

        # Compute metrics
        n_samples = len(self.train_loader.dataset)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            roc_auc = 0.5

        try:
            prc_auc = average_precision_score(all_labels, all_probs)
        except ValueError:
            prc_auc = 0.5

        return {
            'loss': total_loss / n_samples,
            'bce_loss': total_bce_loss / n_samples,
            'contrast_loss': total_contrast_loss / n_samples,
            'distance_loss': total_distance_loss / n_samples,
            'roc_auc': roc_auc,
            'prc_auc': prc_auc
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate"""
        self.model.eval()

        total_loss = 0.0
        all_labels = []
        all_probs = []

        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        for batch in pbar:
            if len(batch) == 4:
                enhancer, promoter, labels, distances = batch
                distances = distances.to(self.device)
            else:
                enhancer, promoter, labels = batch
                distances = None

            enhancer = enhancer.to(self.device)
            promoter = promoter.to(self.device)
            labels = labels.float().to(self.device)

            # V2 model
            if hasattr(self.model, 'use_contrastive'):
                result = self.model(enhancer, promoter, distances, return_loss=False)
                logits = result['logits'].squeeze(-1)
                proba = result['proba'].squeeze(-1)
            else:
                if hasattr(self.model, 'distance_encoder') and distances is not None:
                    logits = self.model(enhancer, promoter, distances).squeeze(-1)
                else:
                    logits = self.model(enhancer, promoter).squeeze(-1)
                proba = torch.sigmoid(logits)

            loss = self.bce_criterion(logits, labels).mean()

            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(proba.cpu().numpy())

        n_samples = len(self.val_loader.dataset)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            roc_auc = 0.5

        try:
            prc_auc = average_precision_score(all_labels, all_probs)
        except ValueError:
            prc_auc = 0.5

        return {
            'loss': total_loss / n_samples,
            'roc_auc': roc_auc,
            'prc_auc': prc_auc
        }

    def train(self, epochs: int, save_every: int = 5, start_epoch: int = 0) -> Dict:
        """
        Full training pipeline

        Args:
            epochs: Total number of training epochs
            save_every: Save checkpoint every N epochs
            start_epoch: Starting epoch (for resume)
        """
        print(f"Starting training for {epochs} epochs...")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Use contrastive: {self.use_contrastive}")
        print("-" * 60)

        start_time = time.time()

        for epoch in range(start_epoch + 1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['roc_auc'])
                else:
                    self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_bce_loss'].append(train_metrics['bce_loss'])
            self.history['train_contrast_loss'].append(train_metrics['contrast_loss'])
            self.history['train_distance_loss'].append(train_metrics['distance_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_roc_auc'].append(train_metrics['roc_auc'])
            self.history['val_roc_auc'].append(val_metrics['roc_auc'])
            self.history['train_prc_auc'].append(train_metrics['prc_auc'])
            self.history['val_prc_auc'].append(val_metrics['prc_auc'])
            self.history['lr'].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Print progress
            if self.use_contrastive:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Loss: {train_metrics['loss']:.4f} "
                      f"(BCE: {train_metrics['bce_loss']:.4f}, "
                      f"Contrast: {train_metrics['contrast_loss']:.4f}) | "
                      f"Val ROC: {val_metrics['roc_auc']:.4f} | "
                      f"Val PRC: {val_metrics['prc_auc']:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.1f}s")
            else:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Train ROC: {train_metrics['roc_auc']:.4f} | "
                      f"Val ROC: {val_metrics['roc_auc']:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.1f}s")

            # Save best model
            if val_metrics['roc_auc'] > self.best_val_roc_auc:
                self.best_val_roc_auc = val_metrics['roc_auc']
                self.save_model(self.best_model_path, epoch=epoch)
                print(f"  -> Saved best model (ROC-AUC: {self.best_val_roc_auc:.4f})")

            # Periodic checkpoint saving
            if epoch % save_every == 0:
                checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
                self.save_model(checkpoint_path, epoch=epoch)

            # Early stopping
            if self.early_stopping(val_metrics['roc_auc']):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Training completed in {total_time:.1f}s")
        print(f"Best validation ROC-AUC: {self.best_val_roc_auc:.4f}")

        # Save training history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")

        return self.history

    def save_model(self, path: str, epoch: int = 0):
        """Save model"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_roc_auc': self.best_val_roc_auc,
            'history': self.history,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, path)

    def load_model(self, path: str, load_optimizer: bool = True, load_history: bool = True):
        """
        Load model

        Args:
            path: Checkpoint path
            load_optimizer: Whether to load optimizer state
            load_history: Whether to load training history

        Returns:
            Loaded epoch number (0 if not present)
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'best_val_roc_auc' in checkpoint:
            self.best_val_roc_auc = checkpoint['best_val_roc_auc']

        if load_history and 'history' in checkpoint:
            self.history = checkpoint['history']

        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint.get('epoch', 0)


def create_trainer_v2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = 'cuda',
    output_dir: str = './outputs',
    patience: int = 10,
    scheduler_type: str = 'cosine',
    use_contrastive: bool = True,
    contrastive_weight: float = 0.3,
    bce_weight: float = 0.7,
    pos_weight: Optional[float] = None,  # Positive sample weight
    use_focal: bool = False,  # Whether to use Focal Loss
    focal_alpha: float = 0.25,  # Focal Loss alpha
    focal_gamma: float = 2.0,  # Focal Loss gamma
    use_ohem: bool = False,  # Whether to use OHEM
    ohem_ratio: float = 0.7  # OHEM keep ratio
) -> TrainerV2:
    """Create trainer V2"""
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    else:
        scheduler = None

    trainer = TrainerV2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        patience=patience,
        use_contrastive=use_contrastive,
        contrastive_weight=contrastive_weight,
        bce_weight=bce_weight,
        pos_weight=pos_weight,
        use_focal=use_focal,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        use_ohem=use_ohem,
        ohem_ratio=ohem_ratio
    )

    return trainer
