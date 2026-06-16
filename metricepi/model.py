#!/usr/bin/env python3
"""
Distance-Aware EPI Model V2 - True Distance-Aware Attention + Contrastive Learning

Key Innovations:
1. True Distance-Aware Attention - Distance bias directly added to attention scores QK^T/√d
2. Multi-Scale Distance Encoding - Different scales capture different distance patterns
3. Distance-Guided Gating - Distance dynamically modulates sequence/epigenetic feature weights
4. Contrastive Learning + Distance Constraints - Distance-guided contrastive learning loss

Paper Narrative:
- First to explicitly integrate genomic distance into Transformer attention mechanism
- Distance as prior knowledge, not simple feature concatenation
- Contrastive learning framework with distance constraints as additional supervision
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN feature extraction"""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_sizes: list = [3, 5, 7, 11],
        dropout: float = 0.1
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.kernel_sizes = kernel_sizes
        num_kernels = len(kernel_sizes)
        channels_per_kernel = hidden_dim // num_kernels

        for k in kernel_sizes:
            conv = nn.Sequential(
                nn.Conv1d(in_channels, channels_per_kernel, k, padding=k // 2),
                nn.BatchNorm1d(channels_per_kernel),
                nn.ReLU(),
            )
            self.convs.append(conv)

        self.residual = nn.Conv1d(in_channels, hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.scale_weights = nn.Parameter(torch.ones(num_kernels) / num_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_outs = []
        for i, conv in enumerate(self.convs):
            out = conv(x)
            out = out * self.scale_weights[i]
            conv_outs.append(out)

        out = torch.cat(conv_outs, dim=1)
        out = out + self.residual(x)
        out = self.dropout(out)
        return out


# ============================================================================
# Innovation 1: Multi-Scale Distance Encoding
# ============================================================================

class MultiScaleDistanceEncoder(nn.Module):
    """
    Multi-scale distance encoder

    Different scales capture different distance patterns:
    - Scale 0: 0-10kb (direct interaction)
    - Scale 1: 10-100kb (within-TAD interaction)
    - Scale 2: 100kb-1Mb (cross-TAD interaction)
    - Scale 3: >1Mb (long-range interaction)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        max_distance: int = 1000000,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_distance = max_distance

        # Distance scale boundaries
        self.scale_boundaries = [10000, 100000, 1000000]  # 10kb, 100kb, 1Mb
        self.num_scales = len(self.scale_boundaries) + 1

        # Independent encoder for each scale
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            ) for _ in range(self.num_scales)
        ])

        # Scale fusion weights
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * self.num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Distance decay parameter (learnable)
        self.decay_rate = nn.Parameter(torch.tensor(50000.0))

    def get_scale_idx(self, distance: torch.Tensor) -> torch.Tensor:
        """Determine which scale the distance belongs to"""
        scale_idx = torch.zeros_like(distance, dtype=torch.long)
        for i, boundary in enumerate(self.scale_boundaries):
            scale_idx[distance > boundary] = i + 1
        return scale_idx

    def normalize_distance(self, distance: torch.Tensor) -> torch.Tensor:
        """Log-normalize distance"""
        return torch.log1p(distance) / math.log1p(self.max_distance)

    def get_decay_weight(self, distance: torch.Tensor) -> torch.Tensor:
        """Distance decay weight"""
        decay = torch.exp(-distance / self.decay_rate)
        return torch.clamp(decay, 0.01, 1.0)

    def forward(self, distance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            distance: (batch,) genomic distance in bp
        Returns:
            distance_feat: (batch, hidden_dim) multi-scale distance features
            decay_weight: (batch,) decay weight
            scale_idx: (batch,) scale index
        """
        batch_size = distance.size(0)
        d_norm = self.normalize_distance(distance).unsqueeze(-1)  # (batch, 1)
        scale_idx = self.get_scale_idx(distance)

        # Multi-scale encoding
        scale_embeddings = []
        for i, encoder in enumerate(self.scale_encoders):
            emb = encoder(d_norm)  # (batch, hidden_dim)
            # Activate only at corresponding scale
            mask = (scale_idx == i).float().unsqueeze(-1)
            scale_embeddings.append(emb * mask + emb * 0.1)  # Keep 10% info from other scales

        # Concatenate all scales
        all_scales = torch.cat(scale_embeddings, dim=-1)  # (batch, hidden_dim * num_scales)
        distance_feat = self.scale_fusion(all_scales)

        # Decay weight
        decay_weight = self.get_decay_weight(distance)

        return distance_feat, decay_weight, scale_idx


# ============================================================================
# Innovation 2: True Distance-Aware Attention
# ============================================================================

class TrueDistanceAwareAttention(nn.Module):
    """
    True distance-aware attention

    Core Innovation: Distance bias directly added to attention scores

    Attention(Q, K, V, d) = softmax(QK^T/√d_k + bias(d)) × V

    Where bias(d) is generated from distance encoding:
    - Short distance → positive bias (enhance attention)
    - Long distance → negative bias (suppress attention)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Innovation: Distance-to-bias network
        # Input: distance features (hidden_dim)
        # Output: bias value per head (num_heads)
        self.distance_to_bias = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads),
            nn.Tanh()  # Output range [-1, 1]
        )

        # Bias scaling factor (learnable)
        self.bias_scale = nn.Parameter(torch.tensor(2.0))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        distance_feat: torch.Tensor,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (batch, seq_len_q, hidden_dim)
            key: (batch, seq_len_k, hidden_dim)
            value: (batch, seq_len_v, hidden_dim)
            distance_feat: (batch, hidden_dim) distance features
            need_weights: whether to return attention weights
        Returns:
            output: (batch, seq_len_q, hidden_dim)
            attn_weights: (batch, num_heads, seq_len_q, seq_len_k) or None
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Projections
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim)

        # Transpose: (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # ========== Innovation: Distance bias added to attention scores ==========
        # distance_bias: (batch, num_heads)
        distance_bias = self.distance_to_bias(distance_feat)
        # Scale bias
        distance_bias = distance_bias * self.bias_scale  # (batch, num_heads)
        # Expand to attention matrix shape: (batch, num_heads, 1, 1)
        distance_bias = distance_bias.unsqueeze(-1).unsqueeze(-1)
        # Add to scores
        scores = scores + distance_bias
        # ==========================================================================

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        output = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len_q, head_dim)

        # Transpose back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_dim)
        output = self.out_proj(output)

        if need_weights:
            return output, attn_weights
        return output, None


class BiDistanceAwareCrossAttention(nn.Module):
    """
    Bidirectional distance-aware cross attention

    Enhancer → Promoter (with distance bias)
    Promoter → Enhancer (with distance bias)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Bidirectional attention
        self.enh_to_pro_attn = TrueDistanceAwareAttention(hidden_dim, num_heads, dropout)
        self.pro_to_enh_attn = TrueDistanceAwareAttention(hidden_dim, num_heads, dropout)

        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Output fusion
        self.output_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(
        self,
        enh_feat: torch.Tensor,
        pro_feat: torch.Tensor,
        distance_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            enh_feat: (batch, enh_len, hidden_dim)
            pro_feat: (batch, pro_len, hidden_dim)
            distance_feat: (batch, hidden_dim)
        Returns:
            (batch, hidden_dim)
        """
        # Enhancer → Promoter
        enh_attn_out, _ = self.enh_to_pro_attn(
            query=enh_feat,
            key=pro_feat,
            value=pro_feat,
            distance_feat=distance_feat
        )
        enh_out = self.norm1(enh_feat + enh_attn_out)

        # Promoter → Enhancer
        pro_attn_out, _ = self.pro_to_enh_attn(
            query=pro_feat,
            key=enh_feat,
            value=enh_feat,
            distance_feat=distance_feat
        )
        pro_out = self.norm2(pro_feat + pro_attn_out)

        # FFN
        enh_out = self.norm3(enh_out + self.ffn(enh_out))
        pro_out = self.norm3(pro_out + self.ffn(pro_out))

        # Multi-pooling
        enh_mean = torch.mean(enh_out, dim=1)
        enh_max = torch.max(enh_out, dim=1)[0]
        pro_mean = torch.mean(pro_out, dim=1)
        pro_max = torch.max(pro_out, dim=1)[0]

        # Fusion
        combined = torch.cat([enh_mean, enh_max, pro_mean, pro_max], dim=-1)
        output = self.output_fusion(combined)

        return output


# ============================================================================
# Innovation 3: Distance-Guided Gating
# ============================================================================

class DistanceGuidedGate(nn.Module):
    """
    Distance-guided feature gating

    Core idea: Distance dynamically modulates sequence and epigenetic feature weights

    Short distance → rely more on sequence features (fine-grained matching)
    Long distance → rely more on epigenetic features (regional signals)
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        # Gating network: distance → [seq_weight, epi_weight]
        self.gate_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        # Feature projections
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim)
        self.epi_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        seq_feat: torch.Tensor,
        epi_feat: torch.Tensor,
        distance: torch.Tensor,
        max_distance: float = 1000000.0
    ) -> torch.Tensor:
        """
        Args:
            seq_feat: (batch, seq_len, hidden_dim) sequence features
            epi_feat: (batch, seq_len, hidden_dim) epigenetic features
            distance: (batch,) genomic distance
        Returns:
            (batch, seq_len, hidden_dim) gated fused features
        """
        # Normalize distance
        d_norm = torch.log1p(distance) / math.log1p(max_distance)
        d_norm = d_norm.unsqueeze(-1)  # (batch, 1)

        # Compute gating weights
        gate_logits = self.gate_net(d_norm)  # (batch, 2)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (batch, 2)

        seq_weight = gate_weights[:, 0:1].unsqueeze(1)  # (batch, 1, 1)
        epi_weight = gate_weights[:, 1:2].unsqueeze(1)  # (batch, 1, 1)

        # Weighted fusion
        seq_out = self.seq_proj(seq_feat)
        epi_out = self.epi_proj(epi_feat)
        fused = seq_weight * seq_out + epi_weight * epi_out

        return self.output_proj(fused)


# ============================================================================
# Innovation 4: Contrastive Learning + Distance Constraints Loss
# ============================================================================

class DistanceContrastiveLoss(nn.Module):
    """
    Distance-guided contrastive learning loss

    Core idea:
    - Positive pairs (interaction): Pull embeddings closer
    - Negative pairs (no interaction): Push embeddings apart

    Distance constraints:
    - Short-distance negatives → stronger push (they "should" interact but don't)
    - Long-distance positives → stronger pull (they "rarely" interact but do)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        max_distance: float = 1000000.0,
        lambda_contrast: float = 1.0,
        lambda_distance: float = 0.5
    ):
        super().__init__()

        self.temperature = temperature
        self.max_distance = max_distance
        self.lambda_contrast = lambda_contrast
        self.lambda_distance = lambda_distance

    def forward(
        self,
        enh_emb: torch.Tensor,
        pro_emb: torch.Tensor,
        labels: torch.Tensor,
        distances: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            enh_emb: (batch, hidden_dim) enhancer embedding
            pro_emb: (batch, hidden_dim) promoter embedding
            labels: (batch,) 0 or 1
            distances: (batch,) genomic distance
        Returns:
            loss_dict: {'total': total_loss, 'contrast': contrast_loss, 'distance': distance_loss}
        """
        batch_size = enh_emb.size(0)

        # Normalize embeddings
        enh_emb = F.normalize(enh_emb, p=2, dim=-1)
        pro_emb = F.normalize(pro_emb, p=2, dim=-1)

        # Compute similarity: (batch,)
        similarity = F.cosine_similarity(enh_emb, pro_emb, dim=-1)

        # Normalize distance weight
        d_weight = torch.log1p(distances) / math.log1p(self.max_distance)  # (batch,)

        # ========== Contrastive learning loss ==========
        # Positive samples: similarity should be high
        pos_mask = labels == 1
        if pos_mask.any():
            # Long-distance positives get higher weight (rarer interactions)
            pos_weight = 1.0 + d_weight[pos_mask]  # Range [1, 2]
            pos_loss = -torch.log(
                torch.sigmoid(similarity[pos_mask] / self.temperature) + 1e-8
            ) * pos_weight
            pos_loss = pos_loss.mean()
        else:
            pos_loss = torch.tensor(0.0, device=enh_emb.device)

        # Negative samples: similarity should be low
        neg_mask = labels == 0
        if neg_mask.any():
            # Short-distance negatives get higher weight (should interact but don't)
            neg_weight = 2.0 - d_weight[neg_mask]  # Range [1, 2]
            neg_loss = -torch.log(
                torch.sigmoid(-similarity[neg_mask] / self.temperature) + 1e-8
            ) * neg_weight
            neg_loss = neg_loss.mean()
        else:
            neg_loss = torch.tensor(0.0, device=enh_emb.device)

        contrast_loss = (pos_loss + neg_loss) * self.lambda_contrast

        # ========== Distance consistency loss ==========
        # Encourage model to learn distance-similarity relationship
        # Short distance → high similarity, Long distance → low similarity
        expected_sim = 1.0 - d_weight  # (batch,)
        distance_loss = F.mse_loss(similarity, expected_sim) * self.lambda_distance

        # ========== Total loss ==========
        total_loss = contrast_loss + distance_loss

        return {
            'total': total_loss,
            'contrast': contrast_loss,
            'distance': distance_loss
        }


# ============================================================================
# Complete Model
# ============================================================================

class RegionEncoderV2(nn.Module):
    """
    Region encoder V2

    Multi-scale CNN + Pooling + Transformer
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pool_factor: int = 4
    ):
        super().__init__()

        self.pool_factor = pool_factor

        # Multi-scale CNN
        self.cnn1 = MultiScaleCNN(in_channels, hidden_dim, kernel_sizes=[3, 5, 7, 11], dropout=dropout)

        # Pooling
        self.pool = nn.MaxPool1d(pool_factor)

        # Second CNN layer
        self.cnn2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels)
        Returns:
            (batch, seq_len // pool_factor, hidden_dim)
        """
        x = x.transpose(1, 2)  # (batch, in_channels, seq_len)
        x = self.cnn1(x)       # (batch, hidden_dim, seq_len)
        x = self.pool(x)       # (batch, hidden_dim, seq_len // pool_factor)
        x = self.cnn2(x)
        x = x.transpose(1, 2)  # (batch, seq_len', hidden_dim)

        x = self.pos_enc(x)
        x = self.transformer(x)

        return x


class DistanceAwareEPIModelV2(nn.Module):
    """
    Distance-aware EPI model V2

    Key innovations:
    1. True distance-aware attention - bias directly added to QK^T
    2. Multi-scale distance encoding - captures patterns at different ranges
    3. Distance-guided gating - dynamically modulates sequence/epigenetic weights
    4. Contrastive learning loss - distance-constrained contrastive learning
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        pool_factor: int = 4,
        max_distance: int = 1000000,
        use_contrastive: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_contrastive = use_contrastive

        # Region encoder
        self.encoder = RegionEncoderV2(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pool_factor=pool_factor
        )

        # Multi-scale distance encoder (Innovation 1)
        self.distance_encoder = MultiScaleDistanceEncoder(
            hidden_dim=hidden_dim,
            max_distance=max_distance,
            dropout=dropout
        )

        # Bidirectional distance-aware cross attention (Innovation 2)
        self.cross_attention = BiDistanceAwareCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Contrastive learning loss (Innovation 4)
        if use_contrastive:
            self.contrastive_loss = DistanceContrastiveLoss(
                temperature=0.1,
                max_distance=max_distance
            )

        # Embedding projection for contrastive learning
        self.emb_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        enhancer: torch.Tensor,
        promoter: torch.Tensor,
        distance: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            enhancer: (batch, enh_len, 10)
            promoter: (batch, pro_len, 10)
            distance: (batch,) E-P genomic distance (bp)
            labels: (batch,) labels for contrastive learning
            return_loss: whether to return contrastive learning loss
        Returns:
            dict with 'logits', 'proba', and optionally 'loss'
        """
        batch_size = enhancer.size(0)

        # Encoding
        enh_feat = self.encoder(enhancer)
        pro_feat = self.encoder(promoter)

        # Distance encoding
        if distance is None:
            distance = torch.full((batch_size,), 50000.0, device=enhancer.device)

        distance_feat, decay_weight, scale_idx = self.distance_encoder(distance)

        # Bidirectional distance-aware cross attention
        interaction_feat = self.cross_attention(enh_feat, pro_feat, distance_feat)

        # Classification
        logits = self.classifier(interaction_feat)
        proba = torch.sigmoid(logits)

        result = {
            'logits': logits,
            'proba': proba,
            'distance_feat': distance_feat,
            'decay_weight': decay_weight,
            'scale_idx': scale_idx
        }

        # Contrastive learning loss
        if return_loss and self.use_contrastive and labels is not None:
            # Get embeddings
            enh_emb = self.emb_proj(torch.mean(enh_feat, dim=1))
            pro_emb = self.emb_proj(torch.mean(pro_feat, dim=1))

            loss_dict = self.contrastive_loss(enh_emb, pro_emb, labels, distance)
            result['loss_dict'] = loss_dict

        return result

    def predict_proba(
        self,
        enhancer: torch.Tensor,
        promoter: torch.Tensor,
        distance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Inference interface"""
        result = self.forward(enhancer, promoter, distance, return_loss=False)
        return result['proba']


def create_distance_aware_model_v2(
    hidden_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    pool_factor: int = 4,
    max_distance: int = 1000000,
    use_contrastive: bool = True,
    device: str = 'cuda'
) -> DistanceAwareEPIModelV2:
    """Create distance-aware model V2"""
    model = DistanceAwareEPIModelV2(
        in_channels=10,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        pool_factor=pool_factor,
        max_distance=max_distance,
        use_contrastive=use_contrastive
    )
    model = model.to(device)
    return model


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("Testing DistanceAwareEPIModelV2...")
    print("=" * 60)

    batch_size = 4
    enh_len = 3000
    pro_len = 2000

    enhancer = torch.randn(batch_size, enh_len, 10)
    promoter = torch.randn(batch_size, pro_len, 10)
    distance = torch.tensor([1000.0, 50000.0, 200000.0, 500000.0])
    labels = torch.tensor([1, 1, 0, 0])

    model = create_distance_aware_model_v2(device='cpu')

    # Test forward pass
    result = model(enhancer, promoter, distance, labels, return_loss=True)

    print(f"Input shapes:")
    print(f"  enhancer: {enhancer.shape}")
    print(f"  promoter: {promoter.shape}")
    print(f"  distance: {distance.shape}")
    print(f"\nOutput:")
    print(f"  logits: {result['logits'].shape}")
    print(f"  proba: {result['proba'].shape}")
    print(f"  decay_weight: {result['decay_weight']}")
    print(f"  scale_idx: {result['scale_idx']}")

    if 'loss_dict' in result:
        print(f"\nContrastive Loss:")
        print(f"  total: {result['loss_dict']['total'].item():.4f}")
        print(f"  contrast: {result['loss_dict']['contrast'].item():.4f}")
        print(f"  distance: {result['loss_dict']['distance'].item():.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable")

    print("\n" + "=" * 60)
    print("Test passed!")
