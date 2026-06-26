import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al. 2018)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).unsqueeze(-1)
        return x * scale


class MaskedPoolingClassifier(nn.Module):
    """BCE-only classifier head for padded protein language model embeddings."""

    def __init__(self, seq_len=512, in_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.input_norm = nn.LayerNorm(in_dim)
        self.token_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.se_block = SEBlock(hidden_dim, reduction=16)
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.pooling_weights = nn.Parameter(torch.ones(3))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _padding_mask(feats):
        return feats.abs().sum(dim=-1).gt(0)

    def _pool(self, feats, mask):
        mask_f = mask.unsqueeze(-1).to(feats.dtype)
        lengths = mask_f.sum(dim=1).clamp_min(1.0)

        mean_pool = (feats * mask_f).sum(dim=1) / lengths

        max_feats = feats.masked_fill(~mask.unsqueeze(-1), torch.finfo(feats.dtype).min)
        max_pool = max_feats.max(dim=1)[0]
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        attn_logits = self.attention_score(feats).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, torch.finfo(feats.dtype).min)
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        attn = torch.where(mask.unsqueeze(-1), attn, torch.zeros_like(attn))
        attn_pool = (feats * attn).sum(dim=1)

        weights = torch.softmax(self.pooling_weights, dim=0)
        pools = (mean_pool, max_pool, attn_pool)
        return torch.cat([pool * weight for pool, weight in zip(pools, weights)], dim=1)

    def forward(self, feats):
        mask = self._padding_mask(feats)
        x = self.input_norm(feats)
        x = self.token_proj(x)
        x = x + self.se_block(self.local_conv(x.transpose(1, 2))).transpose(1, 2)
        pooled = self._pool(x, mask)
        logits = self.classifier(pooled)
        return logits, pooled


class SoluModel(MaskedPoolingClassifier):
    pass
