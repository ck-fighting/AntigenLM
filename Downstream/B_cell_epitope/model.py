import torch
import torch.nn as nn
import torch.nn.functional as F


CLASSIFIER_TYPES = ("context_cnn",)
DEFAULT_RESIDUE_INPUT_DIM = 2240
DEFAULT_HIDDEN_DIM = 384
DEFAULT_DROPOUT = 0.45
DEFAULT_CNN_KERNEL_SIZES = (3, 5, 9)


class ContextCNNResidueClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, kernel_sizes=DEFAULT_CNN_KERNEL_SIZES):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * len(kernel_sizes), hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def masked_global_context(x, residue_mask):
        if residue_mask is None:
            return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)

        mask = residue_mask.unsqueeze(-1).to(dtype=x.dtype)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        mean_pool = (x * mask).sum(dim=1) / lengths
        masked_x = x.masked_fill(~residue_mask.unsqueeze(-1), -torch.finfo(x.dtype).max)
        max_pool = masked_x.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(self, features, residue_mask=None):
        x = self.input_norm(features)
        x = self.dropout(F.gelu(self.input_proj(x)))
        if residue_mask is not None:
            x = x * residue_mask.unsqueeze(-1).to(dtype=x.dtype)

        conv_input = x.transpose(1, 2)
        branch_outputs = [branch(conv_input) for branch in self.branches]
        x = self.fuse(torch.cat(branch_outputs, dim=1)).transpose(1, 2) + x
        if residue_mask is not None:
            x = x * residue_mask.unsqueeze(-1).to(dtype=x.dtype)

        context = self.context_proj(self.masked_global_context(x, residue_mask))
        context = context.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, context], dim=-1)
        if residue_mask is not None:
            x = x * residue_mask.unsqueeze(-1).to(dtype=x.dtype)
        return self.output(x).squeeze(-1)


class WeightedBCEFocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, labels):
        labels = labels.float()
        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probabilities, 1 - probabilities)
        focal_weight = (1 - pt).pow(self.gamma)
        return (focal_weight * bce).mean()


def parse_kernel_sizes(kernel_sizes):
    if isinstance(kernel_sizes, str):
        kernel_sizes = tuple(int(part.strip()) for part in kernel_sizes.split(",") if part.strip())
    else:
        kernel_sizes = tuple(int(size) for size in kernel_sizes)

    if not kernel_sizes:
        raise ValueError("At least one CNN kernel size is required.")
    if any(size <= 0 or size % 2 == 0 for size in kernel_sizes):
        raise ValueError("CNN kernel sizes must be positive odd integers.")
    return kernel_sizes


def create_residue_classifier(
    classifier_type,
    input_dim=DEFAULT_RESIDUE_INPUT_DIM,
    hidden_dim=DEFAULT_HIDDEN_DIM,
    dropout=DEFAULT_DROPOUT,
    cnn_kernel_sizes=DEFAULT_CNN_KERNEL_SIZES,
    bilstm_layers=1,
):
    if classifier_type != "context_cnn":
        raise ValueError(f"Only context_cnn is kept in this final B-cell epitope pipeline: {classifier_type}")
    return ContextCNNResidueClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        kernel_sizes=parse_kernel_sizes(cnn_kernel_sizes),
    )


def infer_classifier_type(checkpoint):
    return checkpoint.get("classifier_type") or "context_cnn"


def classifier_state_dict(checkpoint):
    return checkpoint["classifier_state_dict"]


def classifier_checkpoint_config(checkpoint):
    config = dict(checkpoint.get("classifier_config") or {})
    config.setdefault("hidden_dim", checkpoint.get("hidden_dim", DEFAULT_HIDDEN_DIM))
    config.setdefault("dropout", checkpoint.get("dropout", DEFAULT_DROPOUT))
    config.setdefault("cnn_kernel_sizes", checkpoint.get("cnn_kernel_sizes", DEFAULT_CNN_KERNEL_SIZES))
    config.setdefault("bilstm_layers", checkpoint.get("bilstm_layers", 1))
    return config
