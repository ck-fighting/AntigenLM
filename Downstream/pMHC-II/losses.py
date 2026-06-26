import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        labels = labels.view(-1, 1)
        batch_size = embeddings.size(0)
        if batch_size <= 1:
            return embeddings.new_tensor(0.0)

        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        logits_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        positive_mask = (labels == labels.T) & logits_mask
        positive_counts = positive_mask.sum(dim=1)
        valid_anchors = positive_counts > 0
        if not torch.any(valid_anchors):
            return embeddings.new_tensor(0.0)

        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
        exp_similarity = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1)
        return -mean_log_prob[valid_anchors].mean()


class WeightedBCEWithContrastiveLoss(nn.Module):
    def __init__(
        self,
        pos_weight,
        contrastive_weight=0.01,
        temperature=0.1,
        window_entropy_weight=0.0,
        window_margin_weight=0.0,
        window_margin=1.0,
        window_loss_temperature=0.25,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32))
        self.contrastive = SupervisedContrastiveLoss(temperature=temperature)
        self.contrastive_weight = float(contrastive_weight)
        self.window_entropy_weight = float(window_entropy_weight)
        self.window_margin_weight = float(window_margin_weight)
        self.window_margin = float(window_margin)
        self.window_loss_temperature = float(window_loss_temperature)

    @property
    def needs_core_info(self):
        return self.window_entropy_weight > 0 or self.window_margin_weight > 0

    def _positive_window_entropy_loss(self, window_logits, window_mask, labels):
        positive_mask = labels > 0.5
        valid_counts = window_mask.long().sum(dim=1)
        selected = positive_mask & (valid_counts > 1)
        if not torch.any(selected):
            return window_logits.new_tensor(0.0)

        temperature = max(self.window_loss_temperature, 1e-6)
        masked_logits = window_logits.masked_fill(~window_mask.bool(), torch.finfo(window_logits.dtype).min)
        weights = torch.softmax(masked_logits / temperature, dim=1)
        entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=1)
        normalizer = torch.log(valid_counts.to(dtype=window_logits.dtype).clamp_min(2))
        normalized_entropy = entropy / normalizer
        return normalized_entropy[selected].mean()

    def _positive_top2_margin_loss(self, window_logits, window_mask, labels):
        positive_mask = labels > 0.5
        valid_counts = window_mask.long().sum(dim=1)
        selected = positive_mask & (valid_counts > 1)
        if not torch.any(selected):
            return window_logits.new_tensor(0.0)

        masked_logits = window_logits.masked_fill(~window_mask.bool(), torch.finfo(window_logits.dtype).min)
        top2 = masked_logits.topk(k=2, dim=1).values
        gap = top2[:, 0] - top2[:, 1]
        return F.relu(self.window_margin - gap[selected]).mean()

    def forward(self, logits, labels, embeddings, core_info=None):
        self.bce.pos_weight = self.bce.pos_weight.to(device=logits.device, dtype=logits.dtype)
        labels = labels.to(dtype=logits.dtype)
        bce_loss = self.bce(logits, labels)
        contrastive_loss = self.contrastive(embeddings, labels.long())
        entropy_loss = logits.new_tensor(0.0)
        margin_loss = logits.new_tensor(0.0)
        if self.needs_core_info:
            if core_info is None:
                raise ValueError("core_info is required when window entropy or margin loss is enabled.")
            window_logits = core_info["window_logits"].to(dtype=logits.dtype)
            window_mask = core_info["window_mask"].bool()
            if self.window_entropy_weight > 0:
                entropy_loss = self._positive_window_entropy_loss(window_logits, window_mask, labels)
            if self.window_margin_weight > 0:
                margin_loss = self._positive_top2_margin_loss(window_logits, window_mask, labels)

        total_loss = (
            bce_loss
            + self.contrastive_weight * contrastive_loss
            + self.window_entropy_weight * entropy_loss
            + self.window_margin_weight * margin_loss
        )
        return total_loss, {
            "bce": float(bce_loss.detach().cpu().item()),
            "contrastive": float(contrastive_loss.detach().cpu().item()),
            "window_entropy": float(entropy_loss.detach().cpu().item()),
            "window_margin": float(margin_loss.detach().cpu().item()),
            "total": float(total_loss.detach().cpu().item()),
        }
