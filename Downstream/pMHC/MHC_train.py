
import os
import sys
import math
import json
from datetime import timedelta
import random
import argparse
from typing import Tuple, List
# ---- Env (keep your NCCL settings; adjust if you know your fabric) ----
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "0")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.metrics import (
    roc_auc_score, auc, accuracy_score, f1_score,
    precision_recall_curve, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef
)

# ---- Your custom models & helpers ----
from MHC_model import *
from feature_extractors import antigenLM_extract, extract_esm2_embeddings, extract_esmc_embeddings


# =========================
# Utilities & Config
# =========================

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log(msg: str) -> None:
    if is_main_process():
        print(msg, flush=True)

def binarize(probs: List[float], thr: float) -> List[int]:
    return [1 if p >= thr else 0 for p in probs]



# =========================
# Losses & Adversarial
# =========================




def _model_id_for_cache(cfg) -> str:
    """根据后端返回一个短模型标识，用于缓存命名。"""
    if cfg.embed_backend == "esm2":
        return cfg.esm2_model_name
    if cfg.embed_backend == "esmc":
        return cfg.esmc_model_name
    if cfg.embed_backend == "antigenLM":
        if cfg.antigenLM_path:
            return os.path.basename(cfg.antigenLM_path.rstrip("/"))
        return "antigenLM"
    if cfg.embed_backend == "antigenLM_withoutSS":
        if cfg.antigenLM_withoutSS:
            return os.path.basename(cfg.antigenLM_withoutSS.rstrip("/"))
        return "antigenLM_withoutSS"
    if cfg.embed_backend == "antigenLM_withoutSS_SW":
        if cfg.antigenLM_withoutSS_SW:
            return os.path.basename(cfg.antigenLM_withoutSS_SW.rstrip("/"))
        return "antigenLM_withoutSS_SW"
    if cfg.embed_backend in ("antigenLM_withoutSlidingwindow", "antigenLM_withoutSlidingWindow"):
        if cfg.antigenLM_withoutSlidingwindow:
            return os.path.basename(cfg.antigenLM_withoutSlidingwindow.rstrip("/"))
        return "antigenLM_withoutSlidingwindow"
    if cfg.embed_backend == "microLM":
        if cfg.microLM_path:
            return os.path.basename(cfg.microLM_path.rstrip("/"))
        return "microLM"
    if cfg.embed_backend == "PathogLM":
        if cfg.pathogLM_path:
            return os.path.basename(cfg.pathogLM_path.rstrip("/"))
        return "PathogLM"
    return "unknown"

def extract_peptide_embeddings(
    pep_list,
    pep_max_len: int,
    device: str,
    cfg
):
    """
    统一入口：根据 cfg.embed_backend 调用不同后端。
    返回 torch.Tensor [N, L, D] 或你模型期望的形状。
    """
    if cfg.embed_backend == "esm2":
        # 你已有的函数（来自 feature_extractors）
        return extract_esm2_embeddings(
            pep_list,
            model_name=cfg.esm2_model_name,
            device=device,
            max_len=pep_max_len if cfg.embed_max_len_override <= 0 else cfg.embed_max_len_override
        )

    elif cfg.embed_backend == "esmc":
        # 延迟导入，避免没装依赖时报错
        try:
            from esm.models.esmc import ESMC
        except Exception as e:
            raise ImportError("需要安装 `esm` 并支持 ESMC 模型。") from e
        client = ESMC.from_pretrained(cfg.esmc_model_name).to(device)
        return extract_esmc_embeddings(
            pep_list,
            client=client,
            device=device,
            batch_size=512,
            max_len=15,
            model_max_len=256,
        )

    elif cfg.embed_backend == "antigenLM":
        # 你的 AntigenLM 提取接口（来自你给的签名）

        return antigenLM_extract(
            pep_list,
            model_name_or_path=cfg.antigenLM_path,
            device=device,
            max_len=pep_max_len if cfg.embed_max_len_override <= 0 else cfg.embed_max_len_override
        )

    elif cfg.embed_backend == "antigenLM_withoutSS":
        return antigenLM_extract(
            pep_list,
            model_name_or_path=cfg.antigenLM_withoutSS,
            device=device,
            max_len=pep_max_len if cfg.embed_max_len_override <= 0 else cfg.embed_max_len_override
        )

    elif cfg.embed_backend == "antigenLM_withoutSS_SW":
        return antigenLM_extract(
            pep_list,
            model_name_or_path=cfg.antigenLM_withoutSS_SW,
            device=device,
            max_len=pep_max_len if cfg.embed_max_len_override <= 0 else cfg.embed_max_len_override
        )

    elif cfg.embed_backend in ("antigenLM_withoutSlidingwindow", "antigenLM_withoutSlidingWindow"):
        return antigenLM_extract(
            pep_list,
            model_name_or_path=cfg.antigenLM_withoutSlidingwindow,
            device=device,
            max_len=pep_max_len if cfg.embed_max_len_override <= 0 else cfg.embed_max_len_override
        )

    elif cfg.embed_backend == "microLM":
        return antigenLM_extract(
            pep_list,
            model_name_or_path=cfg.microLM_path,
            device=device,
            max_len=pep_max_len if cfg.embed_max_len_override <= 0 else cfg.embed_max_len_override
        )

    elif cfg.embed_backend == "PathogLM":
        return antigenLM_extract(
            pep_list,
            model_name_or_path=cfg.pathogLM_path,
            device=device,
            max_len=pep_max_len if cfg.embed_max_len_override <= 0 else cfg.embed_max_len_override
        )

    else:
        raise ValueError(f"未知的 embed_backend: {cfg.embed_backend}")

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (from https://arxiv.org/abs/2004.11362)"""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, embedding_dim] or [batch_size, n_views, embedding_dim]
            labels: [batch_size]
        Returns:
            scalar loss
        """
        # 如果特征输入是 [B, D]，加一个view维度变成 [B, 1, D]
        if features.dim() < 3:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        # [B, n_views, D] -> [B * n_views, D]
        features = F.normalize(features, dim=-1)
        features = features.view(batch_size, -1, features.shape[-1])
        anchor_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # repeat labels for n_views
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)  # [B, B]

        anchor_feature = contrast_feature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # 为了数值稳定，减去每行最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask自己
        mask_self = torch.eye(labels.shape[0] * anchor_count, dtype=torch.float32).to(features.device)
        mask = mask.repeat(anchor_count, anchor_count)
        mask = mask * (1 - mask_self)

        # 计算log_prob
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 只统计正样本对
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss

class FGM:
    """
    Fast Gradient Method on embedding parameters.
    Pass a list of substrings to match parameter names you want to perturb (e.g., ['encoder_H.src_emb', 'encoder_P.src_emb']).
    """
    def __init__(self, model: nn.Module, target_param_substrings: List[str]):
        self.model = model
        self.backup = {}
        self.targets = target_param_substrings

    @torch.no_grad()
    def attack(self, epsilon: float = 1.0):
        self.backup.clear()
        for name, p in self.model.named_parameters():
            if (not p.requires_grad) or (p.grad is None):
                continue
            if not any(t in name for t in self.targets):
                continue
            grad_norm = torch.norm(p.grad)
            if grad_norm == 0:
                continue
            self.backup[name] = p.data.clone()
            r_at = epsilon * p.grad / grad_norm
            p.add_(r_at)

    @torch.no_grad()
    def restore(self):
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup.clear()

# =========================
# Dataset & Data Functions
# =========================

class HLADataset(Dataset):
    """Minimal tensor dataset: returns (pep_embed, hla_ids, label)."""
    def __init__(self, pep_embeds: torch.Tensor, hla_ids: torch.LongTensor, labels: torch.LongTensor):
        assert pep_embeds.shape[0] == hla_ids.shape[0] == labels.shape[0]
        self.pep = pep_embeds
        self.hla = hla_ids
        self.y = labels

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.pep[idx], self.hla[idx], self.y[idx]



def compute_lengths(series: pd.Series) -> int:
    return max(int(len(s)) for s in series.tolist())

def data_process_HLA(
    data: pd.DataFrame,
    fold: int,
    type_: str,
    seed: int,
    device: torch.device,
    cfg
):
    data = data.copy()
    if "Peptide" in data.columns and "peptide" not in data.columns:
        data = data.rename(columns={"Peptide": "peptide"})
    if "antigen" in data.columns and "peptide" not in data.columns:
        data = data.rename(columns={"antigen": "peptide"})
    if "hla" in data.columns and "HLA" not in data.columns:
        data = data.rename(columns={"hla": "HLA"})
    if "HLA" not in data.columns or "peptide" not in data.columns:
        raise ValueError(f"缺少列: HLA/peptide, 实际列: {set(data.columns)}")

    os.makedirs(cfg.emb_cache_dir, exist_ok=True)

    # —— 肽段 padding —— 
    pep_list = []
    pep_max_len = compute_lengths(data.peptide)
    if cfg.embed_max_len_override > 0:
        pep_len_tag = cfg.embed_max_len_override
    else:
        pep_len_tag = pep_max_len
    for pep in data.peptide:
        pep_list.append(pep.ljust(pep_max_len, "-"))

    # —— 缓存命名：区分后端/模型/长度/折 —— 
    model_id = _model_id_for_cache(cfg)
    suffix = f"{type_}"
    if type_ in ("train", "val"):
        suffix += f"_{fold}_{seed}"
    cache_name = f"cached_pep_embeddings_{suffix}_{cfg.embed_backend}.pt"
    cache_path = os.path.join(cfg.emb_cache_dir, cache_name)

    # —— 嵌入：仅 rank0 抽取并缓存，其它 rank 等待后加载 —— 
    if not os.path.exists(cache_path) and is_main_process():
        log(f"[Cache] Extracting ({cfg.embed_backend}:{model_id}) for {len(pep_list)} peptides -> {cache_path}")
        pep_emb = extract_peptide_embeddings(
            pep_list=pep_list,
            pep_max_len=pep_max_len,
            device=str(device),
            cfg=cfg
        )
        torch.save(pep_emb.detach().cpu(), cache_path)
        log(f"[Cache] Saved peptide embeddings to {cache_path}")

    if is_dist_avail_and_initialized():
        dist.barrier()

    pep_embeddings = torch.load(cache_path, map_location="cpu")
    if isinstance(pep_embeddings, np.ndarray):
        pep_embeddings = torch.from_numpy(pep_embeddings)

    # —— HLA tokenization & labels —— 
    try:
        _hla_max_len = hla_max_len
    except NameError:
        _hla_max_len = compute_lengths(data.HLA)

    hla_ids, labels = [], []
    for hla_seq, y in zip(data.HLA, data.label):
        padded = hla_seq.ljust(_hla_max_len, "-")
        ids = [vocab[c] for c in padded]
        hla_ids.append(ids)
        labels.append(int(y))

    hla_tensor = torch.LongTensor(hla_ids)
    label_tensor = torch.LongTensor(labels)

    assert pep_embeddings.shape[0] == hla_tensor.shape[0], \
        f"Peptide N={pep_embeddings.shape[0]} != HLA N={hla_tensor.shape[0]}"

    return pep_embeddings, hla_tensor, label_tensor


def build_loader_ddp(
    data: pd.DataFrame,
    fold: int,
    type_: str,
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int,
    device: torch.device,
    cfg
) -> Tuple[DataLoader, DistributedSampler]:
    pep, hla, y = data_process_HLA(data, fold, type_, seed, device, cfg)
    dataset = HLADataset(pep, hla, y)
    shuffle = (type_ == 'train')
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=False,
        seed=seed
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False
    )
    return loader, sampler

# =========================
# Metrics
# =========================

def compute_performance(y_true: List[int], y_prob: List[float], y_pred: List[int]) -> Tuple[float, float, float, float, float, float, float, float, float]:
    eps = 1e-12
    # AUC may error if single class; guard
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float('nan')

    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec) if len(reca) > 1 else float('nan')

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    mcc = matthews_corrcoef(y_true, y_pred) if (tp+tn+fp+fn) else 0.0

    if is_main_process():
        log(f"tn={tn}, fp={fp}, fn={fn}, tp={tp}")
        c_pred = Counter(y_pred)
        c_true = Counter(y_true)
        log(f"y_pred: 0={c_pred.get(0,0)} | 1={c_pred.get(1,0)}")
        log(f"y_true: 0={c_true.get(0,0)} | 1={c_true.get(1,0)}")
        log(f"auc={roc_auc:.4f} | sens={sensitivity:.4f} | spec={specificity:.4f} | acc={acc:.4f} | mcc={mcc:.4f}")
        log(f"precision={precision:.4f} | recall={recall:.4f} | f1={f1:.4f} | aupr={aupr:.4f}")

    return (roc_auc, acc, mcc, f1, aupr, sensitivity, specificity, precision, recall)

# =========================
# Train / Valid
# =========================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    sampler: DistributedSampler,
    optimizer: torch.optim.Optimizer,
    bce: nn.Module,
    supcon: SupConLoss,
    fgm: FGM,
    device: torch.device,
    threshold: float,
    epoch: int,
    use_amp: bool,
    supcon_lambda: float,
    adv_epsilon: float
) -> Tuple[List[int], List[float], List[int], float]:
    model.train()
    sampler.set_epoch(epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    y_true_all, y_prob_all, loss_list = [], [], []

    for batch in train_loader:
        pep, hla, labels = [x.to(device, non_blocking=True) for x in batch]

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _, pep_hla = model(pep, hla)  # logits: [B,1] or [B]
            logits = logits.view(-1)
            main_loss = bce(logits, labels.float())

            # SupCon on pooled features
            # feats = pep_hla.mean(dim=1)  # [B, D]
            s_loss = supcon(pep_hla, labels)
            loss = main_loss + supcon_lambda * s_loss

        scaler.scale(loss).backward()

        # adversarial training
        fgm.attack(epsilon=adv_epsilon)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_adv, _, pep_hla_adv = model(pep, hla)
            logits_adv = logits_adv.view(-1)
            main_loss_adv = bce(logits_adv, labels.float())
            s_loss_adv = supcon(pep_hla_adv.mean(dim=1), labels)
            loss_adv = main_loss_adv + supcon_lambda * s_loss_adv
        scaler.scale(loss_adv).backward()
        fgm.restore()

        scaler.step(optimizer)
        scaler.update()

        y_true_all.extend(labels.detach().cpu().numpy().tolist())
        y_prob_all.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        loss_list.append(main_loss.item())

    y_pred_all = binarize(y_prob_all, threshold)
    return y_true_all, y_prob_all, y_pred_all, float(np.mean(loss_list)) if loss_list else math.nan

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    bce: nn.Module,
    device: torch.device,
    threshold: float,
    use_amp: bool
) -> Tuple[List[int], List[float], List[int], float]:
    model.eval()
    y_true_all, y_prob_all, loss_list = [], [], []
    for batch in val_loader:
        pep, hla, labels = [x.to(device, non_blocking=True) for x in batch]
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _, _ = model(pep, hla)
            logits = logits.view(-1)
            loss = bce(logits, labels.float())

        y_true_all.extend(labels.detach().cpu().numpy().tolist())
        y_prob_all.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        loss_list.append(loss.item())

    y_pred_all = binarize(y_prob_all, threshold)
    return y_true_all, y_prob_all, y_pred_all, float(np.mean(loss_list)) if loss_list else math.nan

# =========================
# Main
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="DDP training for pMHC (HLA) binding (clean version)")
    p.add_argument("--cv_dir", type=str, default="./data/cv_splits", help="Directory with train_fold_*.csv & val_fold_*.csv")
    p.add_argument("--emb_cache_dir", type=str, default="./data_cached/Ablation", help="Directory to cache peptide embeddings")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--supcon_temp", type=float, default=0.07)
    p.add_argument("--supcon_lambda", type=float, default=0.05)
    p.add_argument("--adv_epsilon", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--find_unused", action="store_true")
    p.add_argument("--no-find_unused", dest="find_unused", action="store_false")
    p.set_defaults(find_unused=True)

    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--save_dir", type=str, default="../trained_model/HLA_micro/Ablation")

    p.add_argument(
        "--embed_backend",
        type=str,
        default="microLM",
        choices=[
            "esm2",
            "esmc",
            "antigenLM",
            "antigenLM_withoutSS",
            "antigenLM_withoutSS_SW",
            "antigenLM_withoutSlidingwindow",
            "microLM",
            "PathogLM",
        ],
    )
    p.add_argument("--esm2_model_name", type=str, default="esm2_t33_650M_UR50D")
    p.add_argument("--esmc_model_name", type=str, default="esmc_300m")
    p.add_argument("--antigenLM_path", type=str, default="../../LLM/Result_antigenLM_300M_SS_2")
    p.add_argument("--antigenLM_withoutSS", type=str, default="../../Other_LLM/AntigenLM_300M_withoutSS")
    p.add_argument("--antigenLM_withoutSS_SW", type=str, default="../../Other_LLM/AntigenLM_300M_withoutSS_SW")
    p.add_argument("--antigenLM_withoutSlidingwindow", type=str, default="../../Other_LLM/AntigenLM_300M_withoutSlidingwindow")
    p.add_argument("--microLM_path", type=str, default="../../LLM/Result_microLM_300M/0406_034622_rank0")
    p.add_argument("--pathogLM_path", type=str, default="../../LLM/Result_PathogLM_300M_SS")
    p.add_argument("--embed_max_len_override", type=int, default=15)

    args = p.parse_args()
    return args

def main():
    # ----- DDP init -----
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("必须用 torchrun 启动。示例：torchrun --standalone --nproc_per_node=... train.py ...")

    # 2) 初始化 DDP（这里加超时）
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=4),   # ← 就是这里
    )

    # 3) 设备设置（必须在 init 之后拿 rank）
    local_rank = int(os.environ["LOCAL_RANK"])
    n_gpus = torch.cuda.device_count()
    if local_rank >= n_gpus:
        raise RuntimeError(f"LOCAL_RANK={local_rank} 超出可见 GPU 范围 0..{n_gpus-1}")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    cfg = parse_args()
    setup_seed(cfg.seed)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    os.makedirs(cfg.save_dir, exist_ok=True)
    log(json.dumps(vars(cfg), indent=2, ensure_ascii=False))

    # ----- Model / Opt / Loss -----
    model = Mymodel_HLA().to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=cfg.find_unused)

    bce = nn.BCEWithLogitsLoss()
    supcon = SupConLoss(temperature=cfg.supcon_temp, base_temperature=cfg.supcon_temp)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_overall = -1.0

    # ----- K-fold loop -----
    for fold in range(1, cfg.num_folds + 1):
        if is_main_process():
            log(f"\n========== Fold {fold}/{cfg.num_folds} ==========")

        train_csv = os.path.join(cfg.cv_dir, f"train_fold_{fold}.csv")
        val_csv   = os.path.join(cfg.cv_dir, f"test_fold_{fold}.csv")
        if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
            if is_main_process():
                log(f"缺少折文件：{train_csv} 或 {val_csv}")
            dist.barrier()
            dist.destroy_process_group()
            sys.exit(1)

        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)

        train_loader, train_sampler = build_loader_ddp(
            train_df, fold, 'train', cfg.batch_size,
            rank, world_size, seed=cfg.seed, device=device, cfg=cfg
        )
        val_loader, _ = build_loader_ddp(
            val_df, fold, 'val', cfg.batch_size,
            rank, world_size, seed=cfg.seed, device=device, cfg=cfg
        )

        # (Re)init best metric for the fold
        best_metric = -1.0
        save_path = os.path.join(cfg.save_dir, f"model_fold{fold}_seed{cfg.seed}_{cfg.embed_backend}.pt")

        # ----- Train epochs -----
        fgm = FGM(model, target_param_substrings=['encoder_H.src_emb'])

        for epoch in range(1, cfg.epochs + 1):
            y_true_tr, y_prob_tr, y_pred_tr, train_loss = train_one_epoch(
                model, train_loader, train_sampler, optimizer,
                bce, supcon, fgm, device, cfg.threshold, epoch,
                use_amp=cfg.use_amp, supcon_lambda=cfg.supcon_lambda, adv_epsilon=cfg.adv_epsilon
            )
            y_true_v, y_prob_v, y_pred_v, val_loss = validate(
                model, val_loader, bce, device, cfg.threshold, use_amp=cfg.use_amp
            )

            if is_main_process():
                perf = compute_performance(y_true_v, y_prob_v, y_pred_v)
                # 综合指标（可自定权重，这里等权平均前5个：AUC/ACC/MCC/F1/AUPR）
                vals = [x for x in perf[:5] if (not isinstance(x, float)) or (not math.isnan(x))]
                perf_avg = sum(vals) / len(vals) if vals else -1.0
                log(f"Fold {fold} | Epoch {epoch}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, ValPerf={perf_avg:.4f}")

                if perf_avg > best_metric:
                    best_metric = perf_avg
                    torch.save(model.module.state_dict(), save_path)
                    log(f"[Fold {fold}] Saved best at epoch {epoch} -> {save_path}")

        if is_main_process():
            log(f"[Fold {fold}] Best Val Perf = {best_metric:.4f}")
            best_overall = max(best_overall, best_metric)


    if is_main_process():
        log(f"\n===== K-fold finished. Best Val Perf across folds = {best_overall:.4f} =====")

    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
