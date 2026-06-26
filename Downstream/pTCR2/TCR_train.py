import os
import sys
import math
import json
import random
import argparse
import hashlib
from datetime import timedelta
from typing import Tuple, List
from collections import Counter

os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "0")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
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

from TCR_model import *
from feature_extractors import *

P = os.path.join

# ===================== Utilities =====================

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log(msg: str):
    if is_main_process():
        print(msg, flush=True)

def binarize(probs: List[float], thr: float) -> List[int]:
    return [1 if p >= thr else 0 for p in probs]


def gather_list_from_all_ranks(values):
    if not is_dist_avail_and_initialized():
        return values
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, values)
    merged = []
    for part in gathered:
        merged.extend(part)
    return merged


def peptide_list_cache_digest(peptides: List[str]) -> str:
    hasher = hashlib.sha1()
    for peptide in peptides:
        hasher.update(str(peptide).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:12]

# ===================== Embedding =====================

def _model_id_for_cache(cfg) -> str:
    if cfg.embed_backend == "esm2":
        return embedding_model_cache_id(cfg.embed_backend, cfg.esm2_model_name)
    if cfg.embed_backend == "esmc":
        return embedding_model_cache_id(cfg.embed_backend, cfg.esmc_model_name)
    if cfg.embed_backend == "AntigenLM":
        return embedding_model_cache_id(cfg.embed_backend, cfg.AntigenLM_path)
    return "unknown"

def extract_peptide_embeddings(pep_list, pep_max_len: int, device: str, cfg):
    max_len = pep_max_len if cfg.embed_max_len_override <= 0 else cfg.embed_max_len_override

    if cfg.embed_backend == "esm2":
        return extract_esm2_embeddings(pep_list, model_name=cfg.esm2_model_name, device=device, max_len=max_len)

    elif cfg.embed_backend == "esmc":
        client = load_esmc_model(cfg.esmc_model_name, device=device)
        return extract_esmc_embeddings(pep_list, client=client, device=device, batch_size=512, max_len=max_len, model_max_len=256)

    elif cfg.embed_backend == "AntigenLM":
        return antigenLM_extract(pep_list, model_name_or_path=cfg.AntigenLM_path, device=device, max_len=max_len)

    raise ValueError(f"Unknown embed_backend: {cfg.embed_backend}")

# ===================== Losses & Adversarial =====================

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (https://arxiv.org/abs/2004.11362)"""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        if features.dim() < 3:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        features = F.normalize(features, dim=-1)
        features = features.view(batch_size, -1, features.shape[-1])
        anchor_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask_self = torch.eye(labels.shape[0] * anchor_count, dtype=torch.float32).to(features.device)
        mask = mask.repeat(anchor_count, anchor_count) * (1 - mask_self)

        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


class FGM:
    """Fast Gradient Method adversarial training on embedding parameters."""
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
            p.add_(epsilon * p.grad / grad_norm)

    @torch.no_grad()
    def restore(self):
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup.clear()

# ===================== Dataset =====================

class TCRDataset(Dataset):
    def __init__(self, pep_embeds: torch.Tensor, tcr_ids: torch.LongTensor, labels: torch.LongTensor):
        assert pep_embeds.shape[0] == tcr_ids.shape[0] == labels.shape[0]
        self.pep, self.tcr, self.y = pep_embeds, tcr_ids, labels

    def __len__(self): return self.y.shape[0]
    def __getitem__(self, idx): return self.pep[idx], self.tcr[idx], self.y[idx]


def compute_lengths(series: pd.Series) -> int:
    return max(int(len(s)) for s in series.tolist())


def data_process_TCR(data: pd.DataFrame, fold: int, type_: str, seed: int, device: torch.device, cfg):
    data = data.copy()
    data["antigen"] = data["antigen"].astype(str).str.strip()
    data["TCR"] = data["TCR"].astype(str).str.strip()
    data = data[(data["antigen"] != "") & (data["TCR"] != "")].reset_index(drop=True)

    os.makedirs(cfg.emb_cache_dir, exist_ok=True)

    pep_max_len = compute_lengths(data.antigen)
    pep_list = data.antigen.tolist()

    # Cache naming
    model_id = _model_id_for_cache(cfg)
    model_max_len = pep_max_len if cfg.embed_max_len_override <= 0 else cfg.embed_max_len_override
    expected_dim = infer_peptide_embedding_dim(
        cfg.embed_backend,
        cfg.AntigenLM_path,
        cfg.esm2_model_name,
        cfg.esmc_model_name,
    )
    suffix = f"{type_}_{fold}_{seed}" if type_ in ("train", "val") else type_
    data_id = peptide_list_cache_digest(pep_list)
    cache_name = f"cached_pep_embeddings_{suffix}_{data_id}_{cfg.embed_backend}_{model_id}_L{model_max_len}.pt"
    cache_path = P(cfg.emb_cache_dir, cache_name)

    # Rank 0 extracts and caches embeddings
    if is_main_process():
        if not os.path.exists(cache_path):
            log(f"[Cache] Extracting ({cfg.embed_backend}:{model_id}) for {len(pep_list)} peptides -> {cache_path}")
            pep_emb = extract_peptide_embeddings(pep_list, pep_max_len, str(device), cfg)
            torch.save(pep_emb.detach().cpu(), cache_path)
            log(f"[Cache] Saved peptide embeddings to {cache_path}")
        else:
            pep_embeddings = torch.load(cache_path, map_location="cpu")
            if isinstance(pep_embeddings, np.ndarray):
                pep_embeddings = torch.from_numpy(pep_embeddings)
            expected_shape = (len(pep_list), model_max_len, expected_dim)
            if tuple(pep_embeddings.shape) != expected_shape:
                log(f"[Cache] Mismatch shape (cache {tuple(pep_embeddings.shape)} vs expected {expected_shape}), recomputing.")
                pep_emb = extract_peptide_embeddings(pep_list, pep_max_len, str(device), cfg)
                torch.save(pep_emb.detach().cpu(), cache_path)

    if is_dist_avail_and_initialized():
        dist.barrier()

    pep_embeddings = torch.load(cache_path, map_location="cpu")
    if isinstance(pep_embeddings, np.ndarray):
        pep_embeddings = torch.from_numpy(pep_embeddings)

    # TCR tokenization/embedding & labels
    try:
        _tcr_max_len = tcr_max_len
    except NameError:
        _tcr_max_len = compute_lengths(data.TCR)

    tcr_ids = []
    for tcr_seq in data.TCR:
        tcr_ids.append([vocab[c] for c in tcr_seq.ljust(_tcr_max_len, "-")])
    tcr_tensor = torch.LongTensor(tcr_ids)

    labels = []
    for y in data.label:
        labels.append(int(y))
    label_tensor = torch.LongTensor(labels)

    assert pep_embeddings.shape[0] == tcr_tensor.shape[0], \
        f"Peptide N={pep_embeddings.shape[0]} != TCR N={tcr_tensor.shape[0]}"

    return pep_embeddings, tcr_tensor, label_tensor


def build_loader_ddp(data, fold, type_, batch_size, rank, world_size, seed, device, cfg):
    pep, tcr, y = data_process_TCR(data, fold, type_, seed, device, cfg)
    dataset = TCRDataset(pep, tcr, y)
    if type_ == "train":
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False, seed=seed)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False)
    return loader, sampler

# ===================== Metrics =====================

def compute_performance(y_true, y_prob, y_pred):
    eps = 1e-12
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
    mcc = matthews_corrcoef(y_true, y_pred) if (tp + tn + fp + fn) else 0.0

    if is_main_process():
        log(f"tn={tn}, fp={fp}, fn={fn}, tp={tp}")
        log(f"auc={roc_auc:.4f} | sens={sensitivity:.4f} | spec={specificity:.4f} | acc={acc:.4f} | mcc={mcc:.4f}")
        log(f"precision={precision:.4f} | recall={recall:.4f} | f1={f1:.4f} | aupr={aupr:.4f}")

    return (roc_auc, acc, mcc, f1, aupr, sensitivity, specificity, precision, recall)

# ===================== Train / Valid =====================

def train_one_epoch(model, train_loader, sampler, optimizer, bce, supcon, fgm, device, threshold, epoch, use_amp, supcon_lambda, adv_epsilon):
    model.train()
    sampler.set_epoch(epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    y_true_all, y_prob_all, loss_list = [], [], []

    for batch in train_loader:
        pep, tcr, labels = [x.to(device, non_blocking=True) for x in batch]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _, pep_tcr = model(pep, tcr)
            logits = logits.view(-1)
            loss = bce(logits, labels.float()) + supcon_lambda * supcon(pep_tcr, labels)

        scaler.scale(loss).backward()

        # Adversarial training
        fgm.attack(epsilon=adv_epsilon)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits_adv, _, pep_tcr_adv = model(pep, tcr)
            logits_adv = logits_adv.view(-1)
            loss_adv = bce(logits_adv, labels.float()) + supcon_lambda * supcon(pep_tcr_adv, labels)
        scaler.scale(loss_adv).backward()
        fgm.restore()

        scaler.step(optimizer)
        scaler.update()

        y_true_all.extend(labels.detach().cpu().numpy().tolist())
        y_prob_all.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        loss_list.append(loss.item())

    return y_true_all, y_prob_all, binarize(y_prob_all, threshold), float(np.mean(loss_list)) if loss_list else math.nan


@torch.no_grad()
def validate(model, val_loader, bce, device, threshold, use_amp):
    model.eval()
    y_true_all, y_prob_all, loss_list = [], [], []

    for batch in val_loader:
        pep, tcr, labels = [x.to(device, non_blocking=True) for x in batch]
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _, _ = model(pep, tcr)
            logits = logits.view(-1)
            loss = bce(logits, labels.float())

        y_true_all.extend(labels.detach().cpu().numpy().tolist())
        y_prob_all.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        loss_list.append(loss.item())

    loss_mean = float(np.mean(loss_list)) if loss_list else math.nan
    return y_true_all, y_prob_all, binarize(y_prob_all, threshold), loss_mean

# ===================== Main =====================

def infer_emb_cache_dir(cv_dir):
    parts = {part.lower() for part in os.path.normpath(cv_dir).split(os.sep)}
    if any("cma" in part for part in parts):
        return "./data_cached_cma_5fold"
    return "./data_cached_seen_5fold"


def parse_args():
    p = argparse.ArgumentParser(description="DDP training for pTCR binding")
    p.add_argument("--cv_dir", type=str, default="./data/Seen_5fold_splits")
    p.add_argument("--emb_cache_dir", type=str, default=None, help="Peptide embedding cache directory. If omitted, inferred from --cv_dir.")
    p.add_argument("--save_dir", type=str, default="../trained_model/pTCR3/Seen")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--supcon_temp", type=float, default=0.07)
    p.add_argument("--supcon_lambda", type=float, default=0)
    p.add_argument("--adv_epsilon", type=float, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--find_unused", action="store_true", default=True)
    p.add_argument("--no-find_unused", dest="find_unused", action="store_false")

    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--fold_prefix", type=str, default="fold_", help="subdirectory prefix, e.g. fold_ for fold_1/train.csv")
    p.add_argument("--monitor_metric", type=str, default="auc", choices=["auc", "aupr", "perf_avg", "loss"])
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--seed", type=int, default=22)

    p.add_argument("--embed_backend", type=str, default="AntigenLM", choices=["esm2", "esmc", "AntigenLM"])
    p.add_argument("--esm2_model_name", type=str, default=DEFAULT_ESM2_MODEL_PATH)
    p.add_argument("--esmc_model_name", type=str, default=DEFAULT_ESMC_MODEL_PATH)
    p.add_argument("--AntigenLM_path", type=str, default="../../LLM/AntigenLM")
    p.add_argument("--embed_max_len_override", type=int, default=15)
    p.add_argument("--pep_input_norm", action="store_true", help="Layer-normalize each residue embedding before peptide projection.")
    p.add_argument("--pep_input_scale", type=float, default=1.0, help="Scale peptide embeddings after optional input normalization.")
    return p.parse_args()


def main():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Must launch with torchrun. Example: torchrun --standalone --nproc_per_node=... TCR_train.py ...")

    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=4))

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank} exceeds available GPUs 0..{torch.cuda.device_count()-1}")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    cfg = parse_args()
    cfg.emb_cache_dir = cfg.emb_cache_dir or infer_emb_cache_dir(cfg.cv_dir)
    setup_seed(cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)
    log(json.dumps(vars(cfg), indent=2, ensure_ascii=False))

    bce = nn.BCEWithLogitsLoss()
    supcon = SupConLoss(temperature=cfg.supcon_temp, base_temperature=cfg.supcon_temp)
    pep_dim = infer_peptide_embedding_dim(
        cfg.embed_backend,
        cfg.AntigenLM_path,
        cfg.esm2_model_name,
        cfg.esmc_model_name,
    )
    log(f"[Model] Peptide embedding dim = {pep_dim}")

    best_overall = float("inf") if cfg.monitor_metric == "loss" else -1.0

    # K-fold loop
    for fold in range(1, cfg.num_folds + 1):
        log(f"\n========== Fold {fold}/{cfg.num_folds} ==========")

        # Re-initialize model and optimizer for each fold
        model = DDP(
            Mymodel_TCR(
                pep_dim=pep_dim,
                pep_input_norm=cfg.pep_input_norm,
                pep_input_scale=cfg.pep_input_scale,
            ).to(device),
            device_ids=[local_rank],
            find_unused_parameters=cfg.find_unused,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        fold_dir = P(cfg.cv_dir, f"{cfg.fold_prefix}{fold}")
        train_csv = P(fold_dir, "train.csv")
        val_csv = P(fold_dir, "val.csv")
        if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
            log(f"[Skip] Missing fold files: {train_csv} or {val_csv}")
            dist.barrier()
            dist.destroy_process_group()
            sys.exit(1)

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        train_loader, train_sampler = build_loader_ddp(train_df, fold, 'train', cfg.batch_size, rank, world_size, seed=cfg.seed, device=device, cfg=cfg)
        val_loader, _ = build_loader_ddp(val_df, fold, 'val', cfg.batch_size, rank, world_size, seed=cfg.seed, device=device, cfg=cfg)

        best_metric = float("inf") if cfg.monitor_metric == "loss" else -1.0
        save_path = P(cfg.save_dir, f"fold{fold}_seed{cfg.seed}_{cfg.embed_backend}.pt")
        fgm = FGM(model, target_param_substrings=['encoder_T.src_emb'])

        for epoch in range(1, cfg.epochs + 1):
            y_true_tr, y_prob_tr, y_pred_tr, train_loss = train_one_epoch(
                model, train_loader, train_sampler, optimizer,
                bce, supcon, fgm, device, cfg.threshold, epoch,
                use_amp=cfg.use_amp, supcon_lambda=cfg.supcon_lambda, adv_epsilon=cfg.adv_epsilon
            )
            y_true_v, y_prob_v, y_pred_v, val_loss = validate(model, val_loader, bce, device, cfg.threshold, use_amp=cfg.use_amp)

            if is_main_process():
                perf = compute_performance(y_true_v, y_prob_v, y_pred_v)
                vals = [x for x in perf[:5] if not (isinstance(x, float) and math.isnan(x))]
                perf_avg = sum(vals) / len(vals) if vals else -1.0
                if cfg.monitor_metric == "auc":
                    current_metric = perf[0]
                elif cfg.monitor_metric == "aupr":
                    current_metric = perf[4]
                elif cfg.monitor_metric == "loss":
                    current_metric = val_loss
                else:
                    current_metric = perf_avg

                log(
                    f"Fold {fold} | Epoch {epoch}: TrainLoss={train_loss:.4f}, "
                    f"ValLoss={val_loss:.4f}, ValPerf={perf_avg:.4f}, "
                    f"Monitor({cfg.monitor_metric})={current_metric:.4f}"
                )

                improved = current_metric < best_metric if cfg.monitor_metric == "loss" else current_metric > best_metric
                if improved:
                    best_metric = current_metric
                    checkpoint = {
                        "model_state_dict": model.module.state_dict(),
                        "fold": fold,
                        "epoch": epoch,
                        "seed": cfg.seed,
                        "embed_backend": cfg.embed_backend,
                        "monitor_metric": cfg.monitor_metric,
                        "monitor_value": current_metric,
                        "val_loss": val_loss,
                        "val_metrics": {
                            "auc": perf[0],
                            "accuracy": perf[1],
                            "mcc": perf[2],
                            "f1": perf[3],
                            "aupr": perf[4],
                            "sensitivity": perf[5],
                            "specificity": perf[6],
                            "precision": perf[7],
                            "recall": perf[8],
                            "perf_avg": perf_avg,
                        },
                        "config": vars(cfg),
                    }
                    torch.save(checkpoint, save_path)
                    log(f"[Fold {fold}] Saved best at epoch {epoch} -> {save_path}")

        if is_main_process():
            log(f"[Fold {fold}] Best Val {cfg.monitor_metric} = {best_metric:.4f}")
            if cfg.monitor_metric == "loss":
                best_overall = min(best_overall, best_metric)
            else:
                best_overall = max(best_overall, best_metric)

    log(f"\n===== K-fold finished. Best Val {cfg.monitor_metric} across folds = {best_overall:.4f} =====")

    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
