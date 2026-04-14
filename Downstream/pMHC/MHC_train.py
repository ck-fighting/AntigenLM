import os
import sys
import math
import json
from datetime import timedelta
import random
import argparse
from typing import Tuple, List

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

from MHC_model import *
from feature_extractors import antigenLM_extract, extract_esm2_embeddings, extract_esmc_embeddings

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def log(msg: str) -> None:
    if is_main_process():
        print(msg, flush=True)

def binarize(probs: List[float], thr: float) -> List[int]:
    return [1 if p >= thr else 0 for p in probs]

def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _model_id_for_cache(cfg) -> str:
    if cfg.embed_backend == "esm2":
        return cfg.esm2_model_name
    if cfg.embed_backend == "esmc":
        return cfg.esmc_model_name
    if cfg.embed_backend == "AntigenLM":
        return os.path.basename(cfg.antigenLM_path.rstrip("/")) if cfg.antigenLM_path else "AntigenLM"
    return "unknown"

def extract_peptide_embeddings(pep_list, pep_max_len: int, device: str, cfg):
    max_len = cfg.embed_max_len_override if cfg.embed_max_len_override > 0 else pep_max_len
    if cfg.embed_backend == "esm2":
        return extract_esm2_embeddings(pep_list, model_name=cfg.esm2_model_name, device=device, max_len=max_len)
    elif cfg.embed_backend == "esmc":
        try:
            from esm.models.esmc import ESMC
        except Exception as e:
            raise ImportError("需要安装 `esm` 才能支持 ESMC。") from e
        
        client = ESMC.from_pretrained(cfg.esmc_model_name).to(device)
        return extract_esmc_embeddings(
            pep_list, client=client, device=device, batch_size=512, max_len=15, model_max_len=256
        )
    elif cfg.embed_backend == "AntigenLM":
        return antigenLM_extract(pep_list, model_name_or_path=cfg.AntigenLM_path, device=device, max_len=max_len)
    
    raise ValueError(f"未知的 embed_backend: {cfg.embed_backend}")

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features, labels):
        if features.dim() < 3:
            features = features.unsqueeze(1)
            
        features = F.normalize(features, dim=-1).view(features.shape[0], -1, features.shape[-1])
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        mask_self = torch.eye(labels.shape[0] * features.shape[1], dtype=torch.float32, device=features.device)
        mask_self = mask_self.repeat(features.shape[1], features.shape[1])
        
        mask = mask.repeat(features.shape[1], features.shape[1]) * (1 - mask_self)
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        return - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()

class FGM:
    def __init__(self, model: nn.Module, target_param_substrings: List[str]):
        self.model = model
        self.backup = {}
        self.targets = target_param_substrings
        
    @torch.no_grad()
    def attack(self, epsilon: float = 1.0):
        self.backup.clear()
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None and any(t in name for t in self.targets) and torch.norm(p.grad) != 0:
                self.backup[name] = p.data.clone()
                p.add_(epsilon * p.grad / torch.norm(p.grad))
                
    @torch.no_grad()
    def restore(self):
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup.clear()

class HLADataset(Dataset):
    def __init__(self, pep_embeds, hla_ids, labels):
        self.pep = pep_embeds
        self.hla = hla_ids
        self.y = labels
        
    def __len__(self):
        return self.y.shape[0]
        
    def __getitem__(self, idx):
        return self.pep[idx], self.hla[idx], self.y[idx]

def compute_lengths(series: pd.Series) -> int:
    return max(int(len(s)) for s in series.tolist())

def data_process_hla(data: pd.DataFrame, fold: int, type_: str, seed: int, device: torch.device, cfg):
    data = data.copy().rename(columns={"Peptide": "peptide", "antigen": "peptide", "hla": "HLA"})
    if not {"HLA", "peptide"}.issubset(data.columns):
        raise ValueError(f"Missing HLA/peptide in {set(data.columns)}")
        
    os.makedirs(cfg.emb_cache_dir, exist_ok=True)
    pep_max_len = compute_lengths(data.peptide)
    pep_list = [p.ljust(pep_max_len, "-") for p in data.peptide]
    
    suffix = f"{type_}_{fold}_{seed}" if type_ in ("train", "val") else type_
    cache_path = os.path.join(cfg.emb_cache_dir, f"cached_pep_embeddings_{suffix}_{cfg.embed_backend}.pt")

    if not os.path.exists(cache_path) and is_main_process():
        pep_emb = extract_peptide_embeddings(pep_list, pep_max_len, str(device), cfg)
        torch.save(pep_emb.detach().cpu(), cache_path)
        log(f"[Cache] Saved peptide embeddings to {cache_path}")
        
    if is_dist_avail_and_initialized():
        dist.barrier()

    pep_embeddings = torch.load(cache_path, map_location="cpu")
    if isinstance(pep_embeddings, np.ndarray):
        pep_embeddings = torch.from_numpy(pep_embeddings)
        
    _hla_max_len = hla_max_len if 'hla_max_len' in globals() else compute_lengths(data.HLA)
    hla_tensor = torch.LongTensor([
        [vocab.get(c, vocab.get('-', 0)) for c in seq.ljust(_hla_max_len, "-")]
        for seq in data.HLA
    ])
    
    label_tensor = torch.LongTensor(data.label.astype(int).tolist())
    assert pep_embeddings.shape[0] == hla_tensor.shape[0]
    
    return pep_embeddings, hla_tensor, label_tensor

def build_loader_ddp(data: pd.DataFrame, fold: int, type_: str, batch_size: int, rank: int, world_size: int, seed: int, device: torch.device, cfg):
    pep, hla, y = data_process_hla(data, fold, type_, seed, device, cfg)
    dataset = HLADataset(pep, hla, y)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=(type_ == 'train'), drop_last=False, seed=seed
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False
    )
    return loader, sampler

def compute_performance(y_true: List[int], y_prob: List[float], y_pred: List[int]):
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = float('nan')
        
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec) if len(reca) > 1 else float('nan')
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if (tp + tn + fp + fn) > 0 else 0.0
    
    if is_main_process():
        c_p, c_t = Counter(y_pred), Counter(y_true)
        log(f"tn={tn}, fp={fp}, fn={fn}, tp={tp}")
        log(f"y_pred: 0={c_p.get(0,0)} | 1={c_p.get(1,0)}")
        log(f"y_true: 0={c_t.get(0,0)} | 1={c_t.get(1,0)}")
        log(
            f"auc={roc_auc:.4f} | sens={sensitivity:.4f} | spec={specificity:.4f} | "
            f"acc={acc:.4f} | mcc={mcc:.4f} | precision={precision:.4f} | "
            f"recall={recall:.4f} | f1={f1:.4f} | aupr={aupr:.4f}"
        )
        
    return roc_auc, acc, mcc, f1, aupr, sensitivity, specificity, precision, recall

def train_one_epoch(
    model, train_loader, sampler, optimizer, bce, supcon, fgm, device, 
    threshold, epoch, use_amp, supcon_lambda, adv_epsilon
):
    model.train()
    sampler.set_epoch(epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    y_true_all, y_prob_all, loss_list = [], [], []
    
    for pep, hla, labels in train_loader:
        pep = pep.to(device, non_blocking=True)
        hla = hla.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _, pep_hla = model(pep, hla)
            loss = bce(logits.view(-1), labels.float()) + supcon_lambda * supcon(pep_hla, labels)
            
        scaler.scale(loss).backward()
        
        fgm.attack(epsilon=adv_epsilon)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_adv, _, pep_hla_adv = model(pep, hla)
            loss_adv = bce(logits_adv.view(-1), labels.float()) + supcon_lambda * supcon(pep_hla_adv.mean(dim=1), labels)
        scaler.scale(loss_adv).backward()
        fgm.restore()
        
        scaler.step(optimizer)
        scaler.update()
        
        y_true_all.extend(labels.tolist())
        y_prob_all.extend(torch.sigmoid(logits.view(-1)).detach().cpu().tolist())
        loss_list.append(loss.item())
        
    return y_true_all, y_prob_all, binarize(y_prob_all, threshold), float(np.mean(loss_list)) if loss_list else math.nan

@torch.no_grad()
def validate(model, val_loader, bce, device, threshold, use_amp):
    model.eval()
    y_true_all, y_prob_all, loss_list = [], [], []
    
    for pep, hla, labels in val_loader:
        pep = pep.to(device, non_blocking=True)
        hla = hla.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(pep, hla)[0].view(-1)
            loss = bce(logits, labels.float())
            
        y_true_all.extend(labels.tolist())
        y_prob_all.extend(torch.sigmoid(logits).detach().cpu().tolist())
        loss_list.append(loss.item())
        
    return y_true_all, y_prob_all, binarize(y_prob_all, threshold), float(np.mean(loss_list)) if loss_list else math.nan

def parse_args():
    p = argparse.ArgumentParser(description="DDP training for pMHC (HLA) binding")
    p.add_argument("--cv_dir", type=str, default="./data/cv_splits")
    p.add_argument("--emb_cache_dir", type=str, default="./data_cached")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--supcon_temp", type=float, default=0.07)
    p.add_argument("--supcon_lambda", type=float, default=0.05)
    p.add_argument("--adv_epsilon", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--find_unused", action="store_true", default=True)
    p.add_argument("--no-find_unused", dest="find_unused", action="store_false")
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--save_dir", type=str, default="../trained_model/pMHC")
    p.add_argument("--embed_backend", type=str, default="AntigenLM", choices=["esm2", "esmc", "AntigenLM"])
    p.add_argument("--esm2_model_name", type=str, default="esm2_t33_650M_UR50D")
    p.add_argument("--esmc_model_name", type=str, default="esmc_300m")
    p.add_argument("--AntigenLM_path", type=str, default="../../LLM/AntigenLM_300M_SS")
    p.add_argument("--embed_max_len_override", type=int, default=15)
    return p.parse_args()

def main():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("必须用 torchrun 启动。示例：torchrun --standalone --nproc_per_node=... train.py ...")
        
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=4))
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank} 超出可见 GPU 范围")
        
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    cfg = parse_args()
    setup_seed(cfg.seed)
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    log(json.dumps(vars(cfg), indent=2, ensure_ascii=False))

    bce = nn.BCEWithLogitsLoss()
    supcon = SupConLoss(temperature=cfg.supcon_temp, base_temperature=cfg.supcon_temp)
    best_overall = -1.0

    for fold in range(1, cfg.num_folds + 1):
        if is_main_process():
            log(f"\n========== Fold {fold}/{cfg.num_folds} ==========")

        # Re-initialize model and optimizer for each fold
        model = DDP(Mymodel_HLA().to(device), device_ids=[local_rank], find_unused_parameters=cfg.find_unused)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            
        train_csv = os.path.join(cfg.cv_dir, f"train_fold_{fold}.csv")
        val_csv = os.path.join(cfg.cv_dir, f"val_fold_{fold}.csv")
        if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
            if is_main_process():
                log(f"Missing fold files: {train_csv} or {val_csv}")
            dist.barrier()
            dist.destroy_process_group()
            sys.exit(1)
        
        train_loader, train_sampler = build_loader_ddp(
            pd.read_csv(train_csv), fold, 'train', cfg.batch_size, rank, world_size, cfg.seed, device, cfg
        )
        val_loader, _ = build_loader_ddp(
            pd.read_csv(val_csv), fold, 'val', cfg.batch_size, rank, world_size, cfg.seed, device, cfg
        )

        best_metric = -1.0
        save_path = os.path.join(cfg.save_dir, f"fold{fold}_seed{cfg.seed}_{cfg.embed_backend}.pt")
        fgm = FGM(model, target_param_substrings=['encoder_H.src_emb'])

        for epoch in range(1, cfg.epochs + 1):
            y_true_tr, y_prob_tr, y_pred_tr, train_loss = train_one_epoch(
                model, train_loader, train_sampler, optimizer, bce, supcon, fgm, device, 
                cfg.threshold, epoch, cfg.use_amp, cfg.supcon_lambda, cfg.adv_epsilon
            )
            y_true_v, y_prob_v, y_pred_v, val_loss = validate(model, val_loader, bce, device, cfg.threshold, cfg.use_amp)
            
            if is_main_process():
                perf = compute_performance(y_true_v, y_prob_v, y_pred_v)
                vals = [x for x in perf[:5] if not isinstance(x, float) or not math.isnan(x)]
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
