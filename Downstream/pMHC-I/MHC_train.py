import os
import math
import json
from datetime import timedelta
import random
import argparse
import glob
from typing import List

os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "0")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
DEFAULT_CV_DIR = "./data/cluster_aware_40_70_15_15_splits"

import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, auc, accuracy_score, f1_score,
    precision_recall_curve, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef
)

from MHC_model import *
from feature_extractors import (
    antigenLM_extract,
    antigenlm_last_hidden,
    extract_esm2_embeddings,
    extract_esmc_embeddings,
    load_antigenlm_model,
    load_antigenlm_tokenizer,
)

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

def setup_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

def peptide_embedding_dim(embed_backend: str, esm2_model_name: str = "", antigenlm_path: str = "") -> int:
    if embed_backend == "esmc":
        return 960
    if embed_backend == "esm2":
        return 1280 if "650M" in esm2_model_name or "t33" in esm2_model_name else 768
    if embed_backend == "AntigenLM" and antigenlm_path:
        config_path = os.path.join(antigenlm_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as handle:
                model_cfg = json.load(handle)
            for key in ("hidden_size", "d_model"):
                if key in model_cfg:
                    return int(model_cfg[key])
    return 768

def _esmc_weights_candidates(model_name: str) -> List[str]:
    if model_name != "esmc_300m":
        return []
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return [
        os.path.join(repo_root, "LLM", "ESMC_300M", "esmc_300m_2024_12_v0.pth"),
    ]

def resolve_esmc_weights_path(cfg) -> str:
    explicit = getattr(cfg, "esmc_weights_path", "")
    if explicit:
        return explicit
    for path in _esmc_weights_candidates(cfg.esmc_model_name):
        if os.path.exists(path):
            return path
    return ""

def load_esmc_client(cfg, device: str):
    from esm.models.esmc import ESMC

    weights_path = resolve_esmc_weights_path(cfg)
    if weights_path:
        if cfg.esmc_model_name != "esmc_300m":
            raise ValueError("--esmc_weights_path currently supports esmc_300m local weights only.")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"ESMC weights not found: {weights_path}")

        from esm.tokenization import get_esmc_model_tokenizers

        log(f"[ESMC] Load local weights: {weights_path}")
        model = ESMC(d_model=960, n_heads=15, n_layers=30, tokenizer=get_esmc_model_tokenizers()).eval()
        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        model = model.to(device)
        if str(device).startswith("cuda"):
            model = model.to(torch.bfloat16)
        log(f"[ESMC] Model ready on {device}")
        return model

    try:
        return ESMC.from_pretrained(cfg.esmc_model_name).to(device)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "ESMC package is installed, but the ESMC weight file is missing from the HuggingFace cache. "
            "Pass --esmc_weights_path /path/to/esmc_300m_2024_12_v0.pth or set HF cache correctly. "
            f"Original error: {e}"
        ) from e

def extract_peptide_embeddings(pep_list, pep_max_len: int, device: str, cfg):
    max_len = cfg.embed_max_len_override if cfg.embed_max_len_override > 0 else pep_max_len
    if cfg.embed_backend == "esm2":
        raw_peptides = [p.rstrip("-") for p in pep_list]
        return extract_esm2_embeddings(
            raw_peptides,
            model_name=cfg.esm2_model_name,
            device=device,
            max_len=max_len,
            batch_size=cfg.embed_extract_batch_size,
        )
    elif cfg.embed_backend == "esmc":
        try:
            from esm.models.esmc import ESMC
        except Exception as e:
            raise ImportError("需要安装 `esm` 才能支持 ESMC。") from e
        
        raw_peptides = [p.rstrip("-") for p in pep_list]
        client = load_esmc_client(cfg, device)
        pep_embeddings = extract_esmc_embeddings(
            raw_peptides,
            client=client,
            device=device,
            batch_size=cfg.embed_extract_batch_size,
            max_len=max_len,
            model_max_len=max(max_len + 2, 32),
        )
        del client
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        return pep_embeddings
    elif cfg.embed_backend == "AntigenLM":
        return antigenLM_extract(pep_list, model_name_or_path=cfg.AntigenLM_path, device=device, max_len=max_len)
    
    raise ValueError(f"未知的 embed_backend: {cfg.embed_backend}")

def embedding_cache_backend_name(embed_backend: str) -> str:
    if embed_backend == "esmc":
        return "esmc_raw"
    return embed_backend


def _safe_cache_id(value: str) -> str:
    raw = os.path.basename(os.path.normpath(str(value))) or str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)


def peptide_cache_backend_name(cfg) -> str:
    if cfg.embed_backend == "AntigenLM":
        return f"AntigenLM_{_safe_cache_id(cfg.AntigenLM_path)}"
    return embedding_cache_backend_name(cfg.embed_backend)


def _embedding_cache_prefix(cfg, suffix: str) -> str:
    cache_backend = peptide_cache_backend_name(cfg)
    return os.path.join(cfg.emb_cache_dir, f"cached_pep_embeddings_{suffix}_{cache_backend}")

def _load_memmap_cache(prefix: str):
    meta_path = prefix + ".npz"
    mmap_path = prefix + ".mmap"
    if not (os.path.exists(meta_path) and os.path.exists(mmap_path)):
        return None
    meta = np.load(meta_path)
    shape = tuple(int(x) for x in meta["shape"])
    dtype = str(meta["dtype"])
    return np.memmap(mmap_path, mode="r", dtype=dtype, shape=shape)

def _load_existing_embedding_cache(prefix: str):
    memmap_cache = _load_memmap_cache(prefix)
    if memmap_cache is not None:
        return memmap_cache
    old_pt_path = prefix + ".pt"
    if os.path.exists(old_pt_path):
        pep_embeddings = torch.load(old_pt_path, map_location="cpu")
        if isinstance(pep_embeddings, np.ndarray):
            pep_embeddings = torch.from_numpy(pep_embeddings)
        return pep_embeddings
    return None

class IndexedEmbeddingCache:
    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shape = (len(self.indices),) + tuple(base.shape[1:])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[int(self.indices[idx])]

def _extract_antigenlm_to_memmap(pep_list: list, pep_max_len: int, device: str, cfg, prefix: str):
    mmap_path = prefix + ".mmap"
    meta_path = prefix + ".npz"
    tmp_path = mmap_path + ".tmp"
    dtype = np.float16 if cfg.emb_cache_dtype == "float16" else np.float32

    tokenizer = load_antigenlm_tokenizer(cfg.AntigenLM_path)
    model = load_antigenlm_model(cfg.AntigenLM_path, device)
    model.eval()

    out = None
    with torch.no_grad():
        iterator = range(0, len(pep_list), cfg.embed_extract_batch_size)
        if is_main_process():
            iterator = tqdm(iterator, desc=f"Extract {os.path.basename(cfg.AntigenLM_path.rstrip('/'))}")
        for start in iterator:
            batch_seqs = [seq.rstrip("-")[:pep_max_len] for seq in pep_list[start : start + cfg.embed_extract_batch_size]]
            tokens = tokenizer(
                batch_seqs,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=pep_max_len + 2,
                add_special_tokens=True,
                return_special_tokens_mask=True,
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            last_hidden = antigenlm_last_hidden(model, input_ids, attention_mask)
            special_tokens_mask = tokens["special_tokens_mask"].to(device).bool()

            batch_embeddings = []
            for j, seq in enumerate(batch_seqs):
                residue_mask = attention_mask[j].bool() & (~special_tokens_mask[j])
                emb = last_hidden[j][residue_mask][:pep_max_len].detach().cpu()
                valid_len = min(len(seq), pep_max_len, emb.size(0))
                emb = emb[:valid_len]
                if valid_len < pep_max_len:
                    emb = F.pad(emb, (0, 0, 0, pep_max_len - valid_len))
                batch_embeddings.append(emb)
            batch_tensor = torch.stack(batch_embeddings, dim=0).float().numpy()

            if out is None:
                shape = (len(pep_list), pep_max_len, int(batch_tensor.shape[-1]))
                out = np.memmap(tmp_path, mode="w+", dtype=dtype, shape=shape)
            out[start : start + len(batch_seqs)] = batch_tensor.astype(dtype, copy=False)
            out.flush()

    if out is None:
        raise ValueError("No peptide sequences to embed")
    shape = out.shape
    del out
    os.replace(tmp_path, mmap_path)
    np.savez(meta_path, shape=np.array(shape, dtype=np.int64), dtype=np.array(str(np.dtype(dtype))))
    return _load_memmap_cache(prefix)

def get_peptide_embedding_cache(pep_list: list, pep_max_len: int, device: str, cfg, suffix: str):
    os.makedirs(cfg.emb_cache_dir, exist_ok=True)
    prefix = _embedding_cache_prefix(cfg, suffix)

    memmap_cache = _load_memmap_cache(prefix)
    if memmap_cache is not None:
        log(f"[Cache] Use memmap peptide embeddings: {prefix}.mmap")
        return memmap_cache

    old_pt_path = prefix + ".pt"
    if os.path.exists(old_pt_path):
        log(f"[Cache] Use tensor peptide embeddings: {old_pt_path}")
        pep_embeddings = torch.load(old_pt_path, map_location="cpu")
        if isinstance(pep_embeddings, np.ndarray):
            pep_embeddings = torch.from_numpy(pep_embeddings)
        return pep_embeddings

    if is_dist_avail_and_initialized() and not is_main_process():
        dist.barrier()
        memmap_cache = _load_memmap_cache(prefix)
        if memmap_cache is None:
            raise RuntimeError(f"Missing peptide embedding cache after barrier: {prefix}.mmap")
        return memmap_cache

    if cfg.embed_backend != "AntigenLM":
        pep_embeddings = extract_peptide_embeddings(pep_list, pep_max_len, device, cfg)
        torch.save(pep_embeddings.detach().cpu(), old_pt_path)
        log(f"[Cache] Saved peptide embeddings to {old_pt_path}")
        if is_dist_avail_and_initialized():
            dist.barrier()
        return pep_embeddings.detach().cpu()

    log(f"[Cache] Build memmap peptide embeddings: {prefix}.mmap")
    memmap_cache = _extract_antigenlm_to_memmap(pep_list, pep_max_len, device, cfg, prefix)
    log(f"[Cache] Saved memmap peptide embeddings to {prefix}.mmap")
    if is_dist_avail_and_initialized():
        dist.barrier()
    return memmap_cache

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
        pep = self.pep[idx]
        if isinstance(pep, np.ndarray):
            pep = torch.from_numpy(pep.astype(np.float32, copy=True))
        hla = self.hla[idx]
        if isinstance(hla, np.ndarray):
            hla = torch.from_numpy(hla.astype(np.int64, copy=True))
        return pep, hla, self.y[idx]

def compute_lengths(series: pd.Series) -> int:
    return max(int(len(s)) for s in series.tolist())

def load_hla_fasta(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"HLA FASTA not found: {path}")

    hla_to_seq = {}
    header, chunks = None, []

    def add_aliases(header_text: str, seq: str):
        for part in header_text.split("|"):
            part = part.strip()
            if not part:
                continue
            hla_to_seq[part] = seq
            if part.startswith("HLA-"):
                hla_to_seq[normalize_independent_mhc_name(part)] = seq
                hla_to_seq[standardize_hla_allele_name(part)] = seq
            elif len(part) >= 6 and part[0] in "ABC" and "*" in part:
                locus, fields = part.split("*", 1)
                hla_to_seq[f"HLA-{locus}{fields}"] = seq
                hla_to_seq[f"HLA-{locus}*{fields}"] = seq

    def flush():
        if not header:
            return
        seq = "".join(chunks).strip()
        if seq:
            add_aliases(header, seq)

    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    flush()
    return hla_to_seq

def get_cached_hla_pseudo_sequences(cfg) -> dict:
    if not getattr(cfg, "hla_pseudo_csv", ""):
        return {}
    if not hasattr(cfg, "_hla_pseudo_sequences"):
        cfg._hla_pseudo_sequences = load_hla_pseudo_sequences(cfg.hla_pseudo_csv)
    return cfg._hla_pseudo_sequences

def get_cached_hla_fasta(cfg) -> dict:
    if not getattr(cfg, "hla_fasta", ""):
        return {}
    if not hasattr(cfg, "_hla_fasta_sequences"):
        cfg._hla_fasta_sequences = load_hla_fasta(cfg.hla_fasta)
        log(f"[HLA] Loaded {len(cfg._hla_fasta_sequences)} FASTA aliases from {cfg.hla_fasta}")
    return cfg._hla_fasta_sequences

def normalize_independent_mhc_name(name: str) -> str:
    name = str(name).strip()
    if name.startswith(("BoLA-", "DLA-", "Mamu-")):
        return name
    if not name.startswith("HLA-"):
        return name
    body = name[4:]
    if "*" in body:
        locus, fields = body.split("*", 1)
        return f"HLA-{locus}{fields}"
    return name

def standardize_hla_allele_name(name: str) -> str:
    name = str(name).strip()
    if not name.startswith("HLA-"):
        return name
    body = name[4:]
    if "*" in body:
        return name
    if len(body) >= 6 and body[0] in "ABC" and body[1:3].isdigit() and body[3] == ":":
        return f"HLA-{body[0]}*{body[1:]}"
    return name

def looks_like_hla_pseudo_sequence(value: str) -> bool:
    value = str(value).strip()
    if len(value) < 20 or "*" in value or ":" in value:
        return False
    return all(ch.isalpha() or ch == "-" for ch in value)

def load_hla_pseudo_sequences(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"HLA pseudo CSV not found: {path}")

    df = pd.read_csv(path, usecols=["MHC_Restriction_Name", "HLA"])
    df["MHC_Restriction_Name"] = df["MHC_Restriction_Name"].astype(str).str.strip()
    df["HLA"] = df["HLA"].astype(str).str.strip()
    mapping = df.drop_duplicates("MHC_Restriction_Name").set_index("MHC_Restriction_Name")["HLA"].to_dict()
    log(f"[HLA] Loaded {len(mapping)} pseudo sequences from {path}")
    return mapping

def normalize_hla_dataframe(data: pd.DataFrame, cfg) -> pd.DataFrame:
    data = data.copy().rename(
        columns={
            "Peptide": "peptide",
            "antigen": "peptide",
            "pep": "peptide",
            "hla": "HLA",
            "mhc": "HLA",
            "Label": "label",
        }
    )
    if "HLA" not in data.columns and "MHC_Restriction_Name" in data.columns:
        data["HLA"] = data["MHC_Restriction_Name"]
    if not {"HLA", "peptide", "label"}.issubset(data.columns):
        raise ValueError(f"Missing HLA/peptide/label in {set(data.columns)}")

    data["_source_row_idx"] = np.arange(len(data), dtype=np.int64)
    data["peptide"] = data["peptide"].astype(str).str.strip()
    data["HLA"] = data["HLA"].astype(str).str.strip()
    hla_name_col = "MHC_Restriction_Name" if "MHC_Restriction_Name" in data.columns else "HLA"
    data["_hla_name"] = data[hla_name_col].astype(str).str.strip()
    data["label"] = data["label"].astype(int)
    data = data.loc[
        (data["peptide"].str.len() > 0)
        & (data["HLA"].str.len() > 0)
        & (data["_hla_name"].str.len() > 0)
    ].copy()

    if cfg.hla_pseudo_csv:
        hla_to_seq = get_cached_hla_pseudo_sequences(cfg)
        normalized = data["_hla_name"].map(standardize_hla_allele_name)
        mapped = normalized.map(hla_to_seq)
        missing = mapped.isna()
        fallback = missing & data["HLA"].map(looks_like_hla_pseudo_sequence)
        if fallback.any():
            mapped.loc[fallback] = data.loc[fallback, "HLA"]
            missing = mapped.isna()
        if missing.any():
            missing_counts = data.loc[missing, "_hla_name"].value_counts()
            msg = (
                f"Dropping {int(missing.sum())} rows with no 34-aa pseudo sequence in {cfg.hla_pseudo_csv}: "
                + ", ".join(f"{k}={v}" for k, v in missing_counts.head(30).items())
            )
            log(msg)
            if not cfg.drop_missing_hla_sequence:
                raise ValueError(msg)
            data = data.loc[~missing].copy()
            mapped = mapped.loc[~missing]
        data["HLA"] = mapped.values
    elif cfg.hla_fasta:
        hla_to_seq = get_cached_hla_fasta(cfg)
        normalized = data["_hla_name"].map(normalize_independent_mhc_name)
        mapped = normalized.map(hla_to_seq)
        missing = mapped.isna()
        if missing.any():
            missing_counts = data.loc[missing, "_hla_name"].value_counts()
            msg = (
                f"Dropping {int(missing.sum())} rows with no sequence in {cfg.hla_fasta}: "
                + ", ".join(f"{k}={v}" for k, v in missing_counts.head(20).items())
            )
            log(msg)
            if not cfg.drop_missing_hla_sequence:
                raise ValueError(msg)
            data = data.loc[~missing].copy()
            mapped = mapped.loc[~missing]
        data["HLA"] = mapped.values

    return data[["_source_row_idx", "peptide", "HLA", "label"]].reset_index(drop=True)

def data_process_hla(data: pd.DataFrame, fold: int, type_: str, seed: int, device: torch.device, cfg):
    log(f"[Data] {type_}: normalize rows={len(data)}")
    data = normalize_hla_dataframe(data, cfg)
    log(f"[Data] {type_}: usable rows={len(data)} unique_peptides={data.peptide.nunique()} unique_hla={data.HLA.nunique()}")
        
    os.makedirs(cfg.emb_cache_dir, exist_ok=True)
    pep_data_max_len = compute_lengths(data.peptide)
    pep_embed_len = cfg.embed_max_len_override if cfg.embed_max_len_override > 0 else pep_data_max_len
    pep_list = [p.ljust(pep_embed_len, "-") for p in data.peptide]
    
    suffix = f"{type_}_{fold}_{seed}"
    prefix = _embedding_cache_prefix(cfg, suffix)
    base_cache = _load_existing_embedding_cache(prefix)
    row_indices = data["_source_row_idx"].to_numpy(dtype=np.int64)
    if base_cache is not None and len(base_cache) == len(data):
        pep_embeddings = base_cache
        log(f"[Cache] Use peptide embeddings: {prefix} rows={len(data)}")
    elif base_cache is not None and len(row_indices) > 0 and int(row_indices.max()) < len(base_cache):
        pep_embeddings = IndexedEmbeddingCache(base_cache, row_indices)
        log(f"[Cache] Use indexed peptide embeddings: {prefix} rows={len(data)}/{len(base_cache)}")
    else:
        filtered_suffix = f"{suffix}_pseudo34"
        if base_cache is not None:
            log(
                f"[Cache] Existing cache shape mismatch for {prefix}: "
                f"cache_rows={len(base_cache)} data_rows={len(data)} max_source_row={int(row_indices.max()) if len(row_indices) else -1}; "
                f"building {filtered_suffix}"
            )
        pep_embeddings = get_peptide_embedding_cache(pep_list, pep_embed_len, str(device), cfg, filtered_suffix)
        
    log(f"[Data] {type_}: tokenize HLA")
    _hla_max_len = hla_max_len if 'hla_max_len' in globals() else compute_lengths(data.HLA)
    hla_codes, unique_hla = pd.factorize(data.HLA, sort=False)
    unique_hla_tokens = np.asarray(
        [[vocab.get(c, vocab.get('-', 0)) for c in seq.ljust(_hla_max_len, "-")] for seq in unique_hla],
        dtype=np.uint8,
    )
    hla_tensor = unique_hla_tokens[hla_codes]
    
    label_tensor = torch.LongTensor(data.label.astype(int).tolist())
    assert pep_embeddings.shape[0] == hla_tensor.shape[0], (
        f"peptide/cache rows {pep_embeddings.shape[0]} != HLA rows {hla_tensor.shape[0]}"
    )
    log(f"[Data] {type_}: ready pep_shape={tuple(pep_embeddings.shape)} hla_shape={tuple(hla_tensor.shape)}")
    
    return pep_embeddings, hla_tensor, label_tensor

def build_loader_ddp(data: pd.DataFrame, fold: int, type_: str, batch_size: int, rank: int, world_size: int, seed: int, device: torch.device, cfg):
    pep, hla, y = data_process_hla(data, fold, type_, seed, device, cfg)
    dataset = HLADataset(pep, hla, y)
    shuffle = type_.endswith("train")
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False, seed=seed
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False
    )
    return loader, sampler

def compute_performance(y_true: List[int], y_prob: List[float], y_pred: List[int]):
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
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if (tp + tn + fp + fn) > 0 else 0.0
    
    if is_main_process():
        c_p, c_t = Counter(y_pred), Counter(y_true)
        log(f"tn={tn}, fp={fp}, fn={fn}, tp={tp}")
        log(f"y_pred: 0={c_p.get(0, 0)} | 1={c_p.get(1, 0)}")
        log(f"y_true: 0={c_t.get(0, 0)} | 1={c_t.get(1, 0)}")
        log(
            f"auc={roc_auc:.4f} | sens={sensitivity:.4f} | spec={specificity:.4f} | "
            f"acc={acc:.4f} | mcc={mcc:.4f} | precision={precision:.4f} | "
            f"recall={recall:.4f} | f1={f1:.4f} | aupr={aupr:.4f}"
        )
        
    return roc_auc, acc, mcc, f1, aupr, sensitivity, specificity, precision, recall

def gather_eval_outputs(y_true, y_prob, y_pred):
    if not is_dist_avail_and_initialized():
        return y_true, y_prob, y_pred

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, (y_true, y_prob, y_pred))
    if not is_main_process():
        return y_true, y_prob, y_pred

    all_true, all_prob, all_pred = [], [], []
    for true_part, prob_part, pred_part in gathered:
        all_true.extend(true_part)
        all_prob.extend(prob_part)
        all_pred.extend(pred_part)
    return all_true, all_prob, all_pred

def make_bce_loss(labels_source, device: torch.device, cfg):
    if not cfg.use_pos_weight:
        return nn.BCEWithLogitsLoss()

    if isinstance(labels_source, pd.DataFrame):
        labels = labels_source["label"].astype(int).to_numpy()
    elif torch.is_tensor(labels_source):
        labels = labels_source.detach().cpu().numpy().astype(int)
    else:
        labels = np.asarray(labels_source, dtype=int)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0:
        log("[Loss] No positive labels found; using unweighted BCE")
        return nn.BCEWithLogitsLoss()

    pos_weight = neg / pos
    log(f"[Loss] BCE pos_weight={pos_weight:.4f} (neg={neg}, pos={pos})")
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 0.0, max_pairs: int = 0):
    scores = logits.view(-1).float()
    labels = labels.view(-1)
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return scores.sum() * 0.0

    num_pairs = int(pos_scores.numel() * neg_scores.numel())
    if max_pairs > 0 and num_pairs > max_pairs:
        pos_idx = torch.randint(pos_scores.numel(), (max_pairs,), device=scores.device)
        neg_idx = torch.randint(neg_scores.numel(), (max_pairs,), device=scores.device)
        diffs = neg_scores[neg_idx] - pos_scores[pos_idx] + margin
    else:
        diffs = neg_scores.unsqueeze(0) - pos_scores.unsqueeze(1) + margin
    return F.softplus(diffs).mean()

def reduce_train_stats(stats: dict, device: torch.device) -> dict:
    keys = ["loss_sum", "bce_sum", "rank_sum", "supcon_sum", "adv_sum", "prob_sum", "pos_sum", "count"]
    values = torch.tensor([float(stats.get(k, 0.0)) for k in keys], dtype=torch.float64, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {k: float(v) for k, v in zip(keys, values.cpu().tolist())}

def average_stats(stats: dict) -> dict:
    count = max(float(stats.get("count", 0.0)), 1.0)
    return {
        "loss": stats.get("loss_sum", 0.0) / count,
        "bce": stats.get("bce_sum", 0.0) / count,
        "rank": stats.get("rank_sum", 0.0) / count,
        "supcon": stats.get("supcon_sum", 0.0) / count,
        "adv": stats.get("adv_sum", 0.0) / count,
        "prob": stats.get("prob_sum", 0.0) / count,
        "pos_rate": stats.get("pos_sum", 0.0) / count,
    }

def empty_stats() -> dict:
    return {
        "loss_sum": 0.0,
        "bce_sum": 0.0,
        "rank_sum": 0.0,
        "supcon_sum": 0.0,
        "adv_sum": 0.0,
        "prob_sum": 0.0,
        "pos_sum": 0.0,
        "count": 0.0,
    }

def make_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)

def cuda_autocast(use_amp: bool):
    try:
        return torch.amp.autocast("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=use_amp)

def train_one_epoch(
    model, train_loader, sampler, optimizer, bce, supcon, fgm, device, 
    epoch, use_amp, supcon_lambda, adv_epsilon, log_interval, grad_clip_norm,
    rank_lambda, rank_margin, rank_max_pairs
):
    model.train()
    sampler.set_epoch(epoch)
    scaler = make_grad_scaler(use_amp)
    
    total_batches = len(train_loader)
    epoch_stats = empty_stats()
    interval_stats = empty_stats()
    
    for step, (pep, hla, labels) in enumerate(train_loader, start=1):
        pep = pep.to(device, non_blocking=True).float()
        hla = hla.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_n = int(labels.numel())
        
        optimizer.zero_grad(set_to_none=True)
        with cuda_autocast(use_amp):
            logits, _, pep_hla = model(pep, hla)
            bce_loss = bce(logits.view(-1), labels.float())
            rank_loss = torch.zeros((), device=device)
            supcon_loss = torch.zeros((), device=device)
            loss = bce_loss
            if rank_lambda > 0:
                rank_loss = pairwise_ranking_loss(logits.view(-1), labels, rank_margin, rank_max_pairs)
                loss = loss + rank_lambda * rank_loss
            if supcon_lambda > 0:
                supcon_loss = supcon(pep_hla, labels)
                loss = loss + supcon_lambda * supcon_loss
            
        scaler.scale(loss).backward()
        
        adv_loss = torch.zeros((), device=device)
        if adv_epsilon > 0:
            fgm.attack(epsilon=adv_epsilon)
            with cuda_autocast(use_amp):
                logits_adv, _, pep_hla_adv = model(pep, hla)
                loss_adv = bce(logits_adv.view(-1), labels.float())
                if supcon_lambda > 0:
                    loss_adv = loss_adv + supcon_lambda * supcon(pep_hla_adv.mean(dim=1), labels)
                adv_loss = loss_adv
            scaler.scale(loss_adv).backward()
            fgm.restore()

        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            total_loss_value = (loss + adv_loss).detach().float().item()
            bce_value = bce_loss.detach().float().item()
            rank_value = (rank_lambda * rank_loss).detach().float().item() if rank_lambda > 0 else 0.0
            supcon_value = (supcon_lambda * supcon_loss).detach().float().item() if supcon_lambda > 0 else 0.0
            adv_value = adv_loss.detach().float().item() if adv_epsilon > 0 else 0.0
            prob_sum = torch.sigmoid(logits.view(-1)).detach().float().sum().item()
            pos_sum = labels.detach().float().sum().item()

        batch_stats = {
            "loss_sum": total_loss_value * batch_n,
            "bce_sum": bce_value * batch_n,
            "rank_sum": rank_value * batch_n,
            "supcon_sum": supcon_value * batch_n,
            "adv_sum": adv_value * batch_n,
            "prob_sum": prob_sum,
            "pos_sum": pos_sum,
            "count": float(batch_n),
        }
        for key, value in batch_stats.items():
            epoch_stats[key] += value
            interval_stats[key] += value

        if log_interval > 0 and step % log_interval == 0:
            interval_global = reduce_train_stats(interval_stats, device)
            epoch_global = reduce_train_stats(epoch_stats, device)
            interval_avg = average_stats(interval_global)
            epoch_avg = average_stats(epoch_global)
            log(
                f"Epoch {epoch} | step {step}/{total_batches} | "
                f"interval_loss={interval_avg['loss']:.4f} "
                f"epoch_avg_loss={epoch_avg['loss']:.4f} "
                f"bce={interval_avg['bce']:.4f} "
                f"rank={interval_avg['rank']:.4f} "
                f"supcon={interval_avg['supcon']:.4f} "
                f"adv={interval_avg['adv']:.4f} "
                f"pos_rate={interval_avg['pos_rate']:.4f} "
                f"prob_mean={interval_avg['prob']:.4f}"
            )
            interval_stats = empty_stats()

    epoch_global = reduce_train_stats(epoch_stats, device)
    return average_stats(epoch_global)

@torch.no_grad()
def evaluate(model, data_loader, bce, device, threshold, use_amp):
    model.eval()
    y_true_all, y_prob_all, loss_list = [], [], []
    
    for batch in data_loader:
        pep, hla, labels = batch[:3]
        pep = pep.to(device, non_blocking=True).float()
        hla = hla.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with cuda_autocast(use_amp):
            logits = model(pep, hla)[0].view(-1)
            loss = bce(logits, labels.float())
            
        y_true_all.extend(labels.tolist())
        y_prob_all.extend(torch.sigmoid(logits).detach().cpu().tolist())
        loss_list.append(loss.item())
        
    return y_true_all, y_prob_all, binarize(y_prob_all, threshold), float(np.mean(loss_list)) if loss_list else math.nan

DEFAULTS = {
    "hla_pseudo_csv": "./data/dataset_all.csv",
    "hla_fasta": "",
    "drop_missing_hla_sequence": True,
    "emb_cache_dir": "./data_cached",
    "supcon_temp": 0.07,
    "supcon_lambda": 0.05,
    "rank_lambda": 0.0,
    "rank_margin": 0.0,
    "rank_max_pairs": 0,
    "adv_epsilon": 1.0,
    "eval_every": 1,
    "early_stop_patience": 10,
    "eval_max_rows": 0,
    "train_log_interval": 500,
    "grad_clip_norm": 1.0,
    "threshold": 0.5,
    "use_pos_weight": True,
    "find_unused": True,
    "num_workers": 2,
    "pin_memory": False,
    "use_amp": False,
    "deterministic": False,
    "esm2_model_name": "esm2_t33_650M_UR50D",
    "esmc_model_name": "esmc_300m",
    "esmc_weights_path": "../../LLM/ESMC_300M/esmc_300m_2024_12_v0.pth",
    "embed_max_len_override": 15,
    "embed_extract_batch_size": 512,
    "emb_cache_dtype": "float16",
}

def parse_args():
    p = argparse.ArgumentParser(
        description="Train pMHC-I binding model on independent split or train/val CV folds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.set_defaults(**DEFAULTS)
    p.add_argument("--mode", choices=["independent", "cv", "all"], default="cv")
    p.add_argument("--train_csv", default="./data/Independent data/el_train.csv")
    p.add_argument("--test_csv", default="./data/Independent data/el_test.csv")
    p.add_argument("--cv_dir", default=DEFAULT_CV_DIR)
    p.add_argument("--folds", type=int, nargs="*", default=None, help="CV folds to run; empty means all discovered folds.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--save_dir", default="../trained_model/pMHC-I")
    p.add_argument("--emb_cache_dir", default=DEFAULTS["emb_cache_dir"])
    p.add_argument("--hla_pseudo_csv", default=DEFAULTS["hla_pseudo_csv"])
    p.add_argument("--embed_backend", choices=["esm2", "esmc", "AntigenLM"], default="AntigenLM")
    p.add_argument("--AntigenLM_path", default="../../LLM/AntigenLM")
    p.add_argument("--esmc_weights_path", default=DEFAULTS["esmc_weights_path"])
    p.add_argument("--eval_every", type=int, default=DEFAULTS["eval_every"])
    p.add_argument("--early_stop_patience", type=int, default=DEFAULTS["early_stop_patience"], help="Stop a split after this many epochs without AUC improvement; <=0 disables.")
    p.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--embed_extract_batch_size", type=int, default=DEFAULTS["embed_extract_batch_size"])
    p.add_argument("--emb_cache_dtype", choices=["float16", "float32"], default=DEFAULTS["emb_cache_dtype"])
    p.add_argument("--use_amp", action="store_true", default=DEFAULTS["use_amp"])
    p.add_argument("--deterministic", action="store_true", default=DEFAULTS["deterministic"])
    return p.parse_args()

def resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(SCRIPT_DIR, path))

def normalize_config_paths(cfg) -> None:
    for name in ("train_csv", "test_csv", "cv_dir", "emb_cache_dir", "save_dir", "hla_pseudo_csv", "hla_fasta", "esmc_weights_path"):
        setattr(cfg, name, resolve_path(getattr(cfg, name)))
    if cfg.AntigenLM_path and (
        cfg.AntigenLM_path.startswith((".", "/")) or os.path.exists(resolve_path(cfg.AntigenLM_path))
    ):
        cfg.AntigenLM_path = resolve_path(cfg.AntigenLM_path)

def _fold_id_from_path(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.rsplit("_", 1)[-1])

def discover_cv_splits(cv_dir: str, folds=None):
    if folds:
        fold_ids = sorted(set(folds))
    else:
        train_files = glob.glob(os.path.join(cv_dir, "train_fold_*.csv"))
        fold_ids = sorted(_fold_id_from_path(path) for path in train_files)
    if not fold_ids:
        raise FileNotFoundError(f"No train_fold_*.csv found in {cv_dir}")

    splits = []
    for fold in fold_ids:
        train_csv = os.path.join(cv_dir, f"train_fold_{fold}.csv")
        val_csv = os.path.join(cv_dir, f"val_fold_{fold}.csv")
        if not os.path.exists(train_csv) or not os.path.exists(val_csv):
            raise FileNotFoundError(f"Missing CV split files: {train_csv} / {val_csv}")
        splits.append((f"cv_fold_{fold}", fold, train_csv, val_csv, f"cv{fold}_train", f"cv{fold}_val"))
    return splits

def build_split_plan(cfg):
    splits = []
    if cfg.mode in ("independent", "all"):
        if not os.path.exists(cfg.train_csv) or not os.path.exists(cfg.test_csv):
            raise FileNotFoundError(f"Missing independent split files: {cfg.train_csv} / {cfg.test_csv}")
        splits.append(("independent", 1, cfg.train_csv, cfg.test_csv, "independent_train", "independent_test"))
    if cfg.mode in ("cv", "all"):
        splits.extend(discover_cv_splits(cfg.cv_dir, cfg.folds))
    return splits

def main():
    cfg = parse_args()
    normalize_config_paths(cfg)

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
    
    setup_seed(cfg.seed, cfg.deterministic)
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    log(json.dumps(vars(cfg), indent=2, ensure_ascii=False))

    supcon = SupConLoss(temperature=cfg.supcon_temp, base_temperature=cfg.supcon_temp)
    best_overall = -1.0
    split_files = build_split_plan(cfg)

    for split_idx, (split_name, fold, train_csv, eval_csv, train_tag, eval_tag) in enumerate(split_files, start=1):
        if is_main_process():
            log(f"\n========== {split_name} ({split_idx}/{len(split_files)}) ==========")

        pep_dim = peptide_embedding_dim(cfg.embed_backend, cfg.esm2_model_name, cfg.AntigenLM_path)
        model = DDP(Mymodel_HLA(pep_dim=pep_dim).to(device), device_ids=[local_rank], find_unused_parameters=cfg.find_unused)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)
        
        train_df = pd.read_csv(train_csv)
        train_loader, train_sampler = build_loader_ddp(
            train_df, fold, train_tag, cfg.batch_size, rank, world_size, cfg.seed, device, cfg
        )
        bce = make_bce_loss(train_loader.dataset.y, device, cfg)
        eval_df = pd.read_csv(eval_csv)
        if cfg.eval_max_rows > 0:
            eval_df = eval_df.head(cfg.eval_max_rows)
        eval_loader, _ = build_loader_ddp(
            eval_df, fold, eval_tag, cfg.batch_size, rank, world_size, cfg.seed, device, cfg
        )

        best_auc = -1.0
        no_auc_improve_epochs = 0
        last_auc_eval_epoch = 0
        save_path = os.path.join(cfg.save_dir, f"{split_name}_seed{cfg.seed}_{cfg.embed_backend}.pt")
        fgm = FGM(model, target_param_substrings=['encoder_H.src_emb'])
        eval_label = "Val" if split_name.startswith("cv_fold_") else "Test"

        for epoch in range(1, cfg.epochs + 1):
            train_stats = train_one_epoch(
                model, train_loader, train_sampler, optimizer, bce, supcon, fgm, device, 
                epoch,
                cfg.use_amp,
                cfg.supcon_lambda,
                cfg.adv_epsilon,
                cfg.train_log_interval,
                cfg.grad_clip_norm,
                cfg.rank_lambda,
                cfg.rank_margin,
                cfg.rank_max_pairs,
            )
            should_eval = cfg.eval_every > 0 and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs)
            if should_eval:
                y_true_te, y_prob_te, y_pred_te, eval_loss = evaluate(model, eval_loader, bce, device, cfg.threshold, cfg.use_amp)
                y_true_te, y_prob_te, y_pred_te = gather_eval_outputs(y_true_te, y_prob_te, y_pred_te)
                stop_training = torch.tensor([0], dtype=torch.int32, device=device)
                
                if is_main_process():
                    perf = compute_performance(y_true_te, y_prob_te, y_pred_te)
                    auc_metric = perf[0]
                    auc_for_compare = auc_metric if not math.isnan(auc_metric) else -1.0
                    
                    log(
                        f"{split_name} | Epoch {epoch}: "
                        f"TrainLoss={train_stats['loss']:.4f} "
                        f"TrainBCE={train_stats['bce']:.4f} "
                        f"TrainRank={train_stats['rank']:.4f} "
                        f"TrainSupCon={train_stats['supcon']:.4f} "
                        f"TrainAdv={train_stats['adv']:.4f} "
                        f"TrainProb={train_stats['prob']:.4f} "
                        f"TrainPosRate={train_stats['pos_rate']:.4f} "
                        f"{eval_label}Loss={eval_loss:.4f}, {eval_label}AUC={auc_metric:.4f}"
                    )
                    if auc_for_compare > best_auc:
                        best_auc = auc_for_compare
                        no_auc_improve_epochs = 0
                        torch.save(model.module.state_dict(), save_path)
                        log(f"[{split_name}] Saved best AUC at epoch {epoch} -> {save_path}")
                    else:
                        no_auc_improve_epochs += max(1, epoch - last_auc_eval_epoch)
                        log(
                            f"[{split_name}] AUC did not improve for "
                            f"{no_auc_improve_epochs}/{cfg.early_stop_patience} epochs "
                            f"(best={best_auc:.4f})"
                        )
                    last_auc_eval_epoch = epoch
                    if cfg.early_stop_patience > 0 and no_auc_improve_epochs >= cfg.early_stop_patience:
                        stop_training.fill_(1)
                        log(
                            f"[{split_name}] Early stop at epoch {epoch}: "
                            f"{cfg.early_stop_patience} epochs without AUC improvement"
                        )
                if is_dist_avail_and_initialized():
                    dist.broadcast(stop_training, src=0)
                if int(stop_training.item()) == 1:
                    break
            elif is_main_process():
                log(
                    f"{split_name} | Epoch {epoch}: "
                    f"TrainLoss={train_stats['loss']:.4f} "
                    f"TrainBCE={train_stats['bce']:.4f} "
                    f"TrainRank={train_stats['rank']:.4f} "
                    f"TrainSupCon={train_stats['supcon']:.4f} "
                    f"TrainAdv={train_stats['adv']:.4f} "
                    f"TrainProb={train_stats['prob']:.4f} "
                    f"TrainPosRate={train_stats['pos_rate']:.4f} | eval skipped"
                )
                torch.save(model.module.state_dict(), save_path)
                log(f"[{split_name}] Saved latest at epoch {epoch} -> {save_path}")

        if is_main_process():
            log(f"[{split_name}] Best {eval_label} AUC = {best_auc:.4f}")
            best_overall = max(best_overall, best_auc)

    if is_main_process():
        log(f"\n===== Training finished. Best Eval AUC = {best_overall:.4f} =====")
        
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
