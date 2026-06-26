import os
import argparse
import glob
import json
from collections import Counter
from typing import Optional
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
P = os.path.join
DEFAULT_CV_DIR = "./data/cluster_aware_40_70_15_15_splits"

class Mymodel_HLA_ESM(nn.Module):
    def __init__(self, d_model=128, hla_dim=1280, dropout=0.16, pep_dim=768):
        super().__init__()
        self.pep_proj = nn.Linear(pep_dim, d_model)
        self.hla_proj = nn.Sequential(
            nn.Linear(hla_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(15 * d_model + d_model, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, pep_emb, hla_emb):
        pep = self.pep_proj(pep_emb.float())
        pep_flat = pep.contiguous().view(pep.shape[0], -1)
        hla_vec = self.hla_proj(hla_emb.float())
        fusion = torch.cat([pep_flat, hla_vec], dim=1)
        logits = self.fuse(fusion)
        return logits, None, fusion

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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

def threshold_predictions(y_prob, threshold: float):
    return (np.asarray(y_prob) >= threshold).astype(np.int32).tolist()

def find_best_threshold(y_true, y_prob, metric: str = "mcc"):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_true) == 0:
        return 0.5, [], {"mcc": 0.0, "f1": 0.0}

    order = np.argsort(-y_prob, kind="mergesort")
    scores = y_prob[order]
    labels = y_true[order]
    unique_ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]

    tp = np.cumsum(labels)[unique_ends].astype(np.float64)
    fp = (unique_ends + 1).astype(np.float64) - tp
    positives = float(labels.sum())
    negatives = float(len(labels) - labels.sum())
    fn = positives - tp
    tn = negatives - fp

    f1_denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, f1_denom, out=np.zeros_like(tp), where=f1_denom > 0)

    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, mcc_denom, out=np.zeros_like(tp), where=mcc_denom > 0)

    if metric == "f1":
        primary, secondary = f1, mcc
    else:
        primary, secondary = mcc, f1

    best_primary = np.nanmax(primary)
    best_candidates = np.flatnonzero(primary == best_primary)
    best_idx = best_candidates[np.nanargmax(secondary[best_candidates])]
    threshold = float(scores[unique_ends[best_idx]])
    y_bin = threshold_predictions(y_prob, threshold)
    return threshold, y_bin, {"mcc": float(mcc[best_idx]), "f1": float(f1[best_idx])}

def performance(y_true, y_prob, y_bin, threshold=None):
    acc = accuracy_score(y_true, y_pred=y_bin)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_bin, zero_division=0)
    recall = recall_score(y_true, y_bin, zero_division=0)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float('nan')
        
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec) if len(reca) > 1 else float('nan')
    mcc = matthews_corrcoef(y_true, y_bin) if (tp + tn + fp + fn) > 0 else 0.0
    pred_counter = Counter(y_bin)
    true_counter = Counter(y_true)

    if threshold is not None:
        print(f'threshold={threshold:.6f}')
    print(f'tn={tn}, fp={fp}, fn={fn}, tp={tp}')
    print(f'y_pred: 0={pred_counter[0]} | 1={pred_counter[1]}')
    print(f'y_true: 0={true_counter[0]} | 1={true_counter[1]}')
    print(f'auc={roc_auc:.4f}|sensitivity={sensitivity:.4f}|specificity={specificity:.4f}|acc={acc:.4f}|mcc={mcc:.4f}')
    print(f'precision={precision:.4f}|recall={recall:.4f}|f1={f1:.4f}|aupr={aupr:.4f}')
    
    return {
        'auc': roc_auc, 'acc': acc, 'mcc': mcc, 'f1': f1, 'aupr': aupr,
        'threshold': threshold,
        'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'recall': recall,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'pred_0': pred_counter[0], 'pred_1': pred_counter[1],
        'true_0': true_counter[0], 'true_1': true_counter[1]
    }

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
    print(f"[HLA] Loaded {len(mapping)} pseudo sequences from {path}")
    return mapping

def load_hla_esm_cache(path: str) -> dict:
    if not path:
        raise ValueError("--hla_esm_cache_path is required when --hla_input esm_cache")
    if not os.path.exists(path):
        raise FileNotFoundError(f"HLA ESM cache not found: {path}")
    cache = torch.load(path, map_location="cpu")
    names = list(cache["allele_names"])
    embeddings = cache["embeddings"]
    mapping = {name: i for i, name in enumerate(names)}
    print(f"[HLA-ESM] Loaded cache {path}: n={len(names)} embeddings_shape={tuple(embeddings.shape)}")
    return {"embeddings": embeddings, "mapping": mapping}

def map_hla_name_to_esm_cache(raw_name: str, name_to_idx: dict):
    name = str(raw_name).strip()
    candidates = [name, normalize_independent_mhc_name(name), standardize_hla_allele_name(name)]
    for candidate in candidates:
        if candidate in name_to_idx:
            return candidate
    return None

def normalize_columns(
    df: pd.DataFrame,
    hla_input: str = "pseudo",
    hla_pseudo_csv: str = "",
    hla_fasta: str = "",
    drop_missing_hla_sequence: bool = True,
    hla_esm_cache: Optional[dict] = None,
) -> pd.DataFrame:
    df = df.copy().rename(
        columns={
            "Peptide": "peptide",
            "antigen": "peptide",
            "pep": "peptide",
            "HLA": "HLA",
            "hla": "HLA",
            "mhc": "HLA",
            "Label": "label",
        }
    )
    if "HLA" not in df.columns and "MHC_Restriction_Name" in df.columns:
        df["HLA"] = df["MHC_Restriction_Name"]
    if not {"peptide", "HLA", "label"}.issubset(df.columns):
        raise ValueError(f"Missing columns: {set(df.columns)}")
        
    df["_source_row_idx"] = np.arange(len(df), dtype=np.int64)
    df["peptide"] = df["peptide"].astype(str).str.strip()
    df["HLA"] = df["HLA"].astype(str).str.strip()
    hla_name_col = "MHC_Restriction_Name" if "MHC_Restriction_Name" in df.columns else "HLA"
    df["_hla_name"] = df[hla_name_col].astype(str).str.strip()
    df["HLA_name"] = df["_hla_name"]
    df["label"] = df["label"].astype(int)
    df = df.loc[
        (df["peptide"].str.len() > 0)
        & (df["HLA"].str.len() > 0)
        & (df["_hla_name"].str.len() > 0)
    ].copy()

    if hla_input == "esm_cache":
        if hla_esm_cache is None:
            raise ValueError("hla_esm_cache is required when hla_input='esm_cache'")
        mapped = df["_hla_name"].map(lambda x: map_hla_name_to_esm_cache(x, hla_esm_cache["mapping"]))
        missing = mapped.isna()
        if missing.any():
            missing_counts = df.loc[missing, "_hla_name"].value_counts()
            msg = (
                f"Dropping {int(missing.sum())} rows with no HLA ESM cache embedding: "
                + ", ".join(f"{k}={v}" for k, v in missing_counts.head(30).items())
            )
            print(msg)
            if not drop_missing_hla_sequence:
                raise ValueError(msg)
            df = df.loc[~missing].copy()
            mapped = mapped.loc[~missing]
        df["HLA"] = mapped.values
    elif hla_pseudo_csv:
        hla_to_seq = load_hla_pseudo_sequences(hla_pseudo_csv)
        normalized = df["_hla_name"].map(standardize_hla_allele_name)
        mapped = normalized.map(hla_to_seq)
        missing = mapped.isna()
        fallback = missing & df["HLA"].map(looks_like_hla_pseudo_sequence)
        if fallback.any():
            mapped.loc[fallback] = df.loc[fallback, "HLA"]
            missing = mapped.isna()
        if missing.any():
            missing_counts = df.loc[missing, "_hla_name"].value_counts()
            msg = (
                f"Dropping {int(missing.sum())} rows with no 34-aa pseudo sequence in {hla_pseudo_csv}: "
                + ", ".join(f"{k}={v}" for k, v in missing_counts.head(30).items())
            )
            print(msg)
            if not drop_missing_hla_sequence:
                raise ValueError(msg)
            df = df.loc[~missing].copy()
            mapped = mapped.loc[~missing]
        df["HLA"] = mapped.values
    elif hla_fasta:
        hla_to_seq = load_hla_fasta(hla_fasta)
        normalized = df["_hla_name"].map(normalize_independent_mhc_name)
        mapped = normalized.map(hla_to_seq)
        missing = mapped.isna()
        if missing.any():
            missing_counts = df.loc[missing, "_hla_name"].value_counts()
            msg = (
                f"Dropping {int(missing.sum())} rows with no sequence in {hla_fasta}: "
                + ", ".join(f"{k}={v}" for k, v in missing_counts.head(20).items())
            )
            print(msg)
            if not drop_missing_hla_sequence:
                raise ValueError(msg)
            df = df.loc[~missing].copy()
            mapped = mapped.loc[~missing]
        df["HLA"] = mapped.values

    return df[["_source_row_idx", "peptide", "HLA", "HLA_name", "label"]].copy()

def _extract_pep_embeddings(pep_list, emb_cfg, device, pep_max_len, batch_size=64):
    backend = emb_cfg["backend"]
    if backend == "AntigenLM":
        return antigenLM_extract(pep_list, model_name_or_path=emb_cfg["AntigenLM_path"], device=device, max_len=pep_max_len)
    elif backend == "esm2":
        raw_peptides = [p.rstrip("-") for p in pep_list]
        return extract_esm2_embeddings(
            raw_peptides,
            model_name=emb_cfg.get("esm2_model_name", "esm2_t33_650M_UR50D"),
            device=device,
            max_len=pep_max_len,
            batch_size=batch_size,
        )
    elif backend == "esmc":
        raw_peptides = [p.rstrip("-") for p in pep_list]
        client = load_esmc_client(emb_cfg, device)
        pep_embeddings = extract_esmc_embeddings(
            raw_peptides,
            client=client,
            device=device,
            batch_size=batch_size,
            max_len=pep_max_len,
            model_max_len=max(pep_max_len + 2, 32),
        )
        del client
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        return pep_embeddings
        
    raise ValueError(f"未知嵌入后端：{backend}")

def _esmc_weights_candidates(model_name: str):
    if model_name != "esmc_300m":
        return []
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return [
        os.path.join(repo_root, "LLM", "ESMC_300M", "esmc_300m_2024_12_v0.pth"),
    ]

def resolve_esmc_weights_path(emb_cfg) -> str:
    explicit = emb_cfg.get("esmc_weights_path", "")
    if explicit:
        return explicit
    for path in _esmc_weights_candidates(emb_cfg.get("esmc_model_name", "esmc_300m")):
        if os.path.exists(path):
            return path
    return ""

def load_esmc_client(emb_cfg, device):
    from esm.models.esmc import ESMC

    weights_path = resolve_esmc_weights_path(emb_cfg)
    if weights_path:
        if emb_cfg.get("esmc_model_name", "esmc_300m") != "esmc_300m":
            raise ValueError("--esmc_weights_path currently supports esmc_300m local weights only.")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"ESMC weights not found: {weights_path}")

        from esm.tokenization import get_esmc_model_tokenizers

        print(f"[ESMC] Load local weights: {weights_path}")
        model = ESMC(d_model=960, n_heads=15, n_layers=30, tokenizer=get_esmc_model_tokenizers()).eval()
        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        model = model.to(device)
        if str(device).startswith("cuda"):
            model = model.to(torch.bfloat16)
        print(f"[ESMC] Model ready on {device}")
        return model

    return ESMC.from_pretrained(emb_cfg.get("esmc_model_name", "esmc_300m")).to(device)

def embedding_cache_backend_names(backend: str):
    if backend == "esmc":
        return ["esmc_raw"]
    return [backend]


def _safe_cache_id(value: str) -> str:
    raw = os.path.basename(os.path.normpath(str(value))) or str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)


def peptide_cache_backend_names(emb_cfg):
    backend = emb_cfg["backend"]
    if backend == "AntigenLM":
        return [f"AntigenLM_{_safe_cache_id(emb_cfg['AntigenLM_path'])}"]
    return embedding_cache_backend_names(backend)


def _embedding_cache_prefix(emb_cache_dir: str, type_tag: str, seed: int, emb_cfg) -> str:
    cache_backend = peptide_cache_backend_names(emb_cfg)[0]
    return P(emb_cache_dir, f"cached_pep_embeddings_{type_tag}_{seed}_{cache_backend}")

def _embedding_cache_prefix_candidates(emb_cache_dir: str, type_tag: str, seed: int, emb_cfg):
    tags = [type_tag]
    if not type_tag.endswith(f"_{seed}") and not type_tag.endswith(f"_1_{seed}"):
        tags.append(f"{type_tag}_1")
    if type_tag.startswith("cv") and type_tag.endswith("_test"):
        fold_text = type_tag[2:-5]
        if fold_text.isdigit():
            tags.extend([f"{type_tag}_{fold_text}", f"test_{fold_text}"])

    candidates = []
    for tag in tags:
        for cache_backend in peptide_cache_backend_names(emb_cfg):
            candidates.append(P(emb_cache_dir, f"cached_pep_embeddings_{tag}_{seed}_{cache_backend}"))
            candidates.append(P(emb_cache_dir, f"cached_pep_embeddings_{tag}_{seed}_pseudo34_{cache_backend}"))

    seen = set()
    unique = []
    for prefix in candidates:
        if prefix not in seen:
            unique.append(prefix)
            seen.add(prefix)
    return unique

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

def _extract_antigenlm_to_memmap(pep_list, emb_cfg, device, pep_max_len, prefix, batch_size, cache_dtype):
    mmap_path = prefix + ".mmap"
    meta_path = prefix + ".npz"
    tmp_path = mmap_path + ".tmp"
    dtype = np.float16 if cache_dtype == "float16" else np.float32

    tokenizer = load_antigenlm_tokenizer(emb_cfg["AntigenLM_path"])
    model = load_antigenlm_model(emb_cfg["AntigenLM_path"], device)
    model.eval()

    out = None
    with torch.no_grad():
        for start in tqdm(range(0, len(pep_list), batch_size), desc=f"Extract {os.path.basename(emb_cfg['AntigenLM_path'].rstrip('/'))}"):
            batch_seqs = [seq.rstrip("-")[:pep_max_len] for seq in pep_list[start : start + batch_size]]
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

def get_peptide_embedding_cache(pep_list, type_tag, seed, device, emb_cache_dir, emb_cfg, pep_max_len, batch_size, cache_dtype, use_cache=True):
    os.makedirs(emb_cache_dir, exist_ok=True)
    prefix = _embedding_cache_prefix(emb_cache_dir, type_tag, seed, emb_cfg)

    if use_cache:
        memmap_cache = _load_memmap_cache(prefix)
        if memmap_cache is not None:
            print(f"[Cache] Use memmap peptide embeddings: {prefix}.mmap")
            return memmap_cache

        old_pt_path = prefix + ".pt"
        if os.path.isfile(old_pt_path):
            print(f"[Cache] Use peptide embeddings: {old_pt_path}")
            return torch.load(old_pt_path, map_location='cpu')

    if emb_cfg["backend"] != "AntigenLM":
        pep_embeddings = _extract_pep_embeddings(pep_list, emb_cfg, device, pep_max_len, batch_size)
        torch.save(pep_embeddings.detach().cpu(), prefix + ".pt")
        return pep_embeddings.detach().cpu()

    print(f"[Cache] Build memmap peptide embeddings: {prefix}.mmap")
    return _extract_antigenlm_to_memmap(pep_list, emb_cfg, device, pep_max_len, prefix, batch_size, cache_dtype)

def data_process_hla(
    data: pd.DataFrame,
    type_tag: str,
    seed: int,
    device: torch.device,
    emb_cache_dir: str,
    pep_max_len=15,
    hla_max_len_=34,
    emb_cfg=None,
    use_cache=True,
    hla_input: str = "pseudo",
    hla_esm_cache_path: str = "",
    hla_pseudo_csv: str = "",
    hla_fasta: str = "",
    drop_missing_hla_sequence: bool = True,
    embed_extract_batch_size: int = 64,
    emb_cache_dtype: str = "float16",
):
    hla_esm_cache = load_hla_esm_cache(hla_esm_cache_path) if hla_input == "esm_cache" else None
    data = normalize_columns(
        data,
        hla_input=hla_input,
        hla_pseudo_csv=hla_pseudo_csv,
        hla_fasta=hla_fasta,
        drop_missing_hla_sequence=drop_missing_hla_sequence,
        hla_esm_cache=hla_esm_cache,
    )
    data = data.loc[(data["peptide"].str.len() > 0) & (data["HLA"].str.len() > 0)].reset_index(drop=True)
    
    pep_list = [p.ljust(pep_max_len, '-') for p in data.peptide]
    pep_raw = data.peptide.tolist()
    prefix = _embedding_cache_prefix(emb_cache_dir, type_tag, seed, emb_cfg)
    base_cache = None
    cache_prefix = prefix
    if use_cache:
        for candidate_prefix in _embedding_cache_prefix_candidates(emb_cache_dir, type_tag, seed, emb_cfg):
            base_cache = _load_existing_embedding_cache(candidate_prefix)
            if base_cache is not None:
                cache_prefix = candidate_prefix
                break
    row_indices = data["_source_row_idx"].to_numpy(dtype=np.int64)
    if base_cache is not None and len(base_cache) == len(data):
        pep_embeddings = base_cache
        print(f"[Cache] Use peptide embeddings: {cache_prefix} rows={len(data)}")
    elif base_cache is not None and len(row_indices) > 0 and int(row_indices.max()) < len(base_cache):
        pep_embeddings = IndexedEmbeddingCache(base_cache, row_indices)
        print(f"[Cache] Use indexed peptide embeddings: {cache_prefix} rows={len(data)}/{len(base_cache)}")
    else:
        filtered_type_tag = f"{type_tag}_pseudo34"
        if base_cache is not None:
            print(
                f"[Cache] Existing cache shape mismatch for {cache_prefix}: "
                f"cache_rows={len(base_cache)} data_rows={len(data)} max_source_row={int(row_indices.max()) if len(row_indices) else -1}; "
                f"building {filtered_type_tag}"
            )
        elif use_cache:
            tried = ", ".join(_embedding_cache_prefix_candidates(emb_cache_dir, type_tag, seed, emb_cfg))
            print(f"[Cache] No peptide embedding cache found; building new cache. Tried prefixes: {tried}")
        pep_embeddings = get_peptide_embedding_cache(
            pep_list,
            filtered_type_tag,
            seed,
            "cuda" if torch.cuda.is_available() else "cpu",
            emb_cache_dir,
            emb_cfg,
            pep_max_len,
            embed_extract_batch_size,
            emb_cache_dtype,
            use_cache,
        )
        
    hla_names = data.HLA_name.tolist() if "HLA_name" in data.columns else data.HLA.tolist()
    if hla_input == "esm_cache":
        hla_indices = data["HLA"].map(hla_esm_cache["mapping"]).to_numpy(dtype=np.int64)
        hla_inputs = hla_esm_cache["embeddings"][hla_indices].numpy()
    else:
        hla_codes, unique_hla = pd.factorize(data.HLA, sort=False)
        unique_hla_tokens = np.asarray(
            [[vocab.get(n, vocab.get('-', 0)) for n in seq.ljust(hla_max_len_, '-')] for seq in unique_hla],
            dtype=np.uint8,
        )
        hla_inputs = unique_hla_tokens[hla_codes]
    labels = [int(y) if str(y).isdigit() else 0 for y in data.label]
    assert pep_embeddings.shape[0] == hla_inputs.shape[0], (
        f"peptide/cache rows {pep_embeddings.shape[0]} != HLA rows {hla_inputs.shape[0]}"
    )
    
    return pep_embeddings, hla_inputs, torch.LongTensor(labels), pep_raw, hla_names

class EvalDataSet_HLA(Dataset):
    def __init__(self, pep_embeds, hla_idx_tensor, labels_tensor, pep_raw, hla_raw):
        self.pep_embeds = pep_embeds
        self.hla_idx = hla_idx_tensor
        self.labels = labels_tensor
        self.pep_raw = pep_raw
        self.hla_raw = hla_raw
        
    def __len__(self):
        return self.labels.size(0)
        
    def __getitem__(self, i):
        pep = self.pep_embeds[i]
        if isinstance(pep, np.ndarray):
            pep = torch.from_numpy(pep.astype(np.float32, copy=True))
        hla = self.hla_idx[i]
        if isinstance(hla, np.ndarray):
            if np.issubdtype(hla.dtype, np.integer):
                hla = torch.from_numpy(hla.astype(np.int64, copy=True))
            else:
                hla = torch.from_numpy(hla.astype(np.float32, copy=True))
        return pep, hla, self.labels[i], self.pep_raw[i], self.hla_raw[i]

def build_loader_from_df(
    df_sub: pd.DataFrame,
    type_tag: str,
    batch_size: int,
    seed: int,
    device: torch.device,
    emb_cache_dir: str,
    emb_cfg,
    hla_input: str,
    hla_esm_cache_path: str,
    hla_pseudo_csv: str,
    hla_fasta: str,
    drop_missing_hla_sequence: bool,
    embed_extract_batch_size: int,
    emb_cache_dtype: str,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    pep_inputs, hla_inputs, labels, pep_raw, hla_raw = data_process_hla(
        df_sub,
        type_tag,
        seed,
        device,
        emb_cache_dir,
        15,
        hla_max_len,
        emb_cfg,
        hla_input=hla_input,
        hla_esm_cache_path=hla_esm_cache_path,
        hla_pseudo_csv=hla_pseudo_csv,
        hla_fasta=hla_fasta,
        drop_missing_hla_sequence=drop_missing_hla_sequence,
        embed_extract_batch_size=embed_extract_batch_size,
        emb_cache_dtype=emb_cache_dtype,
    )
    ds = EvalDataSet_HLA(pep_inputs, hla_inputs, labels, pep_raw, hla_raw)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return loader, ds

@torch.no_grad()
def eval_on_loader(model, loader, device, threshold):
    model.eval()
    y_true, y_prob, y_bin = [], [], []
    pep_list, hla_list, feat_chunks = [], [], []
    
    for anti_inputs, hla_inputs, labels, pep_strs, hla_strs in tqdm(loader, colour='blue'):
        logits, _, features = model(anti_inputs.to(device).float(), hla_inputs.to(device))
        probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
        
        y_prob.extend(probs.tolist())
        y_true.extend(labels.tolist())
        y_bin.extend((probs >= threshold).astype(np.int32).tolist())
        
        pep_list.extend(list(pep_strs))
        hla_list.extend(list(hla_strs))
        feat_chunks.append(features.detach().cpu().numpy())
        
    feats = np.concatenate(feat_chunks, axis=0) if feat_chunks else np.zeros((0, 1), np.float32)
    return y_true, y_prob, y_bin, pep_list, hla_list, feats

DEFAULTS = {
    "hla_input": "pseudo",
    "hla_esm_cache_path": "./data_cached/cached_hla_esm2_fullseq_mean.pt",
    "hla_pseudo_csv": "./data/dataset_all.csv",
    "hla_fasta": "",
    "drop_missing_hla_sequence": True,
    "esm2_model_name": "esm2_t33_650M_UR50D",
    "esmc_model_name": "esmc_300m",
    "esmc_weights_path": "../../LLM/ESMC_300M/esmc_300m_2024_12_v0.pth",
    "embed_extract_batch_size": 512,
    "emb_cache_dtype": "float16",
    "num_workers": 2,
}

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate pMHC-I checkpoints on independent split or test CV folds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.set_defaults(**DEFAULTS)
    ap.add_argument("--mode", choices=["independent", "cv", "all"], default="cv")
    ap.add_argument("--data_csv", default="./data/Independent data/el_test.csv")
    ap.add_argument("--cv_dir", default=DEFAULT_CV_DIR)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="CV folds to evaluate; empty means all discovered folds.")
    ap.add_argument("--weights_dir", default="../trained_model/pMHC-I/")
    ap.add_argument("--weights_path", default="", help="Explicit checkpoint path. Intended for single-split evaluation.")
    ap.add_argument("--out_dir", default="../result/pMHC-I/")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--threshold_opt_metric",
        choices=["mcc", "f1", "none"],
        default="mcc",
        help="Choose the decision threshold on each evaluated split. Use 'none' to keep --threshold fixed.",
    )
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--emb_cache_dir", default="./data_cached/")
    ap.add_argument("--hla_pseudo_csv", default=DEFAULTS["hla_pseudo_csv"])
    ap.add_argument("--embed_backend", choices=["AntigenLM", "esm2", "esmc"], default="AntigenLM")
    ap.add_argument("--AntigenLM_path", default="../../LLM/AntigenLM")
    ap.add_argument("--esmc_weights_path", default=DEFAULTS["esmc_weights_path"])
    ap.add_argument("--hla_input", choices=["pseudo", "esm_cache"], default=DEFAULTS["hla_input"])
    ap.add_argument("--hla_esm_cache_path", default=DEFAULTS["hla_esm_cache_path"])
    ap.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    ap.add_argument("--embed_extract_batch_size", type=int, default=DEFAULTS["embed_extract_batch_size"])
    ap.add_argument("--emb_cache_dtype", choices=["float16", "float32"], default=DEFAULTS["emb_cache_dtype"])
    return ap.parse_args()

def resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(SCRIPT_DIR, path))

def normalize_config_paths(args) -> None:
    for name in (
        "data_csv", "cv_dir", "weights_dir", "weights_path", "out_dir", "emb_cache_dir",
        "hla_pseudo_csv", "hla_fasta", "hla_esm_cache_path", "esmc_weights_path",
    ):
        setattr(args, name, resolve_path(getattr(args, name)))
    if args.AntigenLM_path and (
        args.AntigenLM_path.startswith((".", "/")) or os.path.exists(resolve_path(args.AntigenLM_path))
    ):
        args.AntigenLM_path = resolve_path(args.AntigenLM_path)

def _fold_id_from_path(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.rsplit("_", 1)[-1])

def discover_cv_tests(cv_dir: str, folds=None):
    if folds:
        fold_ids = sorted(set(folds))
    else:
        test_files = glob.glob(os.path.join(cv_dir, "test_fold_*.csv"))
        fold_ids = sorted(_fold_id_from_path(path) for path in test_files)
    if not fold_ids:
        raise FileNotFoundError(f"No test_fold_*.csv found in {cv_dir}")

    splits = []
    for fold in fold_ids:
        test_csv = os.path.join(cv_dir, f"test_fold_{fold}.csv")
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"Missing CV test file: {test_csv}")
        splits.append((f"cv_fold_{fold}", fold, test_csv, f"cv{fold}_test", f"cv_fold_{fold}_test"))
    return splits

def build_eval_plan(args):
    splits = []
    if args.mode in ("independent", "all"):
        if not os.path.exists(args.data_csv):
            raise FileNotFoundError(f"Missing independent test file: {args.data_csv}")
        splits.append(("independent", 1, args.data_csv, "independent_test", "independent_test"))
    if args.mode in ("cv", "all"):
        splits.extend(discover_cv_tests(args.cv_dir, args.folds))
    return splits

def checkpoint_candidates(args, split_name: str, fold: int):
    if args.weights_path:
        return [args.weights_path]

    suffix = args.embed_backend + ("_hlaesm" if args.hla_input == "esm_cache" else "")
    candidates = []
    if split_name == "independent":
        candidates.extend([
            P(args.weights_dir, f"independent_seed{args.seed}_{suffix}.pt"),
            P(args.weights_dir, f"fold1_seed{args.seed}_{suffix}.pt"),
            P(args.weights_dir, "Independent", f"fold1_seed{args.seed}_{suffix}.pt"),
        ])
    else:
        candidates.extend([
            P(args.weights_dir, f"{split_name}_seed{args.seed}_{suffix}.pt"),
            P(args.weights_dir, f"fold{fold}_seed{args.seed}_{suffix}.pt"),
        ])

    seen, unique = set(), []
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique

def load_checkpoint(model, ckpt: str, device: torch.device):
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)

def metric_row(split_name: str, m: dict):
    return {
        "fold": split_name,
        "threshold": m["threshold"],
        "auc": m["auc"],
        "accuracy": m["acc"],
        "mcc": m["mcc"],
        "f1": m["f1"],
        "pr_auc": m["aupr"],
        "Sensitivity": m["sensitivity"],
        "Specificity": m["specificity"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "tn": m["tn"],
        "fp": m["fp"],
        "fn": m["fn"],
        "tp": m["tp"],
        "pred_0": m["pred_0"],
        "pred_1": m["pred_1"],
        "true_0": m["true_0"],
        "true_1": m["true_1"],
    }

def main():
    args = parse_args()
    normalize_config_paths(args)

    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    emb_cfg = {
        "backend": args.embed_backend,
        "AntigenLM_path": args.AntigenLM_path,
        "esm2_model_name": args.esm2_model_name,
        "esmc_model_name": args.esmc_model_name,
        "esmc_weights_path": args.esmc_weights_path,
    }
    output_prefix = args.embed_backend + ("_hlaesm" if args.hla_input == "esm_cache" else "")
    pep_dim = peptide_embedding_dim(args.embed_backend, args.esm2_model_name, args.AntigenLM_path)
    model = (Mymodel_HLA_ESM(pep_dim=pep_dim) if args.hla_input == "esm_cache" else Mymodel_HLA(pep_dim=pep_dim)).to(device)

    metrics_rows = []
    for split_name, fold, data_csv, type_tag, result_tag in build_eval_plan(args):
        print(f"\n========== {split_name} | {data_csv} ==========")
        loader, _ = build_loader_from_df(
            pd.read_csv(data_csv),
            type_tag,
            args.batch_size,
            args.seed,
            device,
            args.emb_cache_dir,
            emb_cfg,
            args.hla_input,
            args.hla_esm_cache_path,
            args.hla_pseudo_csv,
            args.hla_fasta,
            args.drop_missing_hla_sequence,
            args.embed_extract_batch_size,
            args.emb_cache_dtype,
            args.num_workers,
            torch.cuda.is_available(),
        )

        ckpt = next((path for path in checkpoint_candidates(args, split_name, fold) if os.path.exists(path)), "")
        if not ckpt:
            tried = "\n  ".join(checkpoint_candidates(args, split_name, fold))
            print(f"[Skip] 未找到权重，已尝试：\n  {tried}")
            continue

        print(f"[Model] Load checkpoint: {ckpt}")
        load_checkpoint(model, ckpt, device)
        y_true, y_prob, y_bin, pep_strs, hla_strs, _ = eval_on_loader(model, loader, device, args.threshold)

        threshold = args.threshold
        if args.threshold_opt_metric != "none":
            threshold, y_bin, best_stats = find_best_threshold(y_true, y_prob, args.threshold_opt_metric)
            print(
                f"[Threshold] optimize={args.threshold_opt_metric} "
                f"threshold={threshold:.6f} mcc={best_stats['mcc']:.4f} f1={best_stats['f1']:.4f}"
            )
        else:
            y_bin = threshold_predictions(y_prob, threshold)
            print(f"[Threshold] fixed threshold={threshold:.6f}")

        df_pred = pd.DataFrame({
            "split": [split_name] * len(y_true),
            "HLA": hla_strs,
            "peptide": pep_strs,
            "label_true": y_true,
            "label_pred": y_bin,
            "label_prob": y_prob,
        })
        pred_path = P(args.out_dir, f"{output_prefix}_{result_tag}_pred_results.csv")
        df_pred.to_csv(pred_path, index=False)
        print(f"[Save] {pred_path}")

        print(f"\n===== {split_name} | EVAL =====")
        metrics_rows.append(metric_row(split_name, performance(y_true, y_prob, y_bin, threshold)))

    if len(metrics_rows) >= 2:
        print("\n===== Average | EVAL =====")
        avg_m = pd.DataFrame(metrics_rows).mean(numeric_only=True)
        print(f"tn={avg_m['tn']:.0f}, fp={avg_m['fp']:.0f}, fn={avg_m['fn']:.0f}, tp={avg_m['tp']:.0f}")
        print(f"y_pred: 0={avg_m['pred_0']:.0f} | 1={avg_m['pred_1']:.0f}")
        print(f"y_true: 0={avg_m['true_0']:.0f} | 1={avg_m['true_1']:.0f}")
        print(
            f"auc={avg_m['auc']:.4f}|sensitivity={avg_m['Sensitivity']:.4f}|"
            f"specificity={avg_m['Specificity']:.4f}|acc={avg_m['accuracy']:.4f}|mcc={avg_m['mcc']:.4f}"
        )
        print(
            f"precision={avg_m['Precision']:.4f}|recall={avg_m['Recall']:.4f}|"
            f"f1={avg_m['f1']:.4f}|aupr={avg_m['pr_auc']:.4f}"
        )

    if metrics_rows:
        metric_cols = ["fold", "threshold", "auc", "accuracy", "mcc", "f1", "pr_auc", "Sensitivity", "Specificity", "Precision", "Recall"]
        df_metrics = pd.DataFrame(metrics_rows, columns=metric_cols)
        avg_row, sd_row = {"fold": "avg"}, {"fold": "SD"}
        for col in metric_cols[1:]:
            avg_row[col] = df_metrics[col].mean()
            sd_row[col] = df_metrics[col].std(ddof=1) if len(df_metrics) >= 2 else 0.0
        df_metrics = pd.concat([df_metrics, pd.DataFrame([avg_row, sd_row])], ignore_index=True)

        metrics_tag = "independent_test" if args.mode == "independent" else args.mode
        out_csv_path = P(args.out_dir, f"{output_prefix}_{metrics_tag}_metrics.csv")
        df_metrics.to_csv(out_csv_path, index=False)
        print(f"\n[Save] {out_csv_path}")

    print("\n===== 评测完成 =====")

if __name__ == "__main__":
    main()
