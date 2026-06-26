import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_ROOT = Path(__file__).resolve().parent


def read_rows(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as handle:
        return [
            {key: value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
            if row
        ]


class PeptideEmbeddingCache:
    def __init__(self, prefix):
        prefix = Path(prefix)
        meta_path = prefix.with_suffix(".npz")
        mmap_path = prefix.with_suffix(".mmap")
        if not meta_path.exists() or not mmap_path.exists():
            raise FileNotFoundError(f"Missing peptide embedding cache for prefix: {prefix}")

        meta = np.load(meta_path)
        self.shape = tuple(int(x) for x in meta["shape"])
        self.dtype = str(meta["dtype"])
        self.row_to_unique = meta["row_to_unique"] if "row_to_unique" in meta.files else None
        self.attention_mask = meta["attention_mask"] if "attention_mask" in meta.files else None
        self.embeddings = np.memmap(mmap_path, mode="r", dtype=self.dtype, shape=self.shape)

    def __len__(self):
        return self.shape[0]

    def get(self, row_index):
        row_index = int(row_index)
        if self.row_to_unique is not None:
            row_index = int(self.row_to_unique[row_index])
        return np.asarray(self.embeddings[row_index], dtype=np.float32)

    def get_mask(self, row_index):
        row_index = int(row_index)
        if self.row_to_unique is not None:
            row_index = int(self.row_to_unique[row_index])
        if self.attention_mask is None:
            return np.ones((self.shape[1],), dtype=np.bool_)
        return np.asarray(self.attention_mask[row_index], dtype=np.bool_)


class HLAESM2Store:
    def __init__(self, sequence_csv, embedding_npy):
        self.alleles = []
        for row in read_rows(sequence_csv):
            self.alleles.append(row["allele"])
        self.allele_to_index = {allele: idx for idx, allele in enumerate(self.alleles)}
        self.embeddings = np.load(embedding_npy).astype(np.float32, copy=False)
        if self.embeddings.shape[0] != len(self.alleles):
            raise ValueError(
                f"HLA embedding count mismatch: {self.embeddings.shape[0]} embeddings for "
                f"{len(self.alleles)} alleles."
            )

    @property
    def embedding_dim(self):
        return int(self.embeddings.shape[-1])

    @property
    def sequence_length(self):
        return int(self.embeddings.shape[1])

    def get(self, allele):
        if allele not in self.allele_to_index:
            raise KeyError(f"HLA allele not found in ESM2 store: {allele}")
        return self.embeddings[self.allele_to_index[allele]]

    def get_index(self, allele):
        if allele not in self.allele_to_index:
            raise KeyError(f"HLA allele not found in ESM2 store: {allele}")
        return self.allele_to_index[allele]


class PMHCIIEmbeddingDataset(Dataset):
    def __init__(self, csv_path, peptide_cache_prefix, hla_store):
        self.csv_path = Path(csv_path)
        self.rows = read_rows(self.csv_path)
        self.peptide_cache = PeptideEmbeddingCache(peptide_cache_prefix) if peptide_cache_prefix is not None else None
        self.hla_store = hla_store
        self.labels = np.asarray([float(row["label"]) for row in self.rows], dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    @property
    def positive_count(self):
        return int(self.labels.sum())

    @property
    def negative_count(self):
        return int(len(self.labels) - self.positive_count)

    def __getitem__(self, index):
        row = self.rows[index]
        source_index = int(row.get("source_index", index))
        alpha_index = np.int64(self.hla_store.get_index(row["hla_alpha"]))
        beta_index = np.int64(self.hla_store.get_index(row["hla_beta"]))
        label = np.float32(row["label"])

        item = {
            "alpha_index": alpha_index,
            "beta_index": beta_index,
            "label": label,
            "peptide": row["peptide"],
            "hla_alpha": row["hla_alpha"],
            "hla_beta": row["hla_beta"],
            "hla_pair": f"{row['hla_alpha']}-{row['hla_beta']}",
        }
        if self.peptide_cache is not None:
            item["peptide_embedding"] = self.peptide_cache.get(source_index)
            item["peptide_mask"] = self.peptide_cache.get_mask(source_index)
        return item


def batch_to_device(batch, device):
    tensors = {
        "label": batch["label"].to(device=device, dtype=torch.float32, non_blocking=True),
    }
    if "alpha_index" in batch:
        tensors["alpha_index"] = batch["alpha_index"].to(device=device, dtype=torch.long, non_blocking=True)
    if "beta_index" in batch:
        tensors["beta_index"] = batch["beta_index"].to(device=device, dtype=torch.long, non_blocking=True)
    if "alpha_embedding" in batch:
        tensors["alpha_embedding"] = batch["alpha_embedding"].to(device=device, dtype=torch.float32, non_blocking=True)
    if "beta_embedding" in batch:
        tensors["beta_embedding"] = batch["beta_embedding"].to(device=device, dtype=torch.float32, non_blocking=True)
    if "peptide_embedding" in batch:
        tensors["peptide_embedding"] = batch["peptide_embedding"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
    if "peptide_mask" in batch:
        tensors["peptide_mask"] = batch["peptide_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    return tensors


def cache_prefix_for_split(root, split_name):
    root = Path(root)
    cache_dir = root / "data_cached"
    mapping = {
        "benchmark": cache_dir / "cached_pep_embeddings_benchmark_train_22_AntigenLM",
        "warm": cache_dir / "cached_pep_embeddings_warm_test_22_AntigenLM",
        "cold": cache_dir / "cached_pep_embeddings_cold_test_22_AntigenLM",
    }
    if split_name not in mapping:
        raise ValueError(f"Unknown split name: {split_name}")
    return mapping[split_name]


def distilled_cache_prefix_for_split(root, split_name, max_length=34):
    root = Path(root)
    cache_dir = root / "data_cached"
    mapping = {
        "benchmark": cache_dir / f"cached_pep_embeddings_benchmark_train_AntigenLM_distilled_len{max_length}",
        "warm": cache_dir / f"cached_pep_embeddings_warm_test_AntigenLM_distilled_len{max_length}",
        "cold": cache_dir / f"cached_pep_embeddings_cold_test_AntigenLM_distilled_len{max_length}",
    }
    if split_name not in mapping:
        raise ValueError(f"Unknown split name: {split_name}")
    return mapping[split_name]


def build_hla_store(root):
    root = Path(root)
    return HLAESM2Store(
        sequence_csv=root / "data" / "hla_dict" / "hla_full_seq_dict.csv",
        embedding_npy=root / "data" / "hla_dict" / "hla_esm_dict.npy",
    )
