import os
import random

import numpy as np
import torch


P = os.path.join
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CV_DATA_DIR = P(SCRIPT_DIR, "data", "30_similarity")
DEFAULT_PLDGL_DATA_DIR = P(SCRIPT_DIR, "data", "Independent_data")
PLDGL_SUBSETS = ("Bacteria", "Viruses")


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sanitize_name(name):
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name.strip())
    return safe.strip("_") or "dataset"


def parse_subset_list(text, run_all=False):
    if run_all or (text or "").strip().lower() == "all":
        return list(PLDGL_SUBSETS)

    subsets = [token.strip() for token in (text or "").split(",") if token.strip()]
    unknown = [item for item in subsets if item not in PLDGL_SUBSETS]
    if unknown:
        raise ValueError(
            f"Unknown PLDGL subset: {unknown[0]}. Available: {', '.join(PLDGL_SUBSETS)}"
        )
    return subsets or ["All"]


def parse_fold_filter(folds_text):
    if not folds_text:
        return None
    folds = set()
    for token in folds_text.split(","):
        token = token.strip()
        if not token:
            continue
        folds.add(int(token.split("_")[-1] if token.startswith("fold_") else token))
    return folds


def _fold_index(path):
    try:
        return int(os.path.basename(path.rstrip(os.sep)).split("_")[-1])
    except ValueError:
        return 10**9


def _split_file_index(filename, prefix):
    if not filename.startswith(prefix) or not filename.endswith(".csv"):
        return None
    try:
        return int(filename[len(prefix):-4])
    except ValueError:
        return None


def has_fold_dirs(path):
    return os.path.isdir(path) and any(
        entry.startswith("fold_") and os.path.isdir(P(path, entry))
        for entry in os.listdir(path)
    )


def has_split_file_folds(path, split_name=None):
    if not os.path.isdir(path):
        return False
    prefixes = [f"{split_name}_fold_"] if split_name else [
        "train_fold_",
        "val_fold_",
        "test_fold_",
    ]
    return any(
        any(_split_file_index(entry, prefix) is not None for prefix in prefixes)
        for entry in os.listdir(path)
    )


def has_cv_layout(path, split_name=None):
    return has_fold_dirs(path) or has_split_file_folds(path, split_name)


def resolve_cv_dir(data_dir, dataset_name=None, split_name=None):
    data_dir = os.path.abspath(data_dir)
    if dataset_name:
        cv_dir = P(data_dir, dataset_name)
        if not os.path.isdir(cv_dir):
            raise FileNotFoundError(f"Dataset directory not found: {cv_dir}")
        if not has_cv_layout(cv_dir, split_name):
            raise FileNotFoundError(f"No CV layout found in: {cv_dir}")
        return cv_dir, dataset_name

    if has_cv_layout(data_dir, split_name):
        return data_dir, os.path.basename(data_dir.rstrip(os.sep))

    matched_datasets = [
        entry for entry in sorted(os.listdir(data_dir))
        if os.path.isdir(P(data_dir, entry)) and has_cv_layout(P(data_dir, entry), split_name)
    ]
    if not matched_datasets:
        raise FileNotFoundError(f"No CV dataset found in: {data_dir}")
    if len(matched_datasets) > 1:
        raise ValueError(
            "Multiple CV datasets found. Please choose one with --dataset_name. "
            f"Available: {', '.join(repr(item) for item in matched_datasets)}"
        )
    return P(data_dir, matched_datasets[0]), matched_datasets[0]


def resolve_cv_datasets(data_dir, dataset_name=None, split_name=None):
    data_dir = os.path.abspath(data_dir)
    if dataset_name or has_cv_layout(data_dir, split_name):
        return [resolve_cv_dir(data_dir, dataset_name, split_name)]

    datasets = [
        (P(data_dir, entry), entry)
        for entry in sorted(os.listdir(data_dir))
        if os.path.isdir(P(data_dir, entry)) and has_cv_layout(P(data_dir, entry), split_name)
    ]
    if not datasets:
        raise FileNotFoundError(f"No CV dataset found in: {data_dir}")
    return datasets


def discover_folds(cv_dir, folds_text=None):
    selected = parse_fold_filter(folds_text)
    fold_dirs = []
    for entry in sorted(os.listdir(cv_dir), key=_fold_index):
        fold_dir = P(cv_dir, entry)
        if not entry.startswith("fold_") or not os.path.isdir(fold_dir):
            continue
        fold_idx = _fold_index(entry)
        if selected is None or fold_idx in selected:
            fold_dirs.append((fold_idx, fold_dir))
    if not fold_dirs:
        raise FileNotFoundError(f"No selected fold_* directories found in: {cv_dir}")
    return fold_dirs


def discover_split_indices(cv_dir, split_name, folds_text=None):
    selected = parse_fold_filter(folds_text)
    fold_indices = []
    for entry in os.listdir(cv_dir):
        fold_idx = _split_file_index(entry, f"{split_name}_fold_")
        if fold_idx is not None and (selected is None or fold_idx in selected):
            fold_indices.append(fold_idx)
    if not fold_indices:
        raise FileNotFoundError(f"No selected {split_name}_fold_*.csv files found in: {cv_dir}")
    return sorted(fold_indices)


def discover_cv_train_files(cv_dir, folds_text=None):
    if has_split_file_folds(cv_dir, "train"):
        fold_files = []
        for fold_idx in discover_split_indices(cv_dir, "train", folds_text):
            train_csv = P(cv_dir, f"train_fold_{fold_idx}.csv")
            val_csv = P(cv_dir, f"val_fold_{fold_idx}.csv")
            test_csv = P(cv_dir, f"test_fold_{fold_idx}.csv")
            if not os.path.exists(val_csv):
                raise FileNotFoundError(f"val_fold_{fold_idx}.csv not found: {val_csv}")
            fold_files.append((fold_idx, train_csv, val_csv, test_csv))
        return fold_files

    return [
        (fold_idx, P(fold_dir, "train.csv"), P(fold_dir, "val.csv"), P(fold_dir, "test.csv"))
        for fold_idx, fold_dir in discover_folds(cv_dir, folds_text)
    ]


def discover_cv_split_files(cv_dir, split_name="test", folds_text=None):
    if has_split_file_folds(cv_dir, split_name):
        return [
            (fold_idx, P(cv_dir, f"{split_name}_fold_{fold_idx}.csv"))
            for fold_idx in discover_split_indices(cv_dir, split_name, folds_text)
        ]
    return [
        (fold_idx, P(fold_dir, f"{split_name}.csv"))
        for fold_idx, fold_dir in discover_folds(cv_dir, folds_text)
    ]


def read_labeled_table(path, with_id=False):
    import pandas as pd

    df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    rename = {}
    if "Sequence" in df.columns and "sequence" not in df.columns:
        rename["Sequence"] = "sequence"
    if "Label" in df.columns and "label" not in df.columns:
        rename["Label"] = "label"
    if "id" in df.columns and "ID" not in df.columns:
        rename["id"] = "ID"
    df = df.rename(columns=rename)

    required = {"sequence", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    keep = ["sequence", "label"]
    if with_id:
        if "ID" not in df.columns:
            df["ID"] = [f"seq{i + 1}" for i in range(len(df))]
        keep.insert(0, "ID")

    df = df[keep].dropna(subset=["sequence", "label"]).reset_index(drop=True)
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["sequence"].str.len() > 0].reset_index(drop=True)
    if with_id:
        df["ID"] = df["ID"].astype(str)
    return df


def print_split_counts(name, df):
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    print(f"{name}: Pos={pos} Neg={neg} Total={len(df)} Ratio={neg / max(pos, 1):.2f}")


def prediction_labels(y_prob, threshold):
    return (np.asarray(y_prob) >= threshold).astype(int)


def binary_metrics(y_true, y_prob, threshold):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = prediction_labels(y_prob, threshold)
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    for name, scorer in [("auc", roc_auc_score), ("aupr", average_precision_score)]:
        try:
            metrics[name] = scorer(y_true, y_prob)
        except Exception:
            metrics[name] = float("nan")
    return metrics


def best_threshold_metrics(y_true, y_prob, metric_name):
    best_score, best_threshold, best_metrics = -float("inf"), 0.5, None
    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = binary_metrics(y_true, y_prob, threshold)
        score = metrics[metric_name]
        if not np.isnan(score) and score > best_score:
            best_score, best_threshold, best_metrics = score, float(threshold), metrics
    return best_threshold, best_metrics or binary_metrics(y_true, y_prob, 0.5)


class AntigenDataset:
    def __init__(self, sequences, labels, ids=None):
        self.sequences = list(sequences)
        self.labels = list(labels)
        self.ids = list(range(len(self.sequences))) if ids is None else list(ids)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.ids[idx], self.sequences[idx], self.labels[idx]

    def get_data(self):
        return self.ids, self.sequences, self.labels
