#!/usr/bin/env python3
"""Post hoc peptide nearest-neighbor similarity audit for pTCR2 splits."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "Downstream/pTCR2/data"
RESULT_ROOT = ROOT / "Result/pTCR2"
OUT_DIR = RESULT_ROOT / "SimilarityAudit"

FOLDS = range(1, 6)
MAIN_METHODS = ["DLpTCR", "ERGO-II", "PanPep", "T-SCAPE", "UniPMT", "UnifyImmun", "AntigenLM"]
LLM_METHODS = ["AntigenLM", "ESM2", "ESMC"]
DISTANCE_BUCKETS = ("0", "1", "2", ">=3")
LENGTH_STRATA = ("all", "len_8", "len_9", "len_10", "len_11", "len_other")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalize_peptide(value: object) -> str:
    return str(value).strip().upper()


def first_present(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        if name in row:
            return name
    raise KeyError(f"None of {list(names)} found in columns: {list(row)}")


def peptide_column(row: dict[str, str]) -> str:
    return first_present(row, ("antigen", "Epitope", "peptide", "Peptide"))


def unique_peptides(path: Path) -> list[str]:
    rows = read_csv(path)
    if not rows:
        return []
    col = peptide_column(rows[0])
    seen: set[str] = set()
    peptides: list[str] = []
    for row in rows:
        peptide = normalize_peptide(row[col])
        if peptide and peptide not in seen:
            seen.add(peptide)
            peptides.append(peptide)
    return peptides


def hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def edit_distance(left: str, right: str) -> int:
    if len(left) == len(right):
        return hamming_distance(left, right)
    return levenshtein_distance(left, right)


def identity_from_distance(left: str, right: str, distance: int) -> float:
    denom = max(len(left), len(right))
    if denom == 0:
        return 1.0
    return max(0.0, (denom - distance) / denom)


def distance_bucket(distance: int) -> str:
    if distance <= 2:
        return str(distance)
    return ">=3"


def nearest_neighbor(test_peptide: str, train_peptides: list[str], train_set: set[str]) -> dict[str, object]:
    if test_peptide in train_set:
        return {
            "min_edit_distance": 0,
            "distance_bucket": "0",
            "nearest_train_peptide": test_peptide,
            "nearest_train_length": len(test_peptide),
            "nearest_identity": 1.0,
            "max_identity": 1.0,
            "max_identity_train_peptide": test_peptide,
            "max_identity_distance": 0,
            "close_neighbor": 1,
        }

    best_distance: int | None = None
    best_identity = -1.0
    best_peptide = ""
    max_identity = -1.0
    max_identity_peptide = ""
    max_identity_distance: int | None = None

    for train_peptide in train_peptides:
        distance = edit_distance(test_peptide, train_peptide)
        identity = identity_from_distance(test_peptide, train_peptide, distance)

        if (
            best_distance is None
            or distance < best_distance
            or (distance == best_distance and identity > best_identity)
            or (distance == best_distance and identity == best_identity and train_peptide < best_peptide)
        ):
            best_distance = distance
            best_identity = identity
            best_peptide = train_peptide

        if (
            identity > max_identity
            or (identity == max_identity and (max_identity_distance is None or distance < max_identity_distance))
            or (
                identity == max_identity
                and distance == max_identity_distance
                and train_peptide < max_identity_peptide
            )
        ):
            max_identity = identity
            max_identity_peptide = train_peptide
            max_identity_distance = distance

        if distance == 0:
            best_distance = 0
            best_identity = 1.0
            best_peptide = train_peptide
            max_identity = 1.0
            max_identity_peptide = train_peptide
            max_identity_distance = 0
            break

    if best_distance is None or max_identity_distance is None:
        raise ValueError("Cannot compute nearest neighbor without training peptides.")

    return {
        "min_edit_distance": best_distance,
        "distance_bucket": distance_bucket(best_distance),
        "nearest_train_peptide": best_peptide,
        "nearest_train_length": len(best_peptide),
        "nearest_identity": best_identity,
        "max_identity": max_identity,
        "max_identity_train_peptide": max_identity_peptide,
        "max_identity_distance": max_identity_distance,
        "close_neighbor": int(best_distance <= 2),
    }


def audit_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = [
        {
            "setting": "Seen_vs_Unseen",
            "fold": "independent",
            "train_path": DATA_ROOT / "Seen.csv",
            "test_path": DATA_ROOT / "Unseen.csv",
        }
    ]
    for setting, base in (
        ("Seen_5fold", DATA_ROOT / "Seen_5fold_splits"),
        ("CMA_5fold", DATA_ROOT / "CMA_5fold_splits"),
    ):
        for fold in FOLDS:
            configs.append(
                {
                    "setting": setting,
                    "fold": fold,
                    "train_path": base / f"fold_{fold}/train.csv",
                    "test_path": base / f"fold_{fold}/test.csv",
                }
            )
    return configs


def audit_split(config: dict[str, object]) -> list[dict[str, object]]:
    train_path = Path(config["train_path"])
    test_path = Path(config["test_path"])
    train_peptides = unique_peptides(train_path)
    test_peptides = unique_peptides(test_path)
    train_set = set(train_peptides)

    rows: list[dict[str, object]] = []
    for peptide in test_peptides:
        nn = nearest_neighbor(peptide, train_peptides, train_set)
        rows.append(
            {
                "setting": config["setting"],
                "fold": config["fold"],
                "train_path": train_path.as_posix(),
                "test_path": test_path.as_posix(),
                "train_unique_peptides": len(train_peptides),
                "test_unique_peptides": len(test_peptides),
                "train_test_exact_overlap": len(train_set.intersection(test_peptides)),
                "test_peptide": peptide,
                "test_length": len(peptide),
                **nn,
            }
        )
    return rows


def stratum_labels(length: int) -> list[str]:
    labels = ["all"]
    if length in (8, 9, 10, 11):
        labels.append(f"len_{length}")
    else:
        labels.append("len_other")
    return labels


def mean_or_nan(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.mean(finite)) if finite else float("nan")


def median_or_nan(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.median(finite)) if finite else float("nan")


def summarize_audit(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        for label in stratum_labels(int(record["test_length"])):
            grouped[(record["setting"], record["fold"], label)].append(record)

    summary: list[dict[str, object]] = []
    for (setting, fold, stratum), rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        n = len(rows)
        bucket_counts = {
            bucket: sum(1 for row in rows if row["distance_bucket"] == bucket)
            for bucket in DISTANCE_BUCKETS
        }
        close_count = bucket_counts["0"] + bucket_counts["1"] + bucket_counts["2"]
        identities = [float(row["max_identity"]) for row in rows]
        summary.append(
            {
                "setting": setting,
                "fold": fold,
                "stratum": stratum,
                "n_test_peptides": n,
                "train_unique_peptides": rows[0]["train_unique_peptides"],
                "test_unique_peptides": rows[0]["test_unique_peptides"],
                "train_test_exact_overlap": rows[0]["train_test_exact_overlap"],
                "dist_0_count": bucket_counts["0"],
                "dist_1_count": bucket_counts["1"],
                "dist_2_count": bucket_counts["2"],
                "dist_ge3_count": bucket_counts[">=3"],
                "dist_0_prop": bucket_counts["0"] / n if n else float("nan"),
                "dist_1_prop": bucket_counts["1"] / n if n else float("nan"),
                "dist_2_prop": bucket_counts["2"] / n if n else float("nan"),
                "dist_ge3_prop": bucket_counts[">=3"] / n if n else float("nan"),
                "close_neighbor_count": close_count,
                "close_neighbor_prop": close_count / n if n else float("nan"),
                "max_identity_mean": mean_or_nan(identities),
                "max_identity_median": median_or_nan(identities),
                "max_identity_min": min(identities) if identities else float("nan"),
                "max_identity_max": max(identities) if identities else float("nan"),
            }
        )
    return summary


def summarize_exact_lengths(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(record["setting"], record["fold"], int(record["test_length"]))].append(record)

    summary: list[dict[str, object]] = []
    for (setting, fold, length), rows in sorted(grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]), item[0][2])):
        n = len(rows)
        bucket_counts = {
            bucket: sum(1 for row in rows if row["distance_bucket"] == bucket)
            for bucket in DISTANCE_BUCKETS
        }
        close_count = bucket_counts["0"] + bucket_counts["1"] + bucket_counts["2"]
        identities = [float(row["max_identity"]) for row in rows]
        summary.append(
            {
                "setting": setting,
                "fold": fold,
                "test_length": length,
                "n_test_peptides": n,
                "dist_0_count": bucket_counts["0"],
                "dist_1_count": bucket_counts["1"],
                "dist_2_count": bucket_counts["2"],
                "dist_ge3_count": bucket_counts[">=3"],
                "dist_0_prop": bucket_counts["0"] / n if n else float("nan"),
                "dist_1_prop": bucket_counts["1"] / n if n else float("nan"),
                "dist_2_prop": bucket_counts["2"] / n if n else float("nan"),
                "dist_ge3_prop": bucket_counts[">=3"] / n if n else float("nan"),
                "close_neighbor_count": close_count,
                "close_neighbor_prop": close_count / n if n else float("nan"),
                "max_identity_mean": mean_or_nan(identities),
                "max_identity_median": median_or_nan(identities),
                "max_identity_min": min(identities) if identities else float("nan"),
                "max_identity_max": max(identities) if identities else float("nan"),
            }
        )
    return summary


def annotate_unseen_rows(unseen_records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = read_csv(DATA_ROOT / "Unseen.csv")
    if not rows:
        return []
    lookup = {str(record["test_peptide"]): record for record in unseen_records}
    peptide_col = peptide_column(rows[0])
    annotated: list[dict[str, object]] = []
    for row_index, row in enumerate(rows, start=1):
        peptide = normalize_peptide(row[peptide_col])
        audit = lookup[peptide]
        annotated.append(
            {
                "row_index": row_index,
                **row,
                "normalized_antigen": peptide,
                "min_edit_distance": audit["min_edit_distance"],
                "distance_bucket": audit["distance_bucket"],
                "close_neighbor_subset": "close_neighbor" if int(audit["close_neighbor"]) else "distant",
                "nearest_train_peptide": audit["nearest_train_peptide"],
                "nearest_train_length": audit["nearest_train_length"],
                "max_identity": audit["max_identity"],
            }
        )
    return annotated


def prediction_specs() -> list[dict[str, object]]:
    return [
        {
            "result_group": "main",
            "base_dir": RESULT_ROOT / "Unseen",
            "methods": MAIN_METHODS,
        },
        {
            "result_group": "llm",
            "base_dir": RESULT_ROOT / "LLM/Unseen",
            "methods": LLM_METHODS,
        },
    ]


def prediction_path(spec: dict[str, object], method: str, fold: int) -> Path:
    return Path(spec["base_dir"]) / f"fold{fold}" / f"{method}_unseen_set_pred_result_fold{fold}.csv"


def parse_prediction_rows(path: Path) -> list[dict[str, object]]:
    rows = read_csv(path)
    if not rows:
        return []
    first = rows[0]
    if "label_true" in first:
        peptide_col_name = "antigen"
        true_col = "label_true"
        pred_col = "label_pred"
        prob_col = "label_prob"
    else:
        peptide_col_name = "Epitope"
        true_col = "y_true"
        pred_col = "y_pred"
        prob_col = "y_prob"

    parsed: list[dict[str, object]] = []
    for row in rows:
        score = float(row[prob_col])
        parsed.append(
            {
                "peptide": normalize_peptide(row[peptide_col_name]),
                "y_true": int(float(row[true_col])),
                "y_pred": int(float(row[pred_col])) if pred_col in row and row[pred_col] != "" else int(score >= 0.5),
                "y_score": score,
            }
        )
    return parsed


def safe_float(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


def compute_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    y_true = np.array([int(row["y_true"]) for row in rows], dtype=int)
    y_pred = np.array([int(row["y_pred"]) for row in rows], dtype=int)
    y_score = np.array([float(row["y_score"]) for row in rows], dtype=float)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)

    if len(rows) == 0:
        return {
            "n_rows": 0,
            "positives": 0,
            "negatives": 0,
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "f1": float("nan"),
            "mcc": float("nan"),
        }

    roc_auc = roc_auc_score(y_true, y_score) if len(set(y_true.tolist())) == 2 else float("nan")
    pr_auc = average_precision_score(y_true, y_score) if positives > 0 else float("nan")
    return {
        "n_rows": len(rows),
        "positives": positives,
        "negatives": negatives,
        "roc_auc": safe_float(roc_auc),
        "pr_auc": safe_float(pr_auc),
        "f1": safe_float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": safe_float(matthews_corrcoef(y_true, y_pred)),
    }


def recompute_unseen_subset_metrics(unseen_records: list[dict[str, object]]) -> list[dict[str, object]]:
    peptide_subset = {
        str(record["test_peptide"]): ("close_neighbor" if int(record["close_neighbor"]) else "distant")
        for record in unseen_records
    }
    metric_rows: list[dict[str, object]] = []

    for spec in prediction_specs():
        for method in spec["methods"]:
            for fold in FOLDS:
                path = prediction_path(spec, str(method), fold)
                if not path.exists():
                    continue
                predictions = parse_prediction_rows(path)
                for subset in (
                    "all",
                    "close_neighbor",
                    "distant",
                    "length_le_20",
                    "length_gt_20",
                    "length_le_30",
                    "length_gt_30",
                ):
                    if subset == "all":
                        subset_rows = predictions
                    elif subset in {"close_neighbor", "distant"}:
                        subset_rows = [
                            row for row in predictions
                            if peptide_subset.get(str(row["peptide"])) == subset
                        ]
                    elif subset == "length_le_30":
                        subset_rows = [
                            row for row in predictions
                            if len(str(row["peptide"])) <= 30
                        ]
                    elif subset == "length_gt_30":
                        subset_rows = [
                            row for row in predictions
                            if len(str(row["peptide"])) > 30
                        ]
                    elif subset == "length_le_20":
                        subset_rows = [
                            row for row in predictions
                            if len(str(row["peptide"])) <= 20
                        ]
                    else:
                        subset_rows = [
                            row for row in predictions
                            if len(str(row["peptide"])) > 20
                        ]
                    metric_rows.append(
                        {
                            "result_group": spec["result_group"],
                            "method": method,
                            "fold": fold,
                            "subset": subset,
                            "prediction_path": path.as_posix(),
                            **compute_metrics(subset_rows),
                        }
                    )
    return metric_rows


def summarize_metric_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object], list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(row["result_group"], row["method"], row["subset"])].append(row)

    summary: list[dict[str, object]] = []
    metrics = ("roc_auc", "pr_auc", "f1", "mcc")
    counts = ("n_rows", "positives", "negatives")
    for (result_group, method, subset), rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        out: dict[str, object] = {
            "result_group": result_group,
            "method": method,
            "subset": subset,
            "folds": len(rows),
        }
        for count in counts:
            values = [float(row[count]) for row in rows]
            out[f"{count}_mean"] = mean_or_nan(values)
        for metric in metrics:
            values = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
            out[f"{metric}_mean"] = mean_or_nan(values)
            out[f"{metric}_std"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0 if values else float("nan")
        summary.append(out)
    return summary


def fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def report_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join([header, sep, *body])


def write_report(
    audit_summary: list[dict[str, object]],
    exact_length_summary: list[dict[str, object]],
    metric_summary: list[dict[str, object]],
    output_path: Path,
) -> None:
    overall_rows = []
    for row in audit_summary:
        if row["stratum"] != "all":
            continue
        overall_rows.append(
            {
                "setting": row["setting"],
                "fold": row["fold"],
                "n": row["n_test_peptides"],
                "d0": f"{row['dist_0_count']} ({fmt(row['dist_0_prop'])})",
                "d1": f"{row['dist_1_count']} ({fmt(row['dist_1_prop'])})",
                "d2": f"{row['dist_2_count']} ({fmt(row['dist_2_prop'])})",
                "d>=3": f"{row['dist_ge3_count']} ({fmt(row['dist_ge3_prop'])})",
                "close": f"{row['close_neighbor_count']} ({fmt(row['close_neighbor_prop'])})",
                "mean_max_identity": fmt(row["max_identity_mean"]),
                "max_identity": fmt(row["max_identity_max"]),
            }
        )

    length_rows = []
    for row in audit_summary:
        if row["setting"] == "Seen_vs_Unseen" and row["stratum"] in {"len_8", "len_9", "len_10", "len_11"}:
            length_rows.append(
                {
                    "length": row["stratum"].replace("len_", ""),
                    "n": row["n_test_peptides"],
                    "d0": f"{row['dist_0_count']} ({fmt(row['dist_0_prop'])})",
                    "d1": f"{row['dist_1_count']} ({fmt(row['dist_1_prop'])})",
                    "d2": f"{row['dist_2_count']} ({fmt(row['dist_2_prop'])})",
                    "d>=3": f"{row['dist_ge3_count']} ({fmt(row['dist_ge3_prop'])})",
                    "mean_max_identity": fmt(row["max_identity_mean"]),
                    "max_identity": fmt(row["max_identity_max"]),
                }
            )
    length_rows.sort(key=lambda row: int(row["length"]))

    all_length_rows = []
    for row in exact_length_summary:
        if row["setting"] == "Seen_vs_Unseen" and row["fold"] == "independent":
            all_length_rows.append(
                {
                    "length": row["test_length"],
                    "n": row["n_test_peptides"],
                    "d0": f"{row['dist_0_count']} ({fmt(row['dist_0_prop'])})",
                    "d1": f"{row['dist_1_count']} ({fmt(row['dist_1_prop'])})",
                    "d2": f"{row['dist_2_count']} ({fmt(row['dist_2_prop'])})",
                    "d>=3": f"{row['dist_ge3_count']} ({fmt(row['dist_ge3_prop'])})",
                    "close": f"{row['close_neighbor_count']} ({fmt(row['close_neighbor_prop'])})",
                    "mean_max_identity": fmt(row["max_identity_mean"]),
                    "max_identity": fmt(row["max_identity_max"]),
                }
            )
    all_length_rows.sort(key=lambda row: int(row["length"]))

    metric_rows = []
    for row in metric_summary:
        if row["result_group"] != "main":
            continue
        metric_rows.append(
            {
                "method": row["method"],
                "subset": row["subset"],
                "n_mean": fmt(row["n_rows_mean"], 1),
                "roc_auc": f"{fmt(row['roc_auc_mean'])} +/- {fmt(row['roc_auc_std'])}",
                "pr_auc": f"{fmt(row['pr_auc_mean'])} +/- {fmt(row['pr_auc_std'])}",
                "f1": f"{fmt(row['f1_mean'])} +/- {fmt(row['f1_std'])}",
                "mcc": f"{fmt(row['mcc_mean'])} +/- {fmt(row['mcc_std'])}",
            }
        )

    text = [
        "# pTCR2 peptide nearest-neighbor similarity audit",
        "",
        "Close neighbor is defined as minimum edit distance <= 2. Equal-length peptide pairs use Hamming distance; unequal-length pairs use Levenshtein distance.",
        "",
        "## Overall peptide-level audit",
        "",
        report_table(overall_rows, ["setting", "fold", "n", "d0", "d1", "d2", "d>=3", "close", "mean_max_identity", "max_identity"]),
        "",
        "## Seen_vs_Unseen 8-11mer stratification",
        "",
        report_table(length_rows, ["length", "n", "d0", "d1", "d2", "d>=3", "mean_max_identity", "max_identity"]),
        "",
        "## Seen_vs_Unseen all-length stratification",
        "",
        report_table(all_length_rows, ["length", "n", "d0", "d1", "d2", "d>=3", "close", "mean_max_identity", "max_identity"]),
        "",
        "## Unseen subset prediction metrics, main baselines",
        "",
        "Values are fold mean +/- sample standard deviation across the existing five prediction files.",
        "",
        report_table(metric_rows, ["method", "subset", "n_mean", "roc_auc", "pr_auc", "f1", "mcc"]),
        "",
    ]
    output_path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_records: list[dict[str, object]] = []
    for config in audit_configs():
        audit_records.extend(audit_split(config))

    audit_fields = [
        "setting",
        "fold",
        "train_path",
        "test_path",
        "train_unique_peptides",
        "test_unique_peptides",
        "train_test_exact_overlap",
        "test_peptide",
        "test_length",
        "min_edit_distance",
        "distance_bucket",
        "close_neighbor",
        "nearest_train_peptide",
        "nearest_train_length",
        "nearest_identity",
        "max_identity",
        "max_identity_train_peptide",
        "max_identity_distance",
    ]
    write_csv(OUT_DIR / "peptide_nn_by_split.csv", audit_records, audit_fields)

    audit_summary = summarize_audit(audit_records)
    summary_fields = [
        "setting",
        "fold",
        "stratum",
        "n_test_peptides",
        "train_unique_peptides",
        "test_unique_peptides",
        "train_test_exact_overlap",
        "dist_0_count",
        "dist_1_count",
        "dist_2_count",
        "dist_ge3_count",
        "dist_0_prop",
        "dist_1_prop",
        "dist_2_prop",
        "dist_ge3_prop",
        "close_neighbor_count",
        "close_neighbor_prop",
        "max_identity_mean",
        "max_identity_median",
        "max_identity_min",
        "max_identity_max",
    ]
    write_csv(OUT_DIR / "peptide_nn_summary.csv", audit_summary, summary_fields)

    exact_length_summary = summarize_exact_lengths(audit_records)
    exact_length_fields = [
        "setting",
        "fold",
        "test_length",
        "n_test_peptides",
        "dist_0_count",
        "dist_1_count",
        "dist_2_count",
        "dist_ge3_count",
        "dist_0_prop",
        "dist_1_prop",
        "dist_2_prop",
        "dist_ge3_prop",
        "close_neighbor_count",
        "close_neighbor_prop",
        "max_identity_mean",
        "max_identity_median",
        "max_identity_min",
        "max_identity_max",
    ]
    write_csv(OUT_DIR / "peptide_nn_exact_length_summary.csv", exact_length_summary, exact_length_fields)

    unseen_records = [
        row for row in audit_records
        if row["setting"] == "Seen_vs_Unseen" and row["fold"] == "independent"
    ]
    annotated_unseen = annotate_unseen_rows(unseen_records)
    unseen_fields = [
        "row_index",
        "antigen",
        "TCR",
        "label",
        "negative_source",
        "normalized_antigen",
        "min_edit_distance",
        "distance_bucket",
        "close_neighbor_subset",
        "nearest_train_peptide",
        "nearest_train_length",
        "max_identity",
    ]
    write_csv(OUT_DIR / "unseen_rows_with_nn_distance.csv", annotated_unseen, unseen_fields)

    metric_rows = recompute_unseen_subset_metrics(unseen_records)
    metric_fields = [
        "result_group",
        "method",
        "fold",
        "subset",
        "prediction_path",
        "n_rows",
        "positives",
        "negatives",
        "roc_auc",
        "pr_auc",
        "f1",
        "mcc",
    ]
    write_csv(OUT_DIR / "unseen_close_neighbor_metrics_by_fold.csv", metric_rows, metric_fields)

    metric_summary = summarize_metric_rows(metric_rows)
    metric_summary_fields = [
        "result_group",
        "method",
        "subset",
        "folds",
        "n_rows_mean",
        "positives_mean",
        "negatives_mean",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "f1_mean",
        "f1_std",
        "mcc_mean",
        "mcc_std",
    ]
    write_csv(OUT_DIR / "unseen_close_neighbor_metrics_summary.csv", metric_summary, metric_summary_fields)
    write_report(audit_summary, exact_length_summary, metric_summary, OUT_DIR / "peptide_nn_similarity_audit_report.md")

    print(f"Wrote audit outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
