import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import compute_metrics
from model import AntigenLMESM2PMHCIIModel
from pmhc_data import PMHCIIEmbeddingDataset, batch_to_device, build_hla_store, distilled_cache_prefix_for_split


CURRENT_DIR = Path(__file__).resolve().parent
DOWNSTREAM_DIR = CURRENT_DIR.parent


def parse_folds(value):
    if value.lower() == "all":
        return [1, 2, 3, 4, 5]
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pMHC-II AntigenLM_distilled+ESM2 cross-attention checkpoints.")
    parser.add_argument("--root", type=Path, default=CURRENT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DOWNSTREAM_DIR / "trained_model" / "pMHC-II")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Evaluate one checkpoint instead of fold checkpoints.")
    parser.add_argument("--folds", type=str, default="all")
    parser.add_argument("--eval-set", choices=("warm", "cold", "both"), default="both")
    parser.add_argument("--results-dir", type=Path, default=DOWNSTREAM_DIR / "result" / "pMHC-II")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache-max-length", type=int, default=34)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-prefix", default="AntigenLM", help="Prefix used for pMHC-I-style result files.")
    return parser.parse_args()


def loader(dataset, batch_size, num_workers):
    options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


def load_model(checkpoint_path, hla_store, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = AntigenLMESM2PMHCIIModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.set_hla_embeddings(hla_store.embeddings)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict(model, dataset, args, device):
    data_loader = loader(dataset, args.batch_size, args.num_workers)
    all_scores = []
    all_labels = []
    all_core_start = []
    all_core_score = []
    all_pairs = []
    all_peptides = []
    all_alpha = []
    all_beta = []

    for batch in tqdm(data_loader, desc="predict", leave=False, dynamic_ncols=True):
        tensors = batch_to_device(batch, device)
        output = model(
            tensors["peptide_embedding"],
            tensors["peptide_mask"],
            tensors["alpha_index"],
            tensors["beta_index"],
            return_embedding=False,
            return_core=True,
        )
        if isinstance(output, tuple):
            logits = output[0]
            core_info = output[1] if len(output) > 1 else None
        else:
            logits = output
            core_info = None
        all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(tensors["label"].detach().cpu().numpy())
        if core_info is None:
            all_core_start.extend([""] * len(batch["peptide"]))
            all_core_score.extend([""] * len(batch["peptide"]))
        else:
            all_core_start.extend(core_info["core_start"].detach().cpu().numpy().astype(np.int64).tolist())
            all_core_score.extend(core_info["core_score"].detach().cpu().numpy().astype(np.float64).tolist())
        all_pairs.extend(batch["hla_pair"])
        all_peptides.extend(batch["peptide"])
        all_alpha.extend(batch["hla_alpha"])
        all_beta.extend(batch["hla_beta"])

    return {
        "scores": np.concatenate(all_scores),
        "labels": np.concatenate(all_labels).astype(np.int64),
        "hla_pair": all_pairs,
        "peptide": all_peptides,
        "hla_alpha": all_alpha,
        "hla_beta": all_beta,
        "core_start": all_core_start,
        "core_score": all_core_score,
    }


def prediction_split_name(eval_name, checkpoint_name):
    if checkpoint_name.startswith("fold"):
        return f"{eval_name}_fold_{checkpoint_name[4:]}"
    return f"{eval_name}_{checkpoint_name}"


def write_predictions(path, split_name, prediction, threshold):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "HLA",
                "peptide",
                "label_true",
                "label_pred",
                "label_prob",
                "hla_alpha",
                "hla_beta",
                "core_start_1based",
                "core_score",
            ]
        )
        core_starts = prediction.get("core_start", [""] * len(prediction["peptide"]))
        core_scores = prediction.get("core_score", [""] * len(prediction["peptide"]))
        for peptide, alpha, beta, pair, label, score, core_start, core_score in zip(
            prediction["peptide"],
            prediction["hla_alpha"],
            prediction["hla_beta"],
            prediction["hla_pair"],
            prediction["labels"],
            prediction["scores"],
            core_starts,
            core_scores,
        ):
            writer.writerow(
                [
                    split_name,
                    pair,
                    peptide,
                    int(label),
                    int(float(score) >= threshold),
                    f"{float(score):.8g}",
                    alpha,
                    beta,
                    core_start,
                    "" if core_score == "" else f"{float(core_score):.8g}",
                ]
            )


def metric_row(split_name, metrics):
    return {
        "fold": split_name,
        "auc": metrics["auc"],
        "accuracy": metrics["accuracy"],
        "mcc": metrics["mcc"],
        "f1": metrics["f1"],
        "pr_auc": metrics["aupr"],
        "Sensitivity": metrics["recall"],
        "Specificity": metrics["specificity"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
    }


def append_average_rows(rows):
    if not rows:
        return []

    metric_cols = ["auc", "accuracy", "mcc", "f1", "pr_auc", "Sensitivity", "Specificity", "Precision", "Recall"]
    output_rows = list(rows)
    fold_rows = [row for row in rows if "_fold_" in row["fold"]]
    if not fold_rows:
        return output_rows

    avg_row = {"fold": "avg"}
    sd_row = {"fold": "SD"}
    for col in metric_cols:
        values = np.asarray([float(row[col]) for row in fold_rows], dtype=np.float64)
        avg_row[col] = float(values.mean())
        sd_row[col] = float(values.std(ddof=1)) if len(values) >= 2 else 0.0
    output_rows.extend([avg_row, sd_row])
    return output_rows


def write_pmchi_style_metrics_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["fold", "auc", "accuracy", "mcc", "f1", "pr_auc", "Sensitivity", "Specificity", "Precision", "Recall"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(append_average_rows(rows))


def dataset_for_eval(root, eval_name, hla_store, cache_max_length):
    if eval_name == "warm":
        return PMHCIIEmbeddingDataset(
            root / "data" / "warm-start-test.csv",
            distilled_cache_prefix_for_split(root, "warm", max_length=cache_max_length),
            hla_store,
        )
    if eval_name == "cold":
        return PMHCIIEmbeddingDataset(
            root / "data" / "cold-start-test.csv",
            distilled_cache_prefix_for_split(root, "cold", max_length=cache_max_length),
            hla_store,
        )
    raise ValueError(f"Unsupported eval set: {eval_name}")


def checkpoint_paths(args):
    if args.checkpoint is not None:
        return [(args.checkpoint.stem, args.checkpoint)]
    return [(f"fold{fold}", args.checkpoint_dir / f"fold{fold}.pt") for fold in parse_folds(args.folds)]


def evaluate_set(eval_name, args, hla_store, device):
    dataset = dataset_for_eval(args.root, eval_name, hla_store, args.cache_max_length)
    rows = []
    fold_predictions = []
    for checkpoint_name, checkpoint_path in checkpoint_paths(args):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model, checkpoint = load_model(checkpoint_path, hla_store, device)
        prediction = predict(model, dataset, args, device)
        metrics = compute_metrics(prediction["labels"], prediction["scores"], threshold=args.threshold)
        split_name = prediction_split_name(eval_name, checkpoint_name)
        row = metric_row(split_name, metrics)
        rows.append(row)
        fold_predictions.append((checkpoint_name, prediction))
        pred_path = args.results_dir / f"{args.output_prefix}_{split_name}_test_pred_results.csv"
        write_predictions(pred_path, split_name, prediction, args.threshold)
        print(
            f"[{eval_name} {checkpoint_name}] auc={metrics['auc']:.4f} aupr={metrics['aupr']:.4f} "
            f"acc={metrics['accuracy']:.4f} mcc={metrics['mcc']:.4f} f1={metrics['f1']:.4f}"
        )

    if len(fold_predictions) > 1:
        base_prediction = fold_predictions[0][1]
        same_order = all(
            item[1]["hla_pair"] == base_prediction["hla_pair"]
            and item[1]["peptide"] == base_prediction["peptide"]
            and np.array_equal(item[1]["labels"], base_prediction["labels"])
            for item in fold_predictions[1:]
        )
        if same_order:
            ensemble = dict(base_prediction)
            ensemble["scores"] = np.mean([item[1]["scores"] for item in fold_predictions], axis=0)
            ensemble["core_start"] = [""] * len(ensemble["peptide"])
            ensemble["core_score"] = [""] * len(ensemble["peptide"])
            metrics = compute_metrics(ensemble["labels"], ensemble["scores"], threshold=args.threshold)
            split_name = prediction_split_name(eval_name, "ensemble_mean")
            rows.append(metric_row(split_name, metrics))
            pred_path = args.results_dir / f"{args.output_prefix}_{split_name}_test_pred_results.csv"
            write_predictions(pred_path, split_name, ensemble, args.threshold)
            print(
                f"[{eval_name} ensemble] auc={metrics['auc']:.4f} aupr={metrics['aupr']:.4f} "
                f"acc={metrics['accuracy']:.4f} mcc={metrics['mcc']:.4f} f1={metrics['f1']:.4f}"
            )
        else:
            print(f"[{eval_name}] skipped ensemble because prediction order differs.")
    return rows


def main():
    args = parse_args()
    args.root = args.root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    hla_store = build_hla_store(args.root)
    eval_sets = ["warm", "cold"] if args.eval_set == "both" else [args.eval_set]

    all_rows = []
    rows_by_eval = {}
    for eval_name in eval_sets:
        rows = evaluate_set(eval_name, args, hla_store, device)
        rows_by_eval[eval_name] = rows
        all_rows.extend(rows)
        out_csv = args.results_dir / f"{args.output_prefix}_{eval_name}_metrics.csv"
        write_pmchi_style_metrics_csv(out_csv, rows)
        print(f"Metrics saved to {out_csv}")

    if len(rows_by_eval) > 1:
        out_csv = args.results_dir / f"{args.output_prefix}_all_metrics.csv"
        write_pmchi_style_metrics_csv(out_csv, all_rows)
        print(f"Combined metrics saved to {out_csv}")


if __name__ == "__main__":
    main()
