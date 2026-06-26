import argparse
import csv
import math
import os
import random

import torch

from feature_extractors import (
    DEFAULT_ANTIGENLM_ENCODER_PATH,
    DEFAULT_ESM2_ENCODER_PATH,
    apply_encoder_state_dict_delta,
    checkpoint_stem,
    encode_residue_batch,
    load_feature_extractor,
    parse_residue_feature_groups,
    set_encoder_mode,
)
from model import (
    DEFAULT_RESIDUE_INPUT_DIM,
    classifier_checkpoint_config,
    classifier_state_dict,
    create_residue_classifier,
    infer_classifier_type,
)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

MODEL_NAME = "esm2_antigenlm"
CLASSIFIER_TYPE = "context_cnn"
CV_FOLDS = 5

DEFAULT_ESM2_PATH = DEFAULT_ESM2_ENCODER_PATH
DEFAULT_ANTIGENLM_PATH = os.path.join(PROJECT_ROOT, "LLM", "AntigenLM")
DEFAULT_TEST_FASTA = os.path.join(CURRENT_DIR, "data", "BP3C50ID_external_test_set.fasta")
DEFAULT_TRAINED_MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "Downstream",
    "trained_model",
    "B_cell_epitope",
)
DEFAULT_RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "Downstream",
    "result",
    "B_cell_epitope",
)

OUTPUT_METRIC_KEYS = ("auc", "aupr", "auc10", "accuracy", "precision", "recall", "mcc")
OUTPUT_METRIC_LABELS = {
    "auc": "auc",
    "aupr": "aupr",
    "auc10": "au10",
    "accuracy": "acc",
    "precision": "pre",
    "recall": "rec",
    "mcc": "mcc",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the final five-fold ESM2+AntigenLM context-CNN B-cell epitope model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trained-model-dir", type=str, default=DEFAULT_TRAINED_MODEL_DIR)
    parser.add_argument("--result-dir", type=str, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--test-fasta", type=str, default=DEFAULT_TEST_FASTA)
    parser.add_argument("--esm2-path", type=str, default=None, help="Optional override for the ESM2 encoder path.")
    parser.add_argument(
        "--antigenlm-path",
        type=str,
        default=None,
        help="Optional override for the AntigenLM encoder path.",
    )
    parser.add_argument("--fold-indices", type=int, nargs="+", default=None, help="Optional 1-based folds to test.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-prediction-csv", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_fasta_samples(fasta_path):
    samples = []
    current_id = None
    current_seq = []

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    samples.append(build_sample(current_id, "".join(current_seq)))
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        samples.append(build_sample(current_id, "".join(current_seq)))

    if not samples:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")
    return samples


def build_sample(sample_id, labeled_sequence):
    sequence = labeled_sequence.upper()
    labels = [1 if residue.isupper() else 0 for residue in labeled_sequence]
    if len(sequence) != len(labels):
        raise ValueError(f"Length mismatch in sample {sample_id}")
    return {"id": sample_id, "sequence": sequence, "labels": labels}


def result_label_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    label = "".join(char if char.isalnum() or char in "_.-" else "_" for char in stem).strip("_")
    return label or "test"


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def write_rows_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_cv_performance_csv(path, metrics_records):
    rows = []
    for metrics in sorted(metrics_records, key=lambda item: int(item["fold"])):
        row = {"fold": int(metrics["fold"])}
        for metric_name in OUTPUT_METRIC_KEYS:
            row[OUTPUT_METRIC_LABELS[metric_name]] = metrics[metric_name]
        rows.append(row)

    mean_row = {"fold": "mean"}
    std_row = {"fold": "std"}
    for metric_name in OUTPUT_METRIC_KEYS:
        values = [record[metric_name] for record in metrics_records]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        label = OUTPUT_METRIC_LABELS[metric_name]
        mean_row[label] = mean
        std_row[label] = math.sqrt(variance)
    rows.extend([mean_row, std_row])

    fieldnames = ["fold"] + [OUTPUT_METRIC_LABELS[key] for key in OUTPUT_METRIC_KEYS]
    write_rows_csv(path, rows, fieldnames)


def compute_roc_curve(labels, scores):
    labels = labels.to(torch.int64).cpu()
    scores = scores.to(torch.float64).cpu()
    positive_count = int(labels.sum().item())
    negative_count = int(labels.numel() - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None, None

    sorted_scores, order = torch.sort(scores, descending=True)
    sorted_labels = labels[order]
    tps = torch.cumsum(sorted_labels, dim=0, dtype=torch.float64)
    fps = torch.cumsum(1 - sorted_labels, dim=0, dtype=torch.float64)

    threshold_indices = torch.nonzero(sorted_scores[1:] != sorted_scores[:-1], as_tuple=False).flatten()
    threshold_indices = torch.cat(
        [threshold_indices, torch.tensor([sorted_labels.numel() - 1], dtype=torch.long)]
    )

    tps = tps[threshold_indices]
    fps = fps[threshold_indices]
    tps = torch.cat([torch.tensor([0.0], dtype=torch.float64), tps])
    fps = torch.cat([torch.tensor([0.0], dtype=torch.float64), fps])
    return fps / negative_count, tps / positive_count


def compute_auc(labels, scores):
    fpr, tpr = compute_roc_curve(labels, scores)
    if fpr is None or tpr is None:
        return 0.0
    return torch.trapz(tpr, fpr).item()


def compute_aupr(labels, scores):
    labels = labels.to(torch.int64).cpu()
    scores = scores.to(torch.float64).cpu()
    positive_count = int(labels.sum().item())
    if positive_count == 0:
        return 0.0

    sorted_scores, order = torch.sort(scores, descending=True)
    sorted_labels = labels[order]
    tps = torch.cumsum(sorted_labels, dim=0, dtype=torch.float64)
    fps = torch.cumsum(1 - sorted_labels, dim=0, dtype=torch.float64)

    threshold_indices = torch.nonzero(sorted_scores[1:] != sorted_scores[:-1], as_tuple=False).flatten()
    threshold_indices = torch.cat(
        [threshold_indices, torch.tensor([sorted_labels.numel() - 1], dtype=torch.long)]
    )

    recall = tps[threshold_indices] / positive_count
    precision = tps[threshold_indices] / (tps[threshold_indices] + fps[threshold_indices]).clamp(min=1)
    recall = torch.cat([torch.tensor([0.0], dtype=torch.float64), recall])
    precision = torch.cat([torch.tensor([1.0], dtype=torch.float64), precision])
    return torch.trapz(precision, recall).item()


def compute_auc10(labels, scores, max_fpr=0.1):
    fpr, tpr = compute_roc_curve(labels, scores)
    if fpr is None or tpr is None:
        return 0.0

    cutoff = float(max_fpr)
    cutoff_tensor = torch.tensor(cutoff, dtype=fpr.dtype)
    exact_matches = torch.nonzero(fpr == cutoff_tensor, as_tuple=False).flatten()
    if exact_matches.numel() > 0:
        end_idx = int(exact_matches[0].item()) + 1
        truncated_fpr = fpr[:end_idx]
        truncated_tpr = tpr[:end_idx]
    else:
        insertion_idx = int(torch.searchsorted(fpr, cutoff_tensor, right=False).item())
        left_idx = max(insertion_idx - 1, 0)
        right_idx = min(insertion_idx, fpr.numel() - 1)
        left_fpr = fpr[left_idx]
        right_fpr = fpr[right_idx]
        left_tpr = tpr[left_idx]
        right_tpr = tpr[right_idx]
        if right_fpr.item() == left_fpr.item():
            interpolated_tpr = right_tpr
        else:
            slope = (right_tpr - left_tpr) / (right_fpr - left_fpr)
            interpolated_tpr = left_tpr + slope * (cutoff_tensor - left_fpr)
        truncated_fpr = torch.cat([fpr[:insertion_idx], cutoff_tensor.unsqueeze(0)])
        truncated_tpr = torch.cat([tpr[:insertion_idx], interpolated_tpr.unsqueeze(0)])

    partial_auc = torch.trapz(truncated_tpr, truncated_fpr).item()
    return partial_auc / cutoff


def compute_metrics(probabilities, labels, threshold=0.5):
    probabilities = probabilities.cpu()
    labels = labels.cpu()
    predictions = (probabilities >= threshold).long()

    tp = int(((predictions == 1) & (labels == 1)).sum().item())
    tn = int(((predictions == 0) & (labels == 0)).sum().item())
    fp = int(((predictions == 1) & (labels == 0)).sum().item())
    fn = int(((predictions == 0) & (labels == 1)).sum().item())

    total = tp + tn + fp + fn
    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_divide(tp * tn - fp * fn, mcc_denominator)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "auc": compute_auc(labels, probabilities),
        "aupr": compute_aupr(labels, probabilities),
        "auc10": compute_auc10(labels, probabilities),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_predictions(path, residue_metadata, predictions, probabilities):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "residue_index",
                "residue",
                "true_label",
                "pred_label",
                "prob_epitope",
            ],
        )
        writer.writeheader()
        for meta, pred, prob in zip(residue_metadata, predictions.tolist(), probabilities.tolist()):
            writer.writerow(
                {
                    "sample_id": meta["sample_id"],
                    "residue_index": meta["residue_index"],
                    "residue": meta["residue"],
                    "true_label": meta["true_label"],
                    "pred_label": pred,
                    "prob_epitope": f"{prob:.6f}",
                }
            )


def fold_checkpoint_path(trained_model_dir, fold_idx):
    stem = checkpoint_stem(MODEL_NAME, CLASSIFIER_TYPE)
    return os.path.join(trained_model_dir, f"{stem}_fold{fold_idx}.pt")


def checkpoint_residue_features(checkpoint):
    raw_groups = checkpoint.get("residue_features")
    if raw_groups is None:
        raw_groups = (checkpoint.get("training_config") or {}).get("residue_features")
    if raw_groups is None:
        raw_groups = "none"
    return parse_residue_feature_groups(raw_groups)


def has_hf_encoder_config(path):
    return path and os.path.exists(os.path.join(path, "config.json"))


def resolve_checkpoint_encoder_ref(component_name, checkpoint_ref):
    default_paths = {
        "esm2": DEFAULT_ESM2_PATH,
        "antigenlm": DEFAULT_ANTIGENLM_PATH,
    }
    if has_hf_encoder_config(checkpoint_ref):
        return checkpoint_ref
    if not checkpoint_ref:
        return default_paths[component_name]

    checkpoint_ref = str(checkpoint_ref)
    if not os.path.isabs(checkpoint_ref) and os.path.sep not in checkpoint_ref:
        return checkpoint_ref

    basename = os.path.basename(checkpoint_ref.rstrip(os.path.sep))
    candidates = (
        default_paths[component_name],
        os.path.join(PROJECT_ROOT, "LLM", basename),
        os.path.join(PROJECT_ROOT, "LLM_MR", basename),
    )
    for candidate in candidates:
        if has_hf_encoder_config(candidate):
            return candidate
    return checkpoint_ref


def checkpoint_encoder_ref(checkpoint, args):
    checkpoint_ref = checkpoint.get("encoder_model_path")
    if isinstance(checkpoint_ref, dict):
        encoder_ref = {key: value for key, value in checkpoint_ref.items() if value}
    else:
        encoder_ref = {}

    if args.esm2_path:
        encoder_ref["esm2"] = args.esm2_path
    else:
        encoder_ref["esm2"] = resolve_checkpoint_encoder_ref("esm2", encoder_ref.get("esm2"))
    if args.antigenlm_path:
        encoder_ref["antigenlm"] = args.antigenlm_path
    else:
        encoder_ref["antigenlm"] = resolve_checkpoint_encoder_ref("antigenlm", encoder_ref.get("antigenlm"))
    return encoder_ref


def load_fold_resources(fold_idx, args, device):
    model_path = fold_checkpoint_path(args.trained_model_dir, fold_idx)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fold checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")

    model_name = checkpoint.get("model_name", MODEL_NAME)
    classifier_type = infer_classifier_type(checkpoint)
    if model_name != MODEL_NAME or classifier_type != CLASSIFIER_TYPE:
        raise ValueError(
            f"Unexpected checkpoint type in {model_path}: model={model_name}, classifier={classifier_type}. "
            f"This final test script only evaluates {MODEL_NAME}/{CLASSIFIER_TYPE}."
        )

    classifier_config = classifier_checkpoint_config(checkpoint)
    input_dim = checkpoint.get("input_dim", DEFAULT_RESIDUE_INPUT_DIM)
    residue_features = checkpoint_residue_features(checkpoint)
    classifier = create_residue_classifier(
        CLASSIFIER_TYPE,
        input_dim=input_dim,
        hidden_dim=classifier_config["hidden_dim"],
        dropout=classifier_config["dropout"],
        cnn_kernel_sizes=classifier_config.get("cnn_kernel_sizes", (3, 5, 9)),
    ).to(device)
    classifier.load_state_dict(classifier_state_dict(checkpoint))
    classifier.eval()

    encoder_ref = checkpoint_encoder_ref(checkpoint, args)
    feature_extractor = load_feature_extractor(MODEL_NAME, device, encoder_path=encoder_ref)
    apply_encoder_state_dict_delta(
        feature_extractor,
        checkpoint.get("encoder_trainable_state_dict"),
    )
    set_encoder_mode(feature_extractor, is_train=False)

    return {
        "checkpoint": checkpoint,
        "model_path": model_path,
        "classifier": classifier,
        "feature_extractor": feature_extractor,
        "classifier_config": classifier_config,
        "residue_features": residue_features,
    }


def release_resources(resources):
    del resources["classifier"], resources["feature_extractor"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_fold(fold_idx, resources, test_name, test_fasta, test_samples, args, device):
    classifier = resources["classifier"]
    feature_extractor = resources["feature_extractor"]
    residue_features = resources["residue_features"]

    all_probabilities = []
    all_predictions = []
    all_labels = []
    all_metadata = []
    truncated_sequences = 0

    with torch.no_grad():
        for start in range(0, len(test_samples), args.batch_size):
            batch = test_samples[start : start + args.batch_size]
            residue_batch = encode_residue_batch(
                feature_extractor,
                batch,
                device,
                args.max_length,
                include_metadata=not args.skip_prediction_csv,
                residue_feature_groups=residue_features,
            )
            logits = classifier(residue_batch["features"], residue_batch["residue_mask"])
            valid_logits = logits[residue_batch["residue_mask"]].cpu()
            labels = residue_batch["labels"][residue_batch["residue_mask"]].cpu()
            probabilities = torch.sigmoid(valid_logits)
            predictions = (probabilities >= 0.5).long()

            all_probabilities.append(probabilities)
            all_predictions.append(predictions)
            all_labels.append(labels)
            if not args.skip_prediction_csv:
                all_metadata.extend(residue_batch["metadata"])
            truncated_sequences += residue_batch["truncated_sequences"]

    if not all_probabilities:
        raise ValueError("No residue predictions were produced.")

    probabilities = torch.cat(all_probabilities, dim=0)
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(probabilities, labels)
    metrics.update(
        {
            "fold": fold_idx,
            "dataset": test_name,
            "test_fasta": test_fasta,
            "num_sequences": len(test_samples),
            "num_residues": int(labels.numel()),
            "truncated_sequences": truncated_sequences,
            "model_path": resources["model_path"],
            "best_epoch": resources["checkpoint"].get("best_epoch"),
            "best_score": resources["checkpoint"].get("best_score"),
        }
    )

    if not args.skip_prediction_csv:
        prediction_path = os.path.join(
            args.result_dir,
            f"{checkpoint_stem(MODEL_NAME, CLASSIFIER_TYPE)}_fold{fold_idx}_{test_name}_predictions.csv",
        )
        write_predictions(prediction_path, all_metadata, predictions, probabilities)

    print(
        f"fold{fold_idx} external test | "
        f"auc={metrics['auc']:.4f} "
        f"aupr={metrics['aupr']:.4f} "
        f"au10={metrics['auc10']:.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"pre={metrics['precision']:.4f} "
        f"rec={metrics['recall']:.4f} "
        f"mcc={metrics['mcc']:.4f}"
    )
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    os.makedirs(args.result_dir, exist_ok=True)

    test_name = result_label_from_path(args.test_fasta)
    test_samples = read_fasta_samples(args.test_fasta)
    fold_indices = args.fold_indices or list(range(1, CV_FOLDS + 1))

    all_metrics = []
    for fold_idx in fold_indices:
        resources = load_fold_resources(fold_idx, args, device)
        try:
            all_metrics.append(evaluate_fold(fold_idx, resources, test_name, args.test_fasta, test_samples, args, device))
        finally:
            release_resources(resources)

    full_fold_indices = list(range(1, CV_FOLDS + 1))
    fold_suffix = ""
    if fold_indices != full_fold_indices:
        fold_suffix = "_" + "_".join(f"fold{fold_idx}" for fold_idx in fold_indices)
    performance_path = os.path.join(
        args.result_dir,
        f"bcell_epitope_{MODEL_NAME}_{test_name}{fold_suffix}_cv_performance.csv",
    )
    write_cv_performance_csv(performance_path, all_metrics)
    print(f"Performance CSV saved: {os.path.basename(performance_path)}")

    auc_values = [metrics["auc"] for metrics in all_metrics]
    mean_auc = sum(auc_values) / len(auc_values)
    std_auc = math.sqrt(sum((value - mean_auc) ** 2 for value in auc_values) / len(auc_values))
    print(f"AUC mean ± std = {mean_auc:.6f} ± {std_auc:.6f}")


if __name__ == "__main__":
    main()
