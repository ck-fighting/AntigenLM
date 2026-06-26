import argparse
import math
import os
import random


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from feature_extractors import (
    DEFAULT_ESM2_ENCODER_PATH,
    augmented_input_dim,
    checkpoint_stem,
    collect_trainable_encoder_state_dict,
    encode_residue_batch,
    freeze_encoder,
    load_feature_extractor,
    parse_residue_feature_groups,
    residue_feature_dim,
    set_encoder_mode,
    unfreeze_last_hf_layers,
)
from model import WeightedBCEFocalLoss, create_residue_classifier, parse_kernel_sizes


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

MODEL_NAME = "esm2_antigenlm"
CLASSIFIER_TYPE = "context_cnn"
CV_FOLDS = 5
CV_FOLD_DIR = os.path.join(CURRENT_DIR, "data", "BP3C50ID_5fold")

DEFAULT_ESM2_PATH = DEFAULT_ESM2_ENCODER_PATH
DEFAULT_ANTIGENLM_PATH = os.path.join(PROJECT_ROOT, "LLM", "AntigenLM")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "Downstream",
    "trained_model",
    "B_cell_epitope",
)


class ResidueDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the final five-fold ESM2+AntigenLM context-CNN B-cell epitope model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--esm2-path", type=str, default=DEFAULT_ESM2_PATH)
    parser.add_argument("--antigenlm-path", type=str, default=DEFAULT_ANTIGENLM_PATH)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--cnn-kernel-sizes", type=str, default="3,5,9")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze-last-layers", type=int, default=1)
    parser.add_argument("--unfreeze-components", type=str, default="antigenlm")
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--rank-loss-weight", type=float, default=0.1)
    parser.add_argument("--rank-loss-negatives", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--residue-features", type=str, default="none")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fold-indices", type=int, nargs="+", default=None, help="Optional 1-based folds to train.")
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


def load_predefined_folds(cv_fold_dir, num_folds):
    cv_fold_dir = os.path.abspath(cv_fold_dir)
    if not os.path.isdir(cv_fold_dir):
        raise FileNotFoundError(f"CV fold directory not found: {cv_fold_dir}")

    folds = []
    seen_test_ids = set()
    for fold_idx in range(1, num_folds + 1):
        fold_dir = os.path.join(cv_fold_dir, f"fold_{fold_idx}")
        train_fasta = os.path.join(fold_dir, "train.fasta")
        test_fasta = os.path.join(fold_dir, "test.fasta")
        if not os.path.exists(train_fasta) or not os.path.exists(test_fasta):
            raise FileNotFoundError(f"Expected fold files not found: {train_fasta} and {test_fasta}")

        train_samples = read_fasta_samples(train_fasta)
        val_samples = read_fasta_samples(test_fasta)
        train_ids = {sample["id"] for sample in train_samples}
        val_ids = {sample["id"] for sample in val_samples}
        overlap = train_ids & val_ids
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"fold_{fold_idx} train/test overlap: {examples}")

        repeated = seen_test_ids & val_ids
        if repeated:
            examples = ", ".join(sorted(repeated)[:5])
            raise ValueError(f"fold_{fold_idx} test set overlaps previous folds: {examples}")
        seen_test_ids.update(val_ids)
        folds.append((train_samples, val_samples))
    return folds


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def count_residues(samples):
    positive_count = 0
    negative_count = 0
    for sample in samples:
        sample_positive_count = sum(sample["labels"])
        positive_count += sample_positive_count
        negative_count += len(sample["labels"]) - sample_positive_count
    return positive_count, negative_count


def parse_component_names(value):
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def iter_feature_extractors(feature_extractor):
    if feature_extractor.backend == "fusion":
        yield from feature_extractor.sub_extractors
    else:
        yield feature_extractor


def configure_encoder_trainability(feature_extractor, args):
    freeze_encoder(feature_extractor)
    if args.unfreeze_last_layers <= 0:
        return 0

    selected_components = parse_component_names(args.unfreeze_components)
    trainable_params = 0
    for sub_extractor in iter_feature_extractors(feature_extractor):
        if sub_extractor.name not in selected_components:
            continue
        trainable_params += unfreeze_last_hf_layers(sub_extractor, args.unfreeze_last_layers)
    return trainable_params


def trainable_encoder_parameters(feature_extractor):
    params = []
    for sub_extractor in iter_feature_extractors(feature_extractor):
        params.extend(param for param in sub_extractor.encoder.parameters() if param.requires_grad)
    return params


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


def compute_auc10(labels, scores, max_fpr=0.1):
    fpr, tpr = compute_roc_curve(labels, scores)
    if fpr is None or tpr is None:
        return 0.0

    cutoff = float(max_fpr)
    cutoff_tensor = torch.tensor(cutoff, dtype=fpr.dtype)
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
    return torch.trapz(truncated_tpr, truncated_fpr).item() / cutoff


def compute_metrics(logits, labels, threshold=0.5):
    probabilities = torch.sigmoid(logits).cpu()
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
        "auc10": compute_auc10(labels, probabilities),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def pairwise_rank_loss(logits, labels, max_negatives):
    positives = logits[labels > 0.5]
    negatives = logits[labels <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return logits.new_zeros(())

    if max_negatives and max_negatives > 0:
        if positives.numel() > max_negatives:
            positives = positives[torch.randperm(positives.numel(), device=positives.device)[:max_negatives]]
        if negatives.numel() > max_negatives:
            negatives = negatives[torch.randperm(negatives.numel(), device=negatives.device)[:max_negatives]]

    pairwise_margin = positives.unsqueeze(1) - negatives.unsqueeze(0)
    return F.softplus(-pairwise_margin).mean()


def run_epoch(feature_extractor, classifier, loader, criterion, device, args, residue_feature_groups, optimizer=None):
    is_train = optimizer is not None
    set_encoder_mode(feature_extractor, is_train)
    classifier.train(is_train)
    if is_train:
        optimizer.zero_grad()

    total_loss = 0.0
    total_residues = 0
    all_logits = []
    all_labels = []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for step, batch in enumerate(tqdm(loader, desc="Training" if is_train else "Validating")):
            residue_batch = encode_residue_batch(
                feature_extractor,
                batch,
                device,
                args.max_length,
                residue_feature_groups=residue_feature_groups,
            )
            features = residue_batch["features"]
            labels = residue_batch["labels"]
            residue_mask = residue_batch["residue_mask"]

            logits = classifier(features, residue_mask)
            valid_logits = logits[residue_mask]
            valid_labels = labels[residue_mask]
            if valid_logits.numel() == 0:
                continue

            loss = criterion(valid_logits, valid_labels)
            if is_train and args.rank_loss_weight > 0:
                loss = loss + args.rank_loss_weight * pairwise_rank_loss(
                    valid_logits,
                    valid_labels,
                    args.rank_loss_negatives,
                )
            if is_train:
                (loss / args.gradient_accumulation_steps).backward()
                if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()

            residue_count = int(valid_labels.numel())
            total_loss += float(loss.item()) * residue_count
            total_residues += residue_count
            all_logits.append(valid_logits.detach().cpu())
            all_labels.append(valid_labels.detach().cpu())

    if not all_logits:
        raise ValueError("No residue logits were produced during the epoch.")

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / max(total_residues, 1)
    return metrics


def train_fold(fold_idx, train_samples, val_samples, device, args):
    set_seed(args.seed + fold_idx - 1)

    feature_extractor = load_feature_extractor(
        MODEL_NAME,
        device,
        esm2_path=args.esm2_path,
        antigenlm_path=args.antigenlm_path,
    )
    trainable_encoder_param_count = configure_encoder_trainability(feature_extractor, args)
    print(f"Trainable encoder parameters: {trainable_encoder_param_count}")

    train_loader = DataLoader(
        ResidueDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch,
    )
    val_loader = DataLoader(
        ResidueDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )

    cnn_kernel_sizes = parse_kernel_sizes(args.cnn_kernel_sizes)
    residue_feature_groups = parse_residue_feature_groups(args.residue_features)
    input_dim = augmented_input_dim(feature_extractor.input_dim, residue_feature_groups)
    print(
        f"Classifier input dim: lm={feature_extractor.input_dim} "
        f"residue_features={residue_feature_dim(residue_feature_groups)} total={input_dim}"
    )
    classifier = create_residue_classifier(
        CLASSIFIER_TYPE,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        cnn_kernel_sizes=cnn_kernel_sizes,
    ).to(device)

    positive_count, negative_count = count_residues(train_samples)
    val_positive_count, val_negative_count = count_residues(val_samples)
    base_pos_weight = negative_count / max(positive_count, 1)
    pos_weight = base_pos_weight * args.pos_weight_scale
    print(
        f"Residue class counts | train_pos={positive_count} train_neg={negative_count} "
        f"val_pos={val_positive_count} val_neg={val_negative_count} "
        f"base_pos_weight={base_pos_weight:.4f} pos_weight={pos_weight:.4f}"
    )

    criterion = WeightedBCEFocalLoss(pos_weight=pos_weight, gamma=args.focal_gamma).to(device)
    optimizer_param_groups = [{"params": classifier.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}]
    encoder_params = trainable_encoder_parameters(feature_extractor)
    if encoder_params:
        optimizer_param_groups.append(
            {"params": encoder_params, "lr": args.encoder_lr, "weight_decay": args.weight_decay}
        )
    optimizer = torch.optim.AdamW(optimizer_param_groups)

    stem = f"{checkpoint_stem(MODEL_NAME, CLASSIFIER_TYPE)}_fold{fold_idx}"
    model_path = os.path.join(args.output_dir, f"{stem}.pt")
    best_score = -1.0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            feature_extractor,
            classifier,
            train_loader,
            criterion,
            device,
            args,
            residue_feature_groups,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            feature_extractor,
            classifier,
            val_loader,
            criterion,
            device,
            args,
            residue_feature_groups,
            optimizer=None,
        )
        epoch_record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        score = val_metrics["auc"]
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"train_auc={train_metrics['auc']:.4f} train_mcc={train_metrics['mcc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_mcc={val_metrics['mcc']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_metrics = epoch_record
            checkpoint = {
                "classifier_state_dict": classifier.state_dict(),
                "classifier_type": CLASSIFIER_TYPE,
                "classifier_config": {
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "cnn_kernel_sizes": cnn_kernel_sizes,
                    "bilstm_layers": 1,
                },
                "input_dim": input_dim,
                "lm_input_dim": feature_extractor.input_dim,
                "residue_features": residue_feature_groups,
                "residue_feature_dim": residue_feature_dim(residue_feature_groups),
                "model_name": MODEL_NAME,
                "model_stem": stem,
                "cv_fold": fold_idx,
                "cv_folds": CV_FOLDS,
                "best_epoch": epoch,
                "best_score_name": "auc",
                "best_score": best_score,
                "best_metrics": best_metrics,
                "loss_name": "weighted_bce_focal",
                "pos_weight": pos_weight,
                "focal_gamma": args.focal_gamma,
                "max_length": args.max_length,
                "encoder_model_path": feature_extractor.encoder_ref,
                "source_encoder_model_path": feature_extractor.encoder_ref,
                "encoder_trainable_state_dict": collect_trainable_encoder_state_dict(feature_extractor),
                "training_config": {
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "encoder_lr": args.encoder_lr,
                    "weight_decay": args.weight_decay,
                    "unfreeze_last_layers": args.unfreeze_last_layers,
                    "unfreeze_components": args.unfreeze_components,
                    "trainable_encoder_param_count": trainable_encoder_param_count,
                    "pos_weight_scale": args.pos_weight_scale,
                    "rank_loss_weight": args.rank_loss_weight,
                    "rank_loss_negatives": args.rank_loss_negatives,
                    "residue_features": residue_feature_groups,
                    "seed": args.seed,
                },
            }
            torch.save(checkpoint, model_path)

    print(f"Best fold {fold_idx} model saved to: {model_path}")
    print(f"Best fold {fold_idx} val_auc: {best_score:.4f}")

    del classifier, feature_extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"fold": fold_idx, "model_path": model_path, "best_score": best_score, "best_metrics": best_metrics}


def main():
    args = parse_args()
    args.residue_features = ",".join(parse_residue_feature_groups(args.residue_features))

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Model: {MODEL_NAME}")
    print(f"Classifier: {CLASSIFIER_TYPE}")
    print(f"CV fold dir: {CV_FOLD_DIR}")
    print(f"ESM2 path: {args.esm2_path}")
    print(f"AntigenLM path: {args.antigenlm_path}")
    print(f"Model output dir: {args.output_dir}")
    print(
        f"Training config: hidden={args.hidden_dim} dropout={args.dropout} kernels={args.cnn_kernel_sizes} "
        f"epochs={args.epochs} batch={args.batch_size} grad_accum={args.gradient_accumulation_steps} "
        f"unfreeze_last_layers={args.unfreeze_last_layers} unfreeze_components={args.unfreeze_components} "
        f"encoder_lr={args.encoder_lr} residue_features={args.residue_features or 'none'} "
        f"rank_loss_weight={args.rank_loss_weight}"
    )

    folds = load_predefined_folds(CV_FOLD_DIR, CV_FOLDS)
    selected_fold_indices = set(args.fold_indices or range(1, CV_FOLDS + 1))
    for fold_idx, (train_samples, val_samples) in enumerate(folds, start=1):
        if fold_idx not in selected_fold_indices:
            continue
        print("=" * 80)
        print(f"Training fold {fold_idx}/{CV_FOLDS}")
        print(f"Train sequences: {len(train_samples)}")
        print(f"Validation fold sequences: {len(val_samples)}")
        train_fold(fold_idx, train_samples, val_samples, device, args)


if __name__ == "__main__":
    main()
