import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from feature_extractor import (
    EMBED_BACKENDS,
    _is_cuda,
    get_model_and_extract_func,
    resolve_backend_path,
    resolve_hf_extract_dtype,
)
from protective_antigen_model import SoluModel
from protective_antigen_utils import (
    AntigenDataset,
    DEFAULT_CV_DATA_DIR,
    DEFAULT_PLDGL_DATA_DIR,
    P,
    best_threshold_metrics,
    binary_metrics,
    discover_cv_train_files,
    parse_subset_list,
    print_split_counts,
    read_labeled_table,
    resolve_cv_datasets,
    sanitize_name,
    setup_seed,
)


def split_train_val(csv_path, val_ratio=0.15, seed=22):
    import pandas as pd

    df = read_labeled_table(csv_path)
    pos_df = df[df["label"] == 1].reset_index(drop=True)
    neg_df = df[df["label"] == 0].reset_index(drop=True)
    n_val_pos = int(len(pos_df) * val_ratio)
    n_val_neg = min(n_val_pos * 10, len(neg_df))

    rng = np.random.default_rng(seed)
    val_pos_idx = rng.choice(pos_df.index, size=n_val_pos, replace=False).tolist()
    val_neg_idx = rng.choice(neg_df.index, size=n_val_neg, replace=False).tolist()

    val_df = pd.concat([pos_df.loc[val_pos_idx], neg_df.loc[val_neg_idx]]).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    train_df = pd.concat([pos_df.drop(val_pos_idx), neg_df.drop(val_neg_idx)]).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    print_split_counts("Train set", train_df)
    print_split_counts("Val set", val_df)
    return train_df, val_df


def _make_loader(embeddings, labels, batch_size, device, use_sampler=False, sampler_pos_fraction=0.5):
    labels_np = np.asarray(labels, dtype=np.int64)
    sampler = None
    shuffle = True
    if use_sampler:
        class_sample_count = np.bincount(labels_np)
        sampler_pos_fraction = float(np.clip(sampler_pos_fraction, 1e-3, 1.0 - 1e-3))
        class_targets = np.asarray([1.0 - sampler_pos_fraction, sampler_pos_fraction], dtype=np.float64)
        sample_weights = (class_targets / class_sample_count)[labels_np]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        list(zip(embeddings, labels_np.tolist())),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=False,
        pin_memory=_is_cuda(device),
    )


def _build_scheduler(optimizer, total_steps, warmup_ratio):
    if warmup_ratio <= 0 or total_steps <= 0:
        return None
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _focal_loss_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets.bool(), probs, 1.0 - probs)
    alpha_t = torch.where(
        targets.bool(),
        torch.full_like(targets, alpha),
        torch.full_like(targets, 1.0 - alpha),
    )
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def _dynamic_focal_loss_with_logits(
    logits,
    targets,
    global_step=0,
    total_steps=1,
    current_aupr=None,
    initial_alpha=0.75,
    initial_gamma=2.0,
):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets.bool(), probs, 1.0 - probs)
    progress = min(1.0, global_step / max(1, total_steps))
    adaptive_alpha = initial_alpha * (1.0 - 0.2 * progress)
    adaptive_gamma = (
        initial_gamma * (1.2 - 0.4 * min(current_aupr / 0.7, 1.0))
        if current_aupr is not None
        else initial_gamma * (1.1 - 0.3 * progress)
    )

    alpha_t = torch.where(
        targets.bool(),
        torch.full_like(targets, adaptive_alpha),
        torch.full_like(targets, 1.0 - adaptive_alpha),
    )

    return (alpha_t * (1.0 - pt).pow(adaptive_gamma) * bce).mean()


def _classification_loss(
    logits,
    targets,
    criterion,
    loss_type,
    focal_alpha,
    focal_gamma,
    global_step=None,
    total_steps=None,
    current_aupr=None,
    use_dynamic_focal=False,
):
    if loss_type == "focal":
        if use_dynamic_focal and global_step is not None and total_steps is not None:
            return _dynamic_focal_loss_with_logits(
                logits,
                targets,
                global_step=global_step,
                total_steps=total_steps,
                current_aupr=current_aupr,
                initial_alpha=focal_alpha,
                initial_gamma=focal_gamma,
            )
        return _focal_loss_with_logits(logits, targets, alpha=focal_alpha, gamma=focal_gamma)
    return criterion(logits, targets)


def _model_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def train(
    train_dataset,
    eval_dataset,
    extract_emb_func,
    emb_dim,
    epochs,
    batch_size,
    device,
    save_path,
    lr,
    weight_decay,
    threshold,
    use_sampler=False,
    eval_name="Val",
    hidden_dim=384,
    dropout=0.25,
    grad_clip=1.0,
    warmup_ratio=0.08,
    patience=8,
    min_delta=1e-4,
    select_metric="aupr",
    sampler_pos_fraction=0.5,
    pos_weight_scale=1.0,
    loss_type="bce",
    focal_alpha=0.75,
    focal_gamma=2.0,
    use_dynamic_focal=False,
    print_best_thresholds=True,
):
    _, train_sequences, train_labels = train_dataset.get_data()
    if eval_dataset is not None:
        _, eval_sequences, eval_labels = eval_dataset.get_data()
    else:
        eval_sequences, eval_labels = [], []

    print("Extracting train embeddings ...")
    train_embeddings = extract_emb_func(train_sequences)
    if eval_dataset is not None:
        print(f"Extracting {eval_name.lower()} embeddings ...")
        eval_embeddings = extract_emb_func(eval_sequences)
    else:
        eval_embeddings = None

    labels_flat = np.asarray(train_labels, dtype=np.int64)
    pos = int(np.sum(labels_flat == 1))
    neg = int(np.sum(labels_flat == 0))

    train_loader = _make_loader(
        train_embeddings,
        train_labels,
        batch_size,
        device,
        use_sampler=use_sampler,
        sampler_pos_fraction=sampler_pos_fraction,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            list(zip(eval_embeddings, np.asarray(eval_labels, dtype=np.int64).tolist())),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=_is_cuda(device),
        )

    model_clf = SoluModel(seq_len=512, in_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    if torch.cuda.device_count() > 1 and _is_cuda(device):
        model_clf = nn.DataParallel(model_clf)

    total_steps = epochs * max(len(train_loader), 1)
    optimizer = torch.optim.AdamW(model_clf.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _build_scheduler(optimizer, total_steps, warmup_ratio)
    pos_weight = (neg / max(pos, 1)) * pos_weight_scale
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float, device=device)
    )
    if use_sampler:
        print(f"Sampler target Pos fraction: {sampler_pos_fraction:.3f}")
    print(f"Loss: {loss_type} | Pos weight: {pos_weight:.4f} | Focal alpha/gamma: {focal_alpha}/{focal_gamma}")

    best_score = -1.0
    stale_epochs = 0
    current_aupr = None
    step_counter = 0

    for epoch in range(1, epochs + 1):
        model_clf.train()
        total_loss, train_probs, train_trues = 0.0, [], []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = torch.as_tensor(y_batch, dtype=torch.float, device=device).unsqueeze(1)

            logits, _ = model_clf(x_batch)

            loss = _classification_loss(
                logits,
                y_batch,
                criterion,
                loss_type,
                focal_alpha,
                focal_gamma,
                global_step=step_counter,
                total_steps=total_steps,
                current_aupr=current_aupr,
                use_dynamic_focal=use_dynamic_focal,
            )
            step_counter += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model_clf.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * x_batch.size(0)
            train_probs.extend(torch.sigmoid(logits).squeeze(-1).detach().cpu().tolist())
            train_trues.extend(y_batch.squeeze(-1).long().cpu().tolist())

        train_metrics = binary_metrics(train_trues, train_probs, threshold)
        avg_loss = total_loss / max(len(train_loader.dataset), 1)

        if eval_loader is None:
            print(
                f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_metrics['acc']:.4f} | Train AUPR: {train_metrics['aupr']:.4f}"
            )
            continue

        model_clf.eval()
        eval_trues, eval_probs = [], []
        with torch.no_grad():
            for x_batch, y_batch in eval_loader:
                logits, _ = model_clf(x_batch.to(device, non_blocking=True))
                probs = torch.sigmoid(logits).squeeze(-1)
                eval_trues.extend(y_batch.tolist())
                eval_probs.extend(probs.cpu().tolist())

        eval_metrics = binary_metrics(eval_trues, eval_probs, threshold)

        current_aupr = eval_metrics.get("aupr")

        best_f1_threshold, best_f1_metrics = best_threshold_metrics(eval_trues, eval_probs, "f1")
        best_mcc_threshold, best_mcc_metrics = best_threshold_metrics(eval_trues, eval_probs, "mcc")
        print(
            f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_metrics['acc']:.4f} | "
            f"{eval_name} Acc: {eval_metrics['acc']:.4f} | "
            f"P: {eval_metrics['precision']:.4f} | R: {eval_metrics['recall']:.4f} | "
            f"F1: {eval_metrics['f1']:.4f} | MCC: {eval_metrics['mcc']:.4f} | "
            f"AUC: {eval_metrics['auc']:.4f} | AUPR: {eval_metrics['aupr']:.4f}"
        )
        if print_best_thresholds:
            print(
                f"    Best thresholds on {eval_name}: "
                f"F1={best_f1_threshold:.2f} ({best_f1_metrics['f1']:.4f}), "
                f"MCC={best_mcc_threshold:.2f} ({best_mcc_metrics['mcc']:.4f})"
            )

        score = eval_metrics[select_metric]
        if not np.isnan(score) and score > best_score + min_delta:
            best_score = score
            stale_epochs = 0
            torch.save(_model_state_dict(model_clf), save_path)
            print(f">>> New best model saved: {save_path} ({eval_name} {select_metric.upper()}: {best_score:.4f})")
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                print(f">>> Early stopping at epoch {epoch}: no {select_metric.upper()} improvement for {patience} epochs")
                break

    if eval_loader is None:
        torch.save(_model_state_dict(model_clf), save_path)
        print(f">>> Final model saved: {save_path} (no eval set)")

    return save_path


def resolve_pldgl_files(data_dir, subset, train_file=None, test_file=None):
    if train_file or test_file:
        if not train_file or not test_file:
            raise ValueError("--train_file and --test_file must be provided together in Independent mode")
        return os.path.abspath(train_file), os.path.abspath(test_file)

    train_path = P(data_dir, f"train_set_{subset}.xlsx")
    test_path = P(data_dir, f"test_set_{subset}.xlsx")
    return os.path.abspath(train_path), os.path.abspath(test_path)


def run_pldgl_subset(args, subset, extract_emb_func, emb_dim, device):
    train_xlsx, test_xlsx = resolve_pldgl_files(
        args.independent_data_dir,
        subset,
        train_file=args.train_file,
        test_file=args.test_file,
    )
    train_df = read_labeled_table(train_xlsx)
    test_df = read_labeled_table(test_xlsx)

    run_name = sanitize_name(f"PLDGL_{subset}")
    run_save_dir = P(args.save_dir, "PLDGL")
    os.makedirs(run_save_dir, exist_ok=True)
    save_path = P(run_save_dir, f"{run_name}_seed{args.seed}_{args.embed_backend}.pt")

    print(f"========== Independent PLDGL {subset} ==========")
    print(f"Train xlsx: {train_xlsx}")
    print(f"Validation/Test xlsx: {test_xlsx}")
    print(f"Save path: {save_path}")
    print_split_counts("Train set", train_df)
    print_split_counts("Independent set", test_df)

    return train(
        train_dataset=AntigenDataset(train_df["sequence"], train_df["label"]),
        eval_dataset=AntigenDataset(test_df["sequence"], test_df["label"]),
        extract_emb_func=extract_emb_func,
        emb_dim=emb_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        save_path=save_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        use_sampler=args.use_sampler,
        eval_name="Independent",
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
        min_delta=args.min_delta,
        select_metric=args.select_metric,
        sampler_pos_fraction=args.sampler_pos_fraction,
        pos_weight_scale=args.pos_weight_scale,
        loss_type=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        use_dynamic_focal=args.use_dynamic_focal,
        print_best_thresholds=not args.no_threshold_search,
    )


def main():
    p = argparse.ArgumentParser(description=" training for Protective Antigen Classification")
    p.add_argument(
        "--mode",
        type=str,
        choices=["CV", "Independent", "cv", "independent"],
        default="CV",
        help="CV trains the default data train/val folds; Independent trains the PLDGL route.",
    )
    p.add_argument("--data_dir", type=str, default=None, help="CV data root; defaults to ./data")
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional CV checkpoint subdirectory name; defaults to the dataset directory name.",
    )
    p.add_argument("--folds", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="../trained_model/protective_antigen")
    p.add_argument(
        "--embed_backend",
        type=str,
        choices=EMBED_BACKENDS,
        default="AntigenLM",
    )
    p.add_argument("--backend_path", type=str, default=None)
    p.add_argument("--hf_extract_batch_size", type=int, default=8)
    p.add_argument("--hf_extract_dtype", type=str, choices=["none", "fp32", "auto", "bf16", "fp16"], default="none")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--use_sampler", action="store_true", help="Enable WeightedRandomSampler")
    p.add_argument(
        "--sampler_pos_fraction",
        type=float,
        default=0.5,
        help="Target positive fraction for WeightedRandomSampler when --use_sampler is enabled.",
    )
    p.add_argument("--pos_weight_scale", type=float, default=1.0)
    p.add_argument("--loss", type=str, choices=["bce", "focal"], default="focal")
    p.add_argument("--focal_alpha", type=float, default=0.75)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.08)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument(
        "--select_metric",
        type=str,
        choices=["auc", "aupr", "acc", "precision", "recall", "f1", "mcc"],
        default="aupr",
    )
    p.add_argument(
        "--hf_no_special_tokens",
        action="store_true",
        help="For AntigenLM/HuggingFace backends, encode residue tokens without CLS/SEP.",
    )
    p.add_argument("--no_threshold_search", action="store_true")
    p.add_argument("--use_dynamic_focal", action="store_true", help="Enable dynamic focal loss with scheduling based on training progress and AUPR feedback.")
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--legacy_split_train_val", action="store_true")
    p.add_argument("--independent_data_dir", type=str, default=None, help="PLDGL data directory for --mode Independent")
    p.add_argument(
        "--subset",
        type=str,
        default="All",
        help="PLDGL subset: All, Bacteria, Eukaryota, Viruses, or a comma-separated list",
    )
    p.add_argument("--run_all_subsets", action="store_true", help="Run all PLDGL subsets in Independent mode")
    p.add_argument("--train_file", type=str, default=None, help="Explicit PLDGL train xlsx for Independent mode")
    p.add_argument("--test_file", type=str, default=None, help="Explicit PLDGL test xlsx for Independent mode")
    args = p.parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "Independent" if args.mode.lower() == "independent" else "CV"
    if mode == "CV":
        args.data_dir = os.path.abspath(args.data_dir or DEFAULT_CV_DATA_DIR)
    else:
        args.independent_data_dir = os.path.abspath(
            args.independent_data_dir or args.data_dir or DEFAULT_PLDGL_DATA_DIR
        )

    print(f"Mode: {mode}")
    print(f"Model: masked pooling | Embedding: {args.embed_backend} | Device: {device}")
    print(f"Learning rate: {args.lr} | Weight decay: {args.weight_decay} | Threshold: {args.threshold}")
    print(
        f"Loss: {args.loss} | Pos weight scale: {args.pos_weight_scale} | "
        f"Sampler pos fraction: {args.sampler_pos_fraction}"
    )
    print(
        f"Classifier hidden_dim: {args.hidden_dim} | Dropout: {args.dropout} | "
        f"Warmup ratio: {args.warmup_ratio} | Select metric: {args.select_metric}"
    )
    print(f"Weighted sampler: {args.use_sampler}")
    if args.use_sampler and args.loss == "bce" and args.pos_weight_scale > 0:
        print(
            "[warn] --use_sampler with BCE pos_weight applies two imbalance corrections. "
            "Usually use either sampler or pos_weight, not both."
        )
    if args.loss == "focal":
        print("[note] Focal loss does not use BCE pos_weight; focal_alpha controls class weighting.")
    print(f"HF add special tokens: {not args.hf_no_special_tokens}")
    print(f"HF extract batch size: {args.hf_extract_batch_size} | dtype: {args.hf_extract_dtype}")
    print(f"Dynamic focal loss: {args.use_dynamic_focal}")
    resolved_backend_path = resolve_backend_path(args.embed_backend, args.backend_path)
    print(f"Backend path: {resolved_backend_path or args.embed_backend}")
    extract_emb_func, emb_dim = get_model_and_extract_func(
        args.embed_backend,
        resolved_backend_path,
        device,
        hf_add_special_tokens=not args.hf_no_special_tokens,
        hf_extract_batch_size=args.hf_extract_batch_size,
        hf_autocast_dtype=resolve_hf_extract_dtype(args.hf_extract_dtype, device),
    )

    if mode == "Independent":
        print(f"Independent data directory: {args.independent_data_dir}")
        for subset in parse_subset_list(args.subset, args.run_all_subsets):
            setup_seed(args.seed)
            run_pldgl_subset(args, subset, extract_emb_func, emb_dim, device)
        return

    print(f"CV data root: {args.data_dir}")
    print("CV training uses train_fold_*.csv and val_fold_*.csv; test_fold_*.csv is reserved for protective_antigen_test.py.")

    for cv_dir, dataset_label in resolve_cv_datasets(args.data_dir, args.dataset_name):
        run_name = sanitize_name(args.run_name or dataset_label)
        run_save_dir = P(args.save_dir, run_name)
        os.makedirs(run_save_dir, exist_ok=True)

        print(f"CV dataset: {dataset_label}")
        print(f"CV directory: {cv_dir}")
        print(f"Save directory: {run_save_dir}")

        for fold_idx, train_csv, val_csv, _test_csv in discover_cv_train_files(cv_dir, args.folds):
            if not os.path.exists(train_csv):
                raise FileNotFoundError(f"Train CSV not found: {train_csv}")

            if os.path.exists(val_csv):
                train_df = read_labeled_table(train_csv)
                eval_df = read_labeled_table(val_csv)
                eval_name = "Val"
            elif args.legacy_split_train_val:
                train_df, eval_df = split_train_val(train_csv, val_ratio=0.15, seed=args.seed)
                eval_name = "Val"
            else:
                raise FileNotFoundError(
                    f"Validation CSV not found for fold {fold_idx}: {val_csv}. "
                    "CV training uses validation data for model selection; use --legacy_split_train_val only for old train-only data."
                )

            print(f"========== {dataset_label} Fold {fold_idx} ==========")
            print(f"Train CSV: {train_csv}")
            print(f"Val CSV: {val_csv}")
            print_split_counts("Train set", train_df)
            print_split_counts(f"{eval_name} set", eval_df)

            train(
                train_dataset=AntigenDataset(train_df["sequence"], train_df["label"]),
                eval_dataset=AntigenDataset(eval_df["sequence"], eval_df["label"]),
                extract_emb_func=extract_emb_func,
                emb_dim=emb_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                save_path=P(run_save_dir, f"fold_{fold_idx}_seed{args.seed}_{args.embed_backend}.pt"),
                lr=args.lr,
                weight_decay=args.weight_decay,
                threshold=args.threshold,
                use_sampler=args.use_sampler,
                eval_name=eval_name,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                grad_clip=args.grad_clip,
                warmup_ratio=args.warmup_ratio,
                patience=args.patience,
                min_delta=args.min_delta,
                select_metric=args.select_metric,
                sampler_pos_fraction=args.sampler_pos_fraction,
                pos_weight_scale=args.pos_weight_scale,
                loss_type=args.loss,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
                use_dynamic_focal=args.use_dynamic_focal,
                print_best_thresholds=not args.no_threshold_search,
            )


if __name__ == "__main__":
    main()
