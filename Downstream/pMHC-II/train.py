import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from losses import WeightedBCEWithContrastiveLoss
from metrics import compute_metrics
from model import AntigenLMESM2PMHCIIModel, count_trainable_parameters
from pmhc_data import PMHCIIEmbeddingDataset, batch_to_device, build_hla_store, distilled_cache_prefix_for_split


CURRENT_DIR = Path(__file__).resolve().parent
DOWNSTREAM_DIR = CURRENT_DIR.parent


def parse_folds(value):
    if value.lower() == "all":
        return [1, 2, 3, 4, 5]
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train pMHC-II with cached peptide tokens, ESM2 MHC-II embeddings, explicit matching features, weighted BCE, and supervised contrastive learning."
    )
    parser.add_argument("--root", type=Path, default=CURRENT_DIR)
    parser.add_argument("--mode", choices=("cv_train", "train"), default="cv_train")
    parser.add_argument("--folds", type=str, default="all", help="CV folds to train, e.g. all or 1,2,3.")
    parser.add_argument("--output-dir", type=Path, default=DOWNSTREAM_DIR / "trained_model" / "pMHC-II")
    parser.add_argument("--init-checkpoint-dir", type=Path, default=None, help="Optional directory containing fold*.pt checkpoints used to initialize matching model weights.")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional checkpoint used to initialize every run.")
    parser.add_argument("--init-allow-partial", action="store_true", help="Allow warm-starting from checkpoints with missing/new architecture keys.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cache-max-length", type=int, default=34)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--peptide-layers", type=int, default=2)
    parser.add_argument("--hla-layers", type=int, default=2)
    parser.add_argument("--hla-fusion-layers", type=int, default=1)
    parser.add_argument("--cross-attention-layers", type=int, default=0, help="Kept for checkpoint config compatibility; current matching model does not use cross-attention.")
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--aggregation", choices=("pool", "mil_window"), default="pool")
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--window-representation", choices=("mean", "position_aware", "dsca"), default="mean")
    parser.add_argument("--mil-pooling", choices=("softmax", "max"), default="softmax")
    parser.add_argument("--mil-temperature", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--window-entropy-weight", type=float, default=0.0)
    parser.add_argument("--window-margin-weight", type=float, default=0.0)
    parser.add_argument("--window-margin", type=float, default=1.0)
    parser.add_argument("--window-loss-temperature", type=float, default=0.25)
    parser.add_argument("--pos-weight", type=float, default=None, help="Default: train negatives / train positives.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--best-metric", choices=("auc", "aupr", "mcc", "f1", "loss"), default="auc")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Used only by --mode train.")
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(dataset, batch_size, shuffle, num_workers):
    options = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


def subset_counts(dataset, indices):
    labels = dataset.labels[np.asarray(indices, dtype=np.int64)]
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    return positives, negatives


def pos_weight_from_counts(positives, negatives):
    return negatives / max(positives, 1)


def model_from_args(args):
    model = AntigenLMESM2PMHCIIModel(
        peptide_dim=960,
        hla_dim=1152,
        hidden_dim=args.hidden_dim,
        peptide_layers=args.peptide_layers,
        hla_layers=args.hla_layers,
        hla_fusion_layers=args.hla_fusion_layers,
        cross_attention_layers=args.cross_attention_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        projection_dim=args.projection_dim,
        aggregation=args.aggregation,
        window_size=args.window_size,
        window_representation=args.window_representation,
        mil_pooling=args.mil_pooling,
        mil_temperature=args.mil_temperature,
    )
    return model


def metric_score(metrics, best_metric):
    if best_metric == "loss":
        return -metrics["loss"]
    return metrics[best_metric]


def run_epoch(model, data_loader, criterion, device, use_amp, optimizer=None, scaler=None, threshold=0.5):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_bce = 0.0
    total_contrastive = 0.0
    total_window_entropy = 0.0
    total_window_margin = 0.0
    total_samples = 0
    all_scores = []
    all_labels = []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        iterator = tqdm(data_loader, desc="train" if is_train else "eval", leave=False, dynamic_ncols=True)
        for batch in iterator:
            tensors = batch_to_device(batch, device)
            labels = tensors["label"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(
                    tensors["peptide_embedding"],
                    tensors["peptide_mask"],
                    tensors["alpha_index"],
                    tensors["beta_index"],
                    return_embedding=True,
                    return_core=criterion.needs_core_info,
                )
                if criterion.needs_core_info:
                    logits, embeddings, core_info = output
                else:
                    logits, embeddings = output
                    core_info = None
                loss, loss_parts = criterion(logits, labels, embeddings, core_info=core_info)

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            batch_size = int(labels.numel())
            total_samples += batch_size
            total_loss += loss_parts["total"] * batch_size
            total_bce += loss_parts["bce"] * batch_size
            total_contrastive += loss_parts["contrastive"] * batch_size
            total_window_entropy += loss_parts.get("window_entropy", 0.0) * batch_size
            total_window_margin += loss_parts.get("window_margin", 0.0) * batch_size
            all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    labels_np = np.concatenate(all_labels)
    scores_np = np.concatenate(all_scores)
    metrics = compute_metrics(labels_np, scores_np, threshold=threshold)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["bce_loss"] = total_bce / max(total_samples, 1)
    metrics["contrastive_loss"] = total_contrastive / max(total_samples, 1)
    metrics["window_entropy_loss"] = total_window_entropy / max(total_samples, 1)
    metrics["window_margin_loss"] = total_window_margin / max(total_samples, 1)
    return metrics


def save_checkpoint(path, model, optimizer, args, epoch, metrics, pos_weight):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.config,
            "epoch": epoch,
            "metrics": metrics,
            "pos_weight": pos_weight,
            "encoder_setup": {
                "peptide": f"Cached AntigenLM_distilled token embeddings, cache max length={args.cache_max_length}",
                "mhc_ii_alpha_beta": "ESM2 cached embeddings from data/hla_dict/hla_esm_dict.npy, shape (98, 100, 1152)",
                "mhc_fusion": "alpha and beta ESM2 token embeddings are concatenated on token dimension before MHC encoding",
                "interaction": "explicit peptide/HLA and alpha/beta matching features",
                "aggregation": args.aggregation,
                "window_size": args.window_size if args.aggregation == "mil_window" else None,
                "window_representation": args.window_representation if args.aggregation == "mil_window" else None,
                "mil_pooling": args.mil_pooling if args.aggregation == "mil_window" else None,
                "mil_temperature": args.mil_temperature if args.aggregation == "mil_window" else None,
                "window_entropy_weight": args.window_entropy_weight,
                "window_margin_weight": args.window_margin_weight,
                "window_margin": args.window_margin,
                "window_loss_temperature": args.window_loss_temperature,
            },
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        path,
    )


def train_one_run(args, train_dataset, val_dataset, run_name, train_counts):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    use_amp = bool((not args.no_amp) and device.type == "cuda")
    positives, negatives = train_counts
    pos_weight = args.pos_weight if args.pos_weight is not None else pos_weight_from_counts(positives, negatives)

    model = model_from_args(args).to(device)
    init_checkpoint_path = None
    if args.init_checkpoint is not None:
        init_checkpoint_path = args.init_checkpoint
    elif args.init_checkpoint_dir is not None:
        init_checkpoint_path = args.init_checkpoint_dir / f"{run_name}.pt"
    if init_checkpoint_path is not None:
        init_checkpoint_path = init_checkpoint_path.resolve()
        checkpoint = torch.load(init_checkpoint_path, map_location="cpu")
        load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=not args.init_allow_partial)
        print(f"[{run_name}] initialized weights from {init_checkpoint_path}")
        if args.init_allow_partial:
            print(f"[{run_name}] warm-start missing keys: {list(load_result.missing_keys)}")
            print(f"[{run_name}] warm-start unexpected keys: {list(load_result.unexpected_keys)}")
    model.set_hla_embeddings(train_dataset.hla_store.embeddings)
    criterion = WeightedBCEWithContrastiveLoss(
        pos_weight=pos_weight,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        window_entropy_weight=args.window_entropy_weight,
        window_margin_weight=args.window_margin_weight,
        window_margin=args.window_margin,
        window_loss_temperature=args.window_loss_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"[{run_name}] train positives={positives} negatives={negatives} pos_weight={pos_weight:.4f}")
    print(f"[{run_name}] trainable parameters={count_trainable_parameters(model):,}")

    train_loader = loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = loader(val_dataset, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    best_score = -float("inf")
    best_metrics = None
    wait = 0
    history = []
    checkpoint_path = args.output_dir / f"{run_name}.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, use_amp, optimizer=optimizer, scaler=scaler, threshold=args.threshold
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, use_amp, optimizer=None, threshold=args.threshold
        )
        score = metric_score(val_metrics, args.best_metric)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"[{run_name}] epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_auc={train_metrics['auc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_aupr={val_metrics['aupr']:.4f} val_mcc={val_metrics['mcc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_went={val_metrics['window_entropy_loss']:.4f} "
            f"val_wmargin={val_metrics['window_margin_loss']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            wait = 0
            save_checkpoint(checkpoint_path, model, optimizer, args, epoch, val_metrics, pos_weight)
            print(f"[{run_name}] saved best checkpoint to {checkpoint_path}")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[{run_name}] early stopping at epoch {epoch}")
                break

    summary_path = args.output_dir / f"{run_name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_name": run_name,
                "best_metric": args.best_metric,
                "best_score": best_score,
                "best_metrics": best_metrics,
                "train_positive_count": positives,
                "train_negative_count": negatives,
                "pos_weight": pos_weight,
                "history": history,
            },
            handle,
            indent=2,
        )


def cv_train(args, hla_store):
    folds = parse_folds(args.folds)
    cv_dir = args.root / "data" / "benchmark_cv_seed1240"
    benchmark_cache = distilled_cache_prefix_for_split(args.root, "benchmark", max_length=args.cache_max_length)
    for fold in folds:
        train_dataset = PMHCIIEmbeddingDataset(cv_dir / f"train_fold_{fold}.csv", benchmark_cache, hla_store)
        val_dataset = PMHCIIEmbeddingDataset(cv_dir / f"val_fold_{fold}.csv", benchmark_cache, hla_store)
        train_counts = (train_dataset.positive_count, train_dataset.negative_count)
        train_one_run(args, train_dataset, val_dataset, f"fold{fold}", train_counts)


def single_train(args, hla_store):
    benchmark_cache = distilled_cache_prefix_for_split(args.root, "benchmark", max_length=args.cache_max_length)
    dataset = PMHCIIEmbeddingDataset(args.root / "data" / "benchmark.csv", benchmark_cache, hla_store)
    indices = np.arange(len(dataset))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(len(indices) * args.val_fraction)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_counts = subset_counts(dataset, train_indices)
    train_one_run(args, Subset(dataset, train_indices), Subset(dataset, val_indices), "single", train_counts)


def main():
    args = parse_args()
    args.root = args.root.resolve()
    args.output_dir = args.output_dir.resolve()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    hla_store = build_hla_store(args.root)
    print(f"HLA ESM2 store: {len(hla_store.alleles)} alleles, length={hla_store.sequence_length}, dim={hla_store.embedding_dim}")

    if args.mode == "cv_train":
        cv_train(args, hla_store)
    else:
        single_train(args, hla_store)


if __name__ == "__main__":
    main()
