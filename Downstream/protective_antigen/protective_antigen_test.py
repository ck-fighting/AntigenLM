import argparse
import os

import torch
from torch.utils.data import DataLoader

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
    binary_metrics,
    discover_cv_split_files,
    parse_subset_list,
    prediction_labels,
    print_split_counts,
    read_labeled_table,
    resolve_cv_datasets,
    sanitize_name,
    setup_seed,
)


def collate_emb_batch(batch):
    ids, x, y = zip(*batch)
    return list(ids), torch.stack(x, dim=0), torch.as_tensor(y, dtype=torch.long)


def load_model(model_path, emb_dim, device, hidden_dim=None, dropout=0.25):
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    if hidden_dim is None:
        hidden_dim = int(state_dict["token_proj.0.weight"].shape[0])

    model_clf = SoluModel(seq_len=512, in_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model_clf.load_state_dict(state_dict, strict=False)
    model_clf.eval()
    return model_clf


def evaluate_dataset(
    eval_dataset,
    extract_emb_func,
    emb_dim,
    model_path,
    output_dir,
    run_label,
    split_name="test",
    batch_size=32,
    device="cuda",
    threshold=0.5,
    model_type="AntigenLM",
    hidden_dim=None,
    dropout=0.25,
):
    ids, eval_sequences, eval_labels = eval_dataset.get_data()
    print(f"Extracting {split_name} embeddings ...")
    eval_embeddings = extract_emb_func(eval_sequences)
    eval_loader = DataLoader(
        list(zip(ids, eval_embeddings, eval_labels)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_emb_batch,
        pin_memory=_is_cuda(device),
    )

    model_clf = load_model(model_path, emb_dim, device, hidden_dim=hidden_dim, dropout=dropout)

    eval_ids, eval_trues, eval_probs = [], [], []
    with torch.no_grad():
        for batch_ids, x_batch, y_batch in eval_loader:
            logits, _ = model_clf(x_batch.to(device, non_blocking=True))
            probs = torch.sigmoid(logits).squeeze(-1)
            eval_ids.extend(batch_ids)
            eval_trues.extend(y_batch.tolist())
            eval_probs.extend(probs.cpu().tolist())

    metrics = binary_metrics(eval_trues, eval_probs, threshold)
    eval_preds = prediction_labels(eval_probs, threshold)

    print(
        f"\n[{run_label} EVAL] "
        f"AUC: {metrics['auc']:.4f} | AUPR: {metrics['aupr']:.4f} | "
        f"Acc: {metrics['acc']:.4f} | P: {metrics['precision']:.4f} | "
        f"R: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | MCC: {metrics['mcc']:.4f}"
    )

    os.makedirs(output_dir, exist_ok=True)
    result_path = P(output_dir, f"{model_type}_{sanitize_name(run_label)}_pred_results.csv")
    import pandas as pd

    pd.DataFrame(
        {
            "id": eval_ids,
            "y_true": eval_trues,
            "y_pred": eval_preds,
            "y_score": eval_probs,
        }
    ).to_csv(result_path, index=False)
    print(f"Predictions saved to {result_path}")

    return {"fold": run_label, **metrics}


def resolve_cv_checkpoint(weights_dir, run_name, fold_idx, seed, embed_backend):
    checkpoint_path = P(weights_dir, run_name, f"fold_{fold_idx}_seed{seed}_{embed_backend}.pt")
    return checkpoint_path if os.path.exists(checkpoint_path) else None, checkpoint_path


def resolve_pldgl_checkpoint(weights_dir, subset, seed, embed_backend):
    checkpoint_name = f"PLDGL_{subset}_seed{seed}_{embed_backend}.pt"
    checkpoint_path = P(weights_dir, "PLDGL", checkpoint_name)
    return checkpoint_path if os.path.exists(checkpoint_path) else None, checkpoint_path


def write_metrics(metrics, out_csv):
    if not metrics:
        return
    import pandas as pd

    metrics_df = pd.DataFrame(metrics)
    avg_row, sd_row = {"fold": "avg"}, {"fold": "sd"}
    for col in ["auc", "aupr", "acc", "precision", "recall", "f1", "mcc"]:
        avg_row[col] = float(metrics_df[col].mean())
        sd_row[col] = float(metrics_df[col].std(ddof=0)) if len(metrics_df) > 1 else 0.0
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_row, sd_row])], ignore_index=True)
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nMetrics saved to: {out_csv}")


def write_single_metrics(metrics, out_csv):
    if not metrics:
        return
    import pandas as pd

    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    print(f"\nMetrics saved to: {out_csv}")


def evaluate_cv(args, extract_emb_func, emb_dim, device):
    print(f"CV data root: {args.data_dir}")
    print(f"Weights directory: {args.weights_dir}")

    split_names = ["val", "test"] if args.eval_split == "both" else [args.eval_split]
    for cv_dir, dataset_label in resolve_cv_datasets(args.data_dir, args.dataset_name):
        run_name = sanitize_name(dataset_label)
        run_out_dir = P(args.out_dir, run_name)
        os.makedirs(run_out_dir, exist_ok=True)

        print(f"CV dataset: {dataset_label}")
        print(f"CV directory: {cv_dir}")
        print(f"Output directory: {run_out_dir}")

        for split_name in split_names:
            all_metrics = []
            for fold_idx, eval_csv in discover_cv_split_files(cv_dir, split_name, args.folds):
                if not os.path.exists(eval_csv):
                    raise FileNotFoundError(f"{split_name}_fold_{fold_idx}.csv not found: {eval_csv}")

                ckpt_path, expected = resolve_cv_checkpoint(args.weights_dir, run_name, fold_idx, args.seed, args.embed_backend)
                if args.model_path:
                    ckpt_path = args.model_path
                if not ckpt_path:
                    print(f"[Skip] Weights not found. Expected path: {expected}")
                    continue

                eval_df = read_labeled_table(eval_csv, with_id=True)
                print(f"========== {dataset_label} Fold {fold_idx} {split_name.upper()} ==========")
                print(f"{split_name.title()} CSV: {eval_csv}")
                print(f"Checkpoint: {ckpt_path}")
                print_split_counts(f"{split_name.title()} set", eval_df)
                run_label = f"fold_{fold_idx}" if split_name == "test" else f"{split_name}_fold_{fold_idx}"
                fold_metrics = evaluate_dataset(
                    eval_dataset=AntigenDataset(
                        eval_df["sequence"],
                        eval_df["label"],
                        ids=eval_df["ID"],
                    ),
                    extract_emb_func=extract_emb_func,
                    emb_dim=emb_dim,
                    model_path=ckpt_path,
                    output_dir=run_out_dir,
                    run_label=run_label,
                    split_name=split_name,
                    batch_size=args.batch_size,
                    device=device,
                    threshold=args.threshold,
                    model_type=args.embed_backend,
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                )
                fold_metrics = {"split": split_name, **fold_metrics}
                all_metrics.append(fold_metrics)

            write_metrics(all_metrics, P(run_out_dir, f"{args.embed_backend}_{run_name}_{split_name}_metrics.csv"))


def evaluate_pldgl(args, extract_emb_func, emb_dim, device):
    data_dir = os.path.abspath(args.pldgl_dir or args.data_dir)
    run_out_dir = P(args.out_dir, "PLDGL")
    os.makedirs(run_out_dir, exist_ok=True)

    print(f"PLDGL data directory: {data_dir}")
    print(f"Weights directory: {args.weights_dir}")
    print(f"Output directory: {run_out_dir}")

    for subset in parse_subset_list(args.subset, args.run_all_pldgl_subsets):
        test_path = args.test_file or P(data_dir, f"test_set_{subset}.xlsx")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"PLDGL test file not found: {test_path}")

        ckpt_path, expected = resolve_pldgl_checkpoint(args.weights_dir, subset, args.seed, args.embed_backend)
        if args.model_path:
            ckpt_path = args.model_path
        if not ckpt_path:
            print(f"[Skip] Weights not found for PLDGL {subset}. Expected path: {expected}")
            continue

        test_df = read_labeled_table(test_path, with_id=True)
        print(f"========== PLDGL {subset} ==========")
        print(f"Test file: {test_path}")
        print(f"Checkpoint: {ckpt_path}")
        print_split_counts("Test set", test_df)
        metrics = evaluate_dataset(
            eval_dataset=AntigenDataset(test_df["sequence"], test_df["label"], ids=test_df["ID"]),
            extract_emb_func=extract_emb_func,
            emb_dim=emb_dim,
            model_path=ckpt_path,
            output_dir=run_out_dir,
            run_label=f"PLDGL_{subset}",
            split_name="test",
            batch_size=args.batch_size,
            device=device,
            threshold=args.threshold,
            model_type=args.embed_backend,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        write_single_metrics(
            metrics,
            P(run_out_dir, f"{args.embed_backend}_PLDGL_{subset}_test_metrics.csv"),
        )


def main():
    p = argparse.ArgumentParser(description="Eval for Protective Antigen Classification")
    p.add_argument(
        "--mode",
        type=str,
        choices=["cv", "independent"],
        default="cv",
        help="CV evaluates the default split_fold_*.csv files; Independent evaluates the PLDGL route.",
    )
    p.add_argument("--data_dir", type=str, default=None, help="CV data root; defaults to ./data")
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--folds", type=str, default=None)
    p.add_argument("--weights_dir", type=str, default="../trained_model/protective_antigen")
    p.add_argument("--model_path", type=str, default=None, help="Evaluate one explicit checkpoint")
    p.add_argument("--out_dir", type=str, default="../result/protective_antigen")
    p.add_argument(
        "--eval_split",
        type=str,
        choices=["test", "val", "both"],
        default="test",
        help="CV split to evaluate. Use both to write val and test predictions with the same checkpoints.",
    )
    p.add_argument(
        "--embed_backend",
        type=str,
        choices=EMBED_BACKENDS,
        default="AntigenLM",
    )
    p.add_argument("--backend_path", type=str, default=None)
    p.add_argument("--hf_extract_batch_size", type=int, default=8)
    p.add_argument("--hf_extract_dtype", type=str, choices=["none", "fp32", "auto", "bf16", "fp16"], default="none")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--hidden_dim", type=int, default=None, help="Classifier hidden dim; inferred from checkpoint by default.")
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument(
        "--hf_no_special_tokens",
        action="store_true",
        help="For AntigenLM/HuggingFace backends, encode residue tokens without CLS/SEP.",
    )
    p.add_argument("--seed", type=int, default=22)

    p.add_argument("--pldgl", action="store_true", help="Deprecated alias for --mode Independent")
    p.add_argument("--pldgl_dir", type=str, default=None)
    p.add_argument("--subset", type=str, default="All")
    p.add_argument("--run_all_pldgl_subsets", action="store_true")
    p.add_argument("--test_file", type=str, default=None, help="Optional explicit csv/xlsx test file")
    args = p.parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "Independent" if args.mode.lower() == "independent" or args.pldgl else "CV"
    if mode == "CV":
        args.data_dir = os.path.abspath(args.data_dir or DEFAULT_CV_DATA_DIR)
    else:
        args.pldgl_dir = os.path.abspath(args.pldgl_dir or args.data_dir or DEFAULT_PLDGL_DATA_DIR)

    print(f"Mode: {mode}")
    print(f"Model: {args.embed_backend} | Device: {device}")
    resolved_backend_path = resolve_backend_path(args.embed_backend, args.backend_path)
    print(f"Backend path: {resolved_backend_path or args.embed_backend}")
    print(f"Threshold: {args.threshold}")
    print(f"Classifier hidden_dim: {args.hidden_dim or 'auto'} | Dropout: {args.dropout}")
    print(f"HF add special tokens: {not args.hf_no_special_tokens}")
    print(f"HF extract batch size: {args.hf_extract_batch_size} | dtype: {args.hf_extract_dtype}")
    extract_emb_func, emb_dim = get_model_and_extract_func(
        args.embed_backend,
        resolved_backend_path,
        device,
        hf_add_special_tokens=not args.hf_no_special_tokens,
        hf_extract_batch_size=args.hf_extract_batch_size,
        hf_autocast_dtype=resolve_hf_extract_dtype(args.hf_extract_dtype, device),
    )

    if mode == "Independent":
        evaluate_pldgl(args, extract_emb_func, emb_dim, device)
    else:
        print(f"CV evaluation split: {args.eval_split}")
        evaluate_cv(args, extract_emb_func, emb_dim, device)


if __name__ == "__main__":
    main()
