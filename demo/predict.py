#!/usr/bin/env python3
"""Run the protective-antigen demo with one explicit classifier checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "Downstream" / "protective_antigen"
sys.path.insert(0, str(TASK_DIR))

from feature_extractor import (  # noqa: E402
    get_model_and_extract_func,
    resolve_hf_extract_dtype,
)
from protective_antigen_test import evaluate_dataset, write_single_metrics  # noqa: E402
from protective_antigen_utils import (  # noqa: E402
    AntigenDataset,
    read_labeled_table,
    setup_seed,
)


RUN_LABEL = "cluster_aware_fold_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AntigenLM protective-antigen demo")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--extract-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=22)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"AntigenLM model: {args.model_dir}")
    print(f"Classifier checkpoint: {args.classifier}")
    print(f"Input: {args.input}")

    frame = read_labeled_table(str(args.input), with_id=True)
    extract_embeddings, embedding_dim = get_model_and_extract_func(
        "AntigenLM",
        str(args.model_dir),
        device,
        hf_add_special_tokens=True,
        hf_extract_batch_size=args.extract_batch_size,
        hf_autocast_dtype=resolve_hf_extract_dtype("auto", device),
    )
    metrics = evaluate_dataset(
        eval_dataset=AntigenDataset(
            frame["sequence"],
            frame["label"],
            ids=frame["ID"],
        ),
        extract_emb_func=extract_embeddings,
        emb_dim=embedding_dim,
        model_path=str(args.classifier),
        output_dir=str(args.output_dir),
        run_label=RUN_LABEL,
        split_name="demo",
        batch_size=args.batch_size,
        device=device,
        threshold=args.threshold,
        model_type="AntigenLM",
    )
    write_single_metrics(
        metrics,
        str(args.output_dir / f"AntigenLM_{RUN_LABEL}_metrics.csv"),
    )


if __name__ == "__main__":
    main()
