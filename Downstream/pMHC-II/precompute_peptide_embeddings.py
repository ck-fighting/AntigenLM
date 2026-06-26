import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pmhc_data import distilled_cache_prefix_for_split, read_rows


CURRENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute AntigenLM peptide token embeddings into mmap/npz caches."
    )
    parser.add_argument("--root", type=Path, default=CURRENT_DIR)
    parser.add_argument("--split", choices=("benchmark", "warm", "cold", "all"), default="all")
    parser.add_argument("--csv-path", type=Path, default=None, help="Optional custom CSV path for one cache.")
    parser.add_argument("--output-prefix", type=Path, default=None, help="Optional custom output prefix for one cache.")
    parser.add_argument(
        "--peptide-model-path",
        type=Path,
        default=CURRENT_DIR.parent.parent / "LLM" / "AntigenLM",
    )
    parser.add_argument("--max-length", type=int, default=34, help="32 aa peptides plus BOS/EOS for ESMC tokenizer.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_csv_path(root, split):
    mapping = {
        "benchmark": root / "data" / "benchmark.csv",
        "warm": root / "data" / "warm-start-test.csv",
        "cold": root / "data" / "cold-start-test.csv",
    }
    return mapping[split]


def unique_peptides(rows):
    peptide_to_unique = {}
    unique = []
    row_to_unique = np.empty((len(rows),), dtype=np.int64)
    for row_index, row in enumerate(rows):
        peptide = row["peptide"]
        unique_index = peptide_to_unique.get(peptide)
        if unique_index is None:
            unique_index = len(unique)
            peptide_to_unique[peptide] = unique_index
            unique.append(peptide)
        row_to_unique[row_index] = unique_index
    return unique, row_to_unique


def load_antigenlm(model_path, device):
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Please install transformers before precomputing AntigenLM_distilled embeddings.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def hidden_dim_from_model(model):
    config = getattr(model, "config", None)
    for name in ("d_model", "hidden_size"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer hidden dimension from AntigenLM_distilled config.")


@torch.no_grad()
def precompute_one(args, csv_path, output_prefix):
    csv_path = Path(csv_path)
    output_prefix = Path(output_prefix)
    meta_path = output_prefix.with_suffix(".npz")
    mmap_path = output_prefix.with_suffix(".mmap")
    if not args.overwrite and (meta_path.exists() or mmap_path.exists()):
        raise FileExistsError(f"Cache already exists for prefix {output_prefix}; pass --overwrite to replace it.")

    rows = read_rows(csv_path)
    peptides, row_to_unique = unique_peptides(rows)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    tokenizer, model = load_antigenlm(args.peptide_model_path, device)
    hidden_dim = hidden_dim_from_model(model)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(peptides), args.max_length, hidden_dim)
    embeddings = np.memmap(mmap_path, mode="w+", dtype=args.dtype, shape=shape)
    attention_mask = np.zeros((len(peptides), args.max_length), dtype=np.uint8)

    print(f"CSV: {csv_path}")
    print(f"Unique peptides: {len(peptides):,} / rows: {len(rows):,}")
    print(f"Writing embeddings: {mmap_path} shape={shape} dtype={args.dtype}")

    for start in tqdm(range(0, len(peptides), args.batch_size), desc=output_prefix.name, dynamic_ncols=True):
        batch_peptides = peptides[start : start + args.batch_size]
        encoded = tokenizer(
            batch_peptides,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True)
        if outputs.hidden_states is None:
            raise RuntimeError("AntigenLM_distilled did not return hidden states.")
        tokens = outputs.hidden_states[-1].detach().cpu().numpy().astype(args.dtype, copy=False)
        end = start + tokens.shape[0]
        embeddings[start:end] = tokens
        attention_mask[start:end] = encoded["attention_mask"].detach().cpu().numpy().astype(np.uint8, copy=False)

    embeddings.flush()
    np.savez_compressed(
        meta_path,
        shape=np.asarray(shape, dtype=np.int64),
        dtype=np.asarray(args.dtype),
        row_to_unique=row_to_unique,
        attention_mask=attention_mask,
        max_length=np.asarray(args.max_length, dtype=np.int64),
        peptide_model_path=np.asarray(str(args.peptide_model_path)),
        csv_path=np.asarray(str(csv_path)),
    )
    print(f"Saved metadata: {meta_path}")


def main():
    args = parse_args()
    args.root = args.root.resolve()
    args.peptide_model_path = args.peptide_model_path.resolve()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.csv_path is not None or args.output_prefix is not None:
        if args.csv_path is None or args.output_prefix is None:
            raise ValueError("--csv-path and --output-prefix must be provided together.")
        precompute_one(args, args.csv_path.resolve(), args.output_prefix.resolve())
        return

    splits = ["benchmark", "warm", "cold"] if args.split == "all" else [args.split]
    for split in splits:
        precompute_one(
            args,
            split_csv_path(args.root, split),
            distilled_cache_prefix_for_split(args.root, split, max_length=args.max_length),
        )


if __name__ == "__main__":
    main()
