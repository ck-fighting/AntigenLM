import os
import argparse
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, auc, accuracy_score, f1_score,
    precision_recall_curve, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef
)
from tqdm import tqdm

from TCR_model import Mymodel_TCR, vocab, tcr_max_len
from feature_extractors import (
    DEFAULT_ESM2_MODEL_PATH,
    DEFAULT_ESMC_MODEL_PATH,
    antigenLM_extract,
    embedding_model_cache_id,
    extract_esm2_embeddings,
    extract_esmc_embeddings,
    infer_peptide_embedding_dim,
    load_esmc_model,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
P = os.path.join


def infer_cv_dir_from_weights(weights_dir):
    parts = set(os.path.normpath(weights_dir).split(os.sep))
    if "CMA" in parts:
        return "./data/CMA_5fold_splits"
    return "./data/Seen_5fold_splits"


def infer_emb_cache_dir(cv_dir):
    parts = {part.lower() for part in os.path.normpath(cv_dir).split(os.sep)}
    if any("cma" in part for part in parts):
        return "./data_cached_cma_5fold"
    return "./data_cached_seen_5fold"


def setup_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def remap_state_dict(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace("encoder_H.", "encoder_T.")] = v
    return new_sd


def load_checkpoint_state(path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    return remap_state_dict(obj)


def binarize(probs, threshold):
    return [1 if p >= threshold else 0 for p in probs]


def peptide_list_cache_digest(peptides):
    hasher = hashlib.sha1()
    for peptide in peptides:
        hasher.update(str(peptide).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:12]


def performance(y_true, y_prob, y_bin):
    accuracy = accuracy_score(y_true, y_bin)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_bin, zero_division=0)
    recall = recall_score(y_true, y_bin, zero_division=0)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_bin)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")

    try:
        prec, reca, _ = precision_recall_curve(y_true, y_prob)
        aupr = auc(reca, prec)
    except Exception:
        aupr = float("nan")

    print(f"tn={tn}, fp={fp}, fn={fn}, tp={tp}")
    print(f"y_pred: 0={Counter(y_bin)[0]} | 1={Counter(y_bin)[1]}")
    print(f"y_true: 0={Counter(y_true)[0]} | 1={Counter(y_true)[1]}")
    print(f"auc={roc_auc:.4f}|sensitivity={sensitivity:.4f}|specificity={specificity:.4f}|acc={accuracy:.4f}|mcc={mcc:.4f}")
    print(f"precision={precision:.4f}|recall={recall:.4f}|f1={f1:.4f}|aupr={aupr:.4f}")

    return {
        "auc": roc_auc,
        "accuracy": accuracy,
        "mcc": mcc,
        "f1": f1,
        "aupr": aupr,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Recall": recall,
    }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    if "Peptide" in df.columns:
        col_map["Peptide"] = "antigen"
    if "CDR3" in df.columns:
        col_map["CDR3"] = "TCR"
    if "Label" in df.columns:
        col_map["Label"] = "label"
    df = df.rename(columns=col_map)
    need = {"antigen", "TCR", "label"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns: {need}; actual: {set(df.columns)}")
    df["antigen"] = df["antigen"].astype(str).str.strip()
    df["TCR"] = df["TCR"].astype(str).str.strip()
    df = df[(df["antigen"] != "") & (df["TCR"] != "")].reset_index(drop=True)
    return df[["antigen", "TCR", "label"]].copy()


def _extract_pep_embeddings(pep_list, emb_cfg, device, pep_max_len):
    backend = emb_cfg["backend"]
    if backend == "AntigenLM":
        if not emb_cfg.get("AntigenLM_path"):
            raise ValueError("AntigenLM_path cannot be empty")
        return antigenLM_extract(
            pep_list,
            model_name_or_path=emb_cfg["AntigenLM_path"],
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            max_len=pep_max_len,
        )
    if backend == "esm2":
        return extract_esm2_embeddings(
            pep_list,
            model_name=emb_cfg.get("esm2_model_name", DEFAULT_ESM2_MODEL_PATH),
            device=device,
            max_len=pep_max_len,
        )
    if backend == "esmc":
        client = load_esmc_model(emb_cfg.get("esmc_model_name", DEFAULT_ESMC_MODEL_PATH), device=device)
        return extract_esmc_embeddings(
            pep_list,
            client=client,
            device=device,
            batch_size=512,
            max_len=pep_max_len,
            model_max_len=256,
        )
    raise ValueError(f"Unknown embedding backend: {backend}")


def data_process_tcr(
    data,
    type_tag,
    seed,
    device,
    emb_cache_dir,
    pep_max_len=15,
    tcr_max_len_=34,
    emb_cfg=None,
    use_cache=True,
):
    if emb_cfg is None:
        raise ValueError("emb_cfg cannot be None")
    data = normalize_columns(data)
    pep_list = data.antigen.tolist()

    os.makedirs(emb_cache_dir, exist_ok=True)
    if emb_cfg["backend"] == "AntigenLM":
        model_id = embedding_model_cache_id(emb_cfg["backend"], emb_cfg["AntigenLM_path"])
    elif emb_cfg["backend"] == "esm2":
        model_id = embedding_model_cache_id(emb_cfg["backend"], emb_cfg.get("esm2_model_name", DEFAULT_ESM2_MODEL_PATH))
    else:
        model_id = embedding_model_cache_id(emb_cfg["backend"], emb_cfg.get("esmc_model_name", DEFAULT_ESMC_MODEL_PATH))
    expected_dim = infer_peptide_embedding_dim(
        emb_cfg["backend"],
        emb_cfg.get("AntigenLM_path", ""),
        emb_cfg.get("esm2_model_name", DEFAULT_ESM2_MODEL_PATH),
        emb_cfg.get("esmc_model_name", DEFAULT_ESMC_MODEL_PATH),
    )
    expected_shape = (len(pep_list), pep_max_len, expected_dim)
    data_id = peptide_list_cache_digest(pep_list)
    cache_name = f"cached_pep_embeddings_{type_tag}_{data_id}_{seed}_{emb_cfg['backend']}_{model_id}_L{pep_max_len}.pt"
    emb_cache_path = P(emb_cache_dir, cache_name)

    if use_cache and os.path.isfile(emb_cache_path):
        print(f"[Cache] Use peptide embeddings: {emb_cache_path}")
        pep_embeddings = torch.load(emb_cache_path, map_location="cpu")
        if isinstance(pep_embeddings, np.ndarray):
            pep_embeddings = torch.from_numpy(pep_embeddings)
        if tuple(pep_embeddings.shape) != expected_shape:
            print(f"[Cache] Shape mismatch {tuple(pep_embeddings.shape)} != {expected_shape}; recomputing.")
            pep_embeddings = _extract_pep_embeddings(pep_list, emb_cfg, device, pep_max_len)
            torch.save(pep_embeddings.detach().cpu(), emb_cache_path)
    else:
        print(f"[Embed] Extracting {len(pep_list)} peptides via {emb_cfg['backend']} ...")
        pep_embeddings = _extract_pep_embeddings(pep_list, emb_cfg, device, pep_max_len)
        torch.save(pep_embeddings.detach().cpu(), emb_cache_path)
        print(f"[Cache] Saved peptide embeddings -> {emb_cache_path}")

    labels, pep_raw, tcr_raw = [], [], []
    for pep_seq, tcr_seq, label in zip(data.antigen, data.TCR, data.label):
        pep_raw.append(pep_seq)
        tcr_raw.append(tcr_seq)
        labels.append(int(label))

    tcr_token_ids = []
    for tcr_seq in tcr_raw:
        tcr_token_ids.append([vocab.get(n, vocab.get("-", 0)) for n in tcr_seq.ljust(tcr_max_len_, "-")])
    tcr_inputs = torch.LongTensor(tcr_token_ids)

    return pep_embeddings, tcr_inputs, torch.LongTensor(labels), pep_raw, tcr_raw


class EvalDataSet_TCR(Dataset):
    def __init__(self, pep_embeds, tcr_idx_tensor, labels_tensor, pep_raw, tcr_raw):
        self.pep_embeds = pep_embeds
        self.tcr_idx = tcr_idx_tensor
        self.labels = labels_tensor
        self.pep_raw = pep_raw
        self.tcr_raw = tcr_raw
        assert len(self.labels) == len(self.tcr_raw) == len(self.pep_raw) == self.tcr_idx.size(0)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, i):
        return self.pep_embeds[i], self.tcr_idx[i], self.labels[i], self.pep_raw[i], self.tcr_raw[i]


def build_loader_from_df(
    df_sub,
    type_tag,
    batch_size,
    seed,
    device,
    emb_cache_dir,
    emb_cfg,
    num_workers,
):
    pep_inputs, tcr_inputs, labels, pep_raw, tcr_raw = data_process_tcr(
        df_sub,
        type_tag,
        seed,
        device,
        emb_cache_dir,
        pep_max_len=15,
        tcr_max_len_=tcr_max_len,
        emb_cfg=emb_cfg,
    )
    ds = EvalDataSet_TCR(pep_inputs, tcr_inputs, labels, pep_raw, tcr_raw)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return loader


@torch.no_grad()
def eval_on_loader(model, loader, device, threshold):
    model.eval()
    y_true, y_prob, y_bin, pep_list, tcr_list = [], [], [], [], []

    for anti_inputs, tcr_inputs, labels, pep_strs, tcr_strs in tqdm(loader, colour="blue"):
        logits, _, _ = model(anti_inputs.to(device), tcr_inputs.to(device))
        probs = torch.sigmoid(logits.view(-1)).cpu().numpy()

        y_prob.extend(probs.tolist())
        y_true.extend(labels.cpu().numpy().tolist())
        y_bin.extend((probs >= threshold).astype(np.int32).tolist())
        pep_list.extend(list(pep_strs))
        tcr_list.extend(list(tcr_strs))

    return y_true, y_prob, y_bin, pep_list, tcr_list


def evaluate_dataset(model, loader, device, threshold, fold, dataset_name, out_dir, embed_backend):
    y_true, y_prob, y_bin, pep_strs, tcr_strs = eval_on_loader(model, loader, device, threshold)
    pred_df = pd.DataFrame({
        "fold": [fold] * len(y_true),
        "TCR": tcr_strs,
        "antigen": pep_strs,
        "label_true": y_true,
        "label_pred": y_bin,
        "label_prob": y_prob,
    })
    pred_path = P(out_dir, f"{embed_backend}_{dataset_name}_pred_results_fold{fold}.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to: {pred_path}")

    print(f"\n===== Fold {fold} | {dataset_name} =====")
    metrics = performance(y_true, y_prob, y_bin)
    metrics.update({"fold": f"fold{fold}", "dataset": dataset_name})
    return pred_df, metrics


def add_avg_sd_rows(metrics_rows):
    df = pd.DataFrame(metrics_rows)
    value_cols = ["auc", "accuracy", "mcc", "f1", "aupr", "Sensitivity", "Specificity", "Precision", "Recall"]
    avg_row = {"fold": "avg", "dataset": df["dataset"].iloc[0]}
    sd_row = {"fold": "sd", "dataset": df["dataset"].iloc[0]}
    for col in value_cols:
        avg_row[col] = df[col].mean()
        sd_row[col] = df[col].std(ddof=1) if len(df) >= 2 else 0.0
    return pd.concat([df, pd.DataFrame([avg_row, sd_row])], ignore_index=True)


def init_external_eval_dataset(csv_path, dataset_name, args, device, emb_cfg):
    if not csv_path:
        return None
    if not os.path.exists(csv_path):
        print(f"[Skip] {dataset_name} CSV not found: {csv_path}")
        return None

    df = normalize_columns(pd.read_csv(csv_path))
    loader = build_loader_from_df(
        df,
        dataset_name,
        args.batch_size,
        args.seed,
        device,
        args.emb_cache_dir,
        emb_cfg,
        args.num_workers,
    )
    print(f"[External] {dataset_name}: {csv_path} | rows={len(df)} | peptides={df['antigen'].nunique()}")
    return {
        "name": dataset_name,
        "csv_path": csv_path,
        "loader": loader,
        "metrics": [],
        "fold_probs": [],
        "true": None,
        "pep": None,
        "tcr": None,
    }


def save_external_eval_results(external, args):
    dataset_name = external["name"]
    metrics_rows = external["metrics"]
    fold_probs = external["fold_probs"]

    if metrics_rows:
        metrics_df = add_avg_sd_rows(metrics_rows)
        metrics_path = P(args.out_dir, f"{args.embed_backend}_{dataset_name}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"{dataset_name} metrics saved to: {metrics_path}")

    if len(fold_probs) >= 2:
        avg_prob = np.mean(fold_probs, axis=0).tolist()
        avg_bin = binarize(avg_prob, args.threshold)
        print(f"\n===== {dataset_name} | 5-fold probability average =====")
        avg_metrics = performance(external["true"], avg_prob, avg_bin)
        avg_pred = pd.DataFrame({
            "TCR": external["tcr"],
            "antigen": external["pep"],
            "label_true": external["true"],
            "label_pred": avg_bin,
            "label_prob": avg_prob,
        })
        avg_pred_path = P(args.out_dir, f"{args.embed_backend}_{dataset_name}_pred_results_5fold_avg.csv")
        avg_pred.to_csv(avg_pred_path, index=False)
        print(f"{dataset_name} 5-fold averaged predictions saved to: {avg_pred_path}")

        avg_row = {"dataset": dataset_name, "fold": "5fold_avg"}
        avg_row.update(avg_metrics)
        avg_metrics_path = P(args.out_dir, f"{args.embed_backend}_{dataset_name}_5fold_avg_metrics.csv")
        pd.DataFrame([avg_row]).to_csv(avg_metrics_path, index=False)
        print(f"{dataset_name} 5-fold averaged metrics saved to: {avg_metrics_path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate fold checkpoints on fold test sets and external datasets.")
    ap.add_argument("--cv_dir", default=None, help="Fold split directory. If omitted, inferred from --weights_dir.")
    ap.add_argument("--fold_prefix", default="fold_")
    ap.add_argument("--num_folds", type=int, default=5)
    ap.add_argument("--independent_csv", default="", help="Optional independent CSV, such as ./data/Covid_set.csv.")
    ap.add_argument("--unseen_csv", default="", help="Optional Unseen CSV, such as ./data/Unseen.csv.")
    ap.add_argument("--weights_dir", default="../trained_model/pTCR3/Seen")
    ap.add_argument("--out_dir", default="../result/pTCR3/AntigenLM_Seen")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--emb_cache_dir", default=None, help="Peptide embedding cache directory. If omitted, inferred from --cv_dir.")
    ap.add_argument("--embed_backend", choices=["AntigenLM", "esm2", "esmc"], default="AntigenLM")
    ap.add_argument("--AntigenLM_path", default="../../LLM/AntigenLM")
    ap.add_argument("--esm2_model_name", default=DEFAULT_ESM2_MODEL_PATH)
    ap.add_argument("--esmc_model_name", default=DEFAULT_ESMC_MODEL_PATH)
    ap.add_argument("--pep_input_norm", action="store_true", help="Layer-normalize each residue embedding before peptide projection.")
    ap.add_argument("--pep_input_scale", type=float, default=1.0, help="Scale peptide embeddings after optional input normalization.")
    args = ap.parse_args()
    args.cv_dir = args.cv_dir or infer_cv_dir_from_weights(args.weights_dir)
    args.emb_cache_dir = args.emb_cache_dir or infer_emb_cache_dir(args.cv_dir)

    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    emb_cfg = {
        "backend": args.embed_backend,
        "AntigenLM_path": args.AntigenLM_path,
        "esm2_model_name": args.esm2_model_name,
        "esmc_model_name": args.esmc_model_name,
    }
    pep_dim = infer_peptide_embedding_dim(
        args.embed_backend,
        args.AntigenLM_path,
        args.esm2_model_name,
        args.esmc_model_name,
    )
    print(f"[Model] Peptide embedding dim = {pep_dim}")
    model = Mymodel_TCR(
        pep_dim=pep_dim,
        pep_input_norm=args.pep_input_norm,
        pep_input_scale=args.pep_input_scale,
    ).to(device)

    cv_metrics, cv_pred_frames = [], []
    external_datasets = []
    for external in [
        init_external_eval_dataset(args.independent_csv, "covid_set", args, device, emb_cfg),
        init_external_eval_dataset(args.unseen_csv, "unseen", args, device, emb_cfg),
    ]:
        if external is not None:
            external_datasets.append(external)

    if not external_datasets:
        print("[External] No external datasets will be evaluated.")

    for fold in range(1, args.num_folds + 1):
        ckpt = P(args.weights_dir, f"fold{fold}_seed{args.seed}_{args.embed_backend}.pt")
        if not os.path.exists(ckpt):
            print(f"[Skip] Weights not found: {ckpt}")
            continue

        model.load_state_dict(load_checkpoint_state(ckpt, device))

        test_csv = P(args.cv_dir, f"{args.fold_prefix}{fold}", "test.csv")
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"Missing fold test file: {test_csv}")

        test_df = normalize_columns(pd.read_csv(test_csv))
        test_loader = build_loader_from_df(
            test_df,
            f"fold{fold}_test",
            args.batch_size,
            args.seed,
            device,
            args.emb_cache_dir,
            emb_cfg,
            args.num_workers,
        )
        pred_df, metrics = evaluate_dataset(
            model,
            test_loader,
            device,
            args.threshold,
            fold,
            "cv_test",
            args.out_dir,
            args.embed_backend,
        )
        cv_pred_frames.append(pred_df)
        cv_metrics.append(metrics)

        for external in external_datasets:
            pred_df_external, metrics_external = evaluate_dataset(
                model,
                external["loader"],
                device,
                args.threshold,
                fold,
                external["name"],
                args.out_dir,
                args.embed_backend,
            )
            external["metrics"].append(metrics_external)
            external["fold_probs"].append(pred_df_external["label_prob"].to_numpy(dtype=np.float32))
            if external["true"] is None:
                external["true"] = pred_df_external["label_true"].tolist()
                external["pep"] = pred_df_external["antigen"].tolist()
                external["tcr"] = pred_df_external["TCR"].tolist()

    if cv_metrics:
        cv_metrics_df = add_avg_sd_rows(cv_metrics)
        cv_metrics_path = P(args.out_dir, f"{args.embed_backend}_cv_test_metrics.csv")
        cv_metrics_df.to_csv(cv_metrics_path, index=False)
        print(f"CV test metrics saved to: {cv_metrics_path}")

        cv_all_pred = pd.concat(cv_pred_frames, ignore_index=True)
        cv_all_pred_path = P(args.out_dir, f"{args.embed_backend}_cv_test_all_folds_pred_results.csv")
        cv_all_pred.to_csv(cv_all_pred_path, index=False)
        print(f"CV test all-fold predictions saved to: {cv_all_pred_path}")

        print("\n===== CV test | pooled fold test rows =====")
        pooled_metrics = performance(
            cv_all_pred["label_true"].tolist(),
            cv_all_pred["label_prob"].tolist(),
            cv_all_pred["label_pred"].tolist(),
        )
        pooled_row = {"dataset": "cv_test_pooled", "fold": "pooled"}
        pooled_row.update(pooled_metrics)
        pd.DataFrame([pooled_row]).to_csv(P(args.out_dir, f"{args.embed_backend}_cv_test_pooled_metrics.csv"), index=False)

    for external in external_datasets:
        save_external_eval_results(external, args)

    print("\n===== Done =====")


if __name__ == "__main__":
    main()
