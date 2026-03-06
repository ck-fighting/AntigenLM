# eval.py
# 单CSV评测（逐fold评测、导出预测/指标/TSNE）

import os
import json
import hashlib
import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, auc, accuracy_score, f1_score,
    precision_recall_curve, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef
)
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# 你的工程内组件（需提供: vocab, Mymodel_HLA, antigenLM_extract, extract_esm2_embeddings, extract_esmc_embeddings）
from MHC_model import *
from feature_extractors import antigenLM_extract, extract_esm2_embeddings, extract_esmc_embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
P = os.path.join  # 路径拼接别名

# ------------------ 基础：随机种子/工具 ------------------
def setup_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def transfer(probs, thr): 
    return [1 if p >= thr else 0 for p in probs]

def performance(y_true, y_prob, y_bin):
    accuracy = accuracy_score(y_true=y_true, y_pred=y_bin)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn) if (tp+fn)>0 else 0.0
    specificity = tn / (tn + fp) if (tn+fp)>0 else 0.0
    precision = precision_score(y_true=y_true, y_pred=y_bin, zero_division=0)
    recall    = recall_score(y_true=y_true,  y_pred=y_bin, zero_division=0)
    f1        = f1_score(y_true=y_true,     y_pred=y_bin, zero_division=0)
    roc_auc   = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr      = auc(reca, prec)
    mcc       = matthews_corrcoef(y_true, y_bin)

    print(f'tn={tn}, fp={fp}, fn={fn}, tp={tp}')
    print(f'y_pred: 0={Counter(y_bin)[0]} | 1={Counter(y_bin)[1]}')
    print(f'y_true: 0={Counter(y_true)[0]} | 1={Counter(y_true)[1]}')
    print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(
        roc_auc, sensitivity, specificity, accuracy, mcc))
    print('precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}'.format(
        precision, recall, f1, aupr))
    return (roc_auc, accuracy, mcc, f1, aupr, sensitivity, specificity, precision, recall)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    if "Peptide" in df.columns: col_map["Peptide"] = "peptide"
    if "antigen" in df.columns: col_map["antigen"] = "peptide"
    if "HLA" in df.columns:     col_map["HLA"]     = "HLA"
    if "hla" in df.columns:     col_map["hla"]     = "HLA"
    if "Label" in df.columns:   col_map["Label"]   = "label"
    df = df.rename(columns=col_map)
    need = {"peptide","HLA","label"}
    if not need.issubset(df.columns):
        raise ValueError(f"缺少列：{need}；实际列：{set(df.columns)}")
    df["peptide"] = df["peptide"].astype(str).str.strip()
    df["HLA"]     = df["HLA"].astype(str).str.strip()
    return df[["peptide","HLA","label"]].copy()

def _checksum_of_list(str_list):
    h = hashlib.md5()
    for s in str_list:
        h.update(s.encode('utf-8', errors='ignore')); h.update(b'\n')
    return h.hexdigest()

def save_npz(feats, labels, out_path):
    feats  = np.asarray(feats, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    np.savez(out_path, feats=feats, labels=labels)
    print(f"📦 保存 t-SNE npz：{out_path}  (feats={feats.shape}, labels={labels.shape})")

# ------------------ 嵌入与数据管道 ------------------
def _extract_pep_embeddings(
    pep_list,
    emb_cfg,
    device,
    pep_max_len,
):
    backend = emb_cfg["backend"]
    if backend == "antigenLM":
        if not emb_cfg.get("antigenLM_path"):
            raise ValueError("antigenLM_path 不能为空")
        return antigenLM_extract(
            pep_list,
            model_name_or_path=emb_cfg["antigenLM_path"],
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            max_len=pep_max_len
        )
    if backend == "antigenLM_withoutSS":
        if not emb_cfg.get("antigenLM_withoutSS_path"):
            raise ValueError("antigenLM_withoutSS_path 不能为空")
        return antigenLM_extract(
            pep_list,
            model_name_or_path=emb_cfg["antigenLM_withoutSS_path"],
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            max_len=pep_max_len
        )
    if backend == "antigenLM_withoutSlidingwindow":
        if not emb_cfg.get("antigenLM_withoutSlidingwindow_path"):
            raise ValueError("antigenLM_withoutSlidingwindow_path 不能为空")
        return antigenLM_extract(
            pep_list,
            model_name_or_path=emb_cfg["antigenLM_withoutSlidingwindow_path"],
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            max_len=pep_max_len
        )
    if backend == "microLM":
        if not emb_cfg.get("microLM_path"):
            raise ValueError("microLM_path 不能为空")
        return antigenLM_extract(
            pep_list,
            model_name_or_path=emb_cfg["microLM_path"],
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            max_len=pep_max_len
        )
    if backend == "PathogLM":
        if not emb_cfg.get("pathogLM_path"):
            raise ValueError("pathogLM_path 不能为空")
        return antigenLM_extract(
            pep_list,
            model_name_or_path=emb_cfg["pathogLM_path"],
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            max_len=pep_max_len
        )
    elif backend == "esm2":
        return extract_esm2_embeddings(
            pep_list, model_name=emb_cfg.get("esm2_model_name", "esm2_t33_650M_UR50D"),
            device=device, max_len=pep_max_len
        )
    elif backend == "esmc":
        from esm.models.esmc import ESMC
        client = ESMC.from_pretrained(emb_cfg.get("esmc_model_name", "esmc_300m")).to(device)
        return extract_esmc_embeddings(
            pep_list, client=client, device=device,
            batch_size=512, max_len=15, model_max_len=256
        )
    else:
        raise ValueError(f"未知嵌入后端：{backend}")

def data_process_hla(
    data: pd.DataFrame,
    type_tag: str,
    seed: int,
    device: torch.device,
    emb_cache_dir: str,
    pep_max_len: int = 15,
    hla_max_len_: int = 34,
    emb_cfg=None,  # {"backend": "...", "antigenLM_path": "...", ...}
    use_cache: bool = True,
):
    if emb_cfg is None:
        raise ValueError("emb_cfg 不能为空")
    data = normalize_columns(data)
    # 过滤空串
    mask = (data["peptide"].str.len() > 0) & (data["HLA"].str.len() > 0)
    data = data.loc[mask].reset_index(drop=True)

    pep_list, pep_raw = [], []
    for pep in data.peptide:
        pep_raw.append(pep)
        pep_list.append(pep.ljust(hla_max_len_, '-'))  # 兼容你原逻辑

    os.makedirs(emb_cache_dir, exist_ok=True)
    cache_name = f"cached_pep_embeddings_{type_tag}_{seed}_{emb_cfg['backend']}.pt"
    emb_cache_path = P(emb_cache_dir, cache_name)

    if use_cache and os.path.isfile(emb_cache_path):
        print(f"[Cache] Use peptide embeddings: {emb_cache_path}")
        pep_embeddings = torch.load(emb_cache_path, map_location='cpu')
    else:
        print(f"[Embed] Extracting {len(pep_list)} peptides via {emb_cfg['backend']} ...")
        pep_embeddings = _extract_pep_embeddings(pep_list, emb_cfg, device, pep_max_len)
        torch.save(pep_embeddings.detach().cpu(), emb_cache_path)
        print(f"[Cache] Saved peptide embeddings -> {emb_cache_path}")

    # HLA索引 + 标签
    hla_inputs, labels, hla_raw = [], [], []
    for hla_seq, label in zip(data.HLA, data.label):
        hla_raw.append(hla_seq)
        idxs = [vocab.get(n, vocab.get('-', 0)) for n in hla_seq.ljust(hla_max_len_, '-')]
        hla_inputs.append(idxs)
        labels.append(int(label) if str(label).isdigit() else 0)

    return (
        pep_embeddings,
        torch.LongTensor(hla_inputs),
        torch.LongTensor(labels),
        pep_raw, hla_raw
    )

class EvalDataSet_HLA(Dataset):
    def __init__(self, pep_embeds, hla_idx_tensor, labels_tensor, pep_raw, hla_raw):
        self.pep_embeds = pep_embeds
        self.hla_idx = hla_idx_tensor
        self.labels = labels_tensor
        self.pep_raw = pep_raw
        self.hla_raw = hla_raw
        assert len(self.labels) == len(self.hla_raw) == len(self.pep_raw) == self.hla_idx.size(0)

    def __len__(self): 
        return self.labels.size(0)

    def __getitem__(self, i):
        return (self.pep_embeds[i], self.hla_idx[i], self.labels[i], self.pep_raw[i], self.hla_raw[i])

def build_loader_from_df(
    df_sub: pd.DataFrame, 
    type_tag: str, 
    batch_size: int, 
    seed: int, 
    device: torch.device,
    emb_cache_dir: str,
    emb_cfg,
):
    pep_inputs, hla_inputs, labels, pep_raw, hla_raw = data_process_hla(
        df_sub,
        type_tag,
        seed,
        device,
        emb_cache_dir,
        pep_max_len=15,
        hla_max_len_=hla_max_len,
        emb_cfg=emb_cfg,
    )
    ds = EvalDataSet_HLA(pep_inputs, hla_inputs, labels, pep_raw, hla_raw)
    shuffle = False
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True, drop_last=False)
    return loader, ds

# ------------------ 推理（不微调） ------------------
@torch.no_grad()
def eval_on_loader(model, loader, device, threshold):
    model.eval()
    y_true, y_prob, y_bin = [], [], []
    pep_list, hla_list, feat_chunks = [], [], []

    for anti_inputs, hla_inputs, labels, pep_strs, hla_strs in tqdm(loader, colour='blue'):
        anti_inputs = anti_inputs.to(device)
        hla_inputs  = hla_inputs.to(device)
        labels      = labels.to(device)

        logits, _, features = model(anti_inputs, hla_inputs)
        probs = torch.sigmoid(logits.view(-1)).cpu().numpy()

        y_prob.extend(probs.tolist())
        y_true.extend(labels.cpu().numpy().tolist())
        y_bin.extend((probs >= threshold).astype(np.int32).tolist())
        pep_list.extend(list(pep_strs))
        hla_list.extend(list(hla_strs))
        feat_chunks.append(features.detach().cpu().numpy())

    feats = np.concatenate(feat_chunks, axis=0) if feat_chunks else np.zeros((0, 1), np.float32)
    return y_true, y_prob, y_bin, pep_list, hla_list, feats

# ------------------ 主流程 ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="./data/micro_set.csv")

    # 通用
    ap.add_argument("--weights_dir", default="../trained_model/HLA_micro/Ablation")
    ap.add_argument("--out_dir", default="../Result/MHC/Ablation")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=22)

    # 嵌入缓存 & 后端
    ap.add_argument("--emb_cache_dir", default="./data_cached/Ablation")
    ap.add_argument(
        "--embed_backend",
        choices=[
            "antigenLM",
            "antigenLM_withoutSS",
            "antigenLM_withoutSlidingwindow",
            "microLM",
            "PathogLM",
            "esm2",
            "esmc",
        ],
        default="microLM",
    )
    ap.add_argument("--antigenLM_path", default="/home/dataset-local/AntigenLM/LLM/Result_antigenLM_300M_SS_2")
    ap.add_argument("--antigenLM_withoutSS_path", default="/home/dataset-local/AntigenLM/Other_LLM/AntigenLM_300M_withoutSS")
    ap.add_argument("--antigenLM_withoutSlidingwindow_path", default="/home/dataset-local/AntigenLM/Other_LLM/AntigenLM_300M_withoutSlidingwindow")
    ap.add_argument("--microLM_path", default="/home/dataset-local/AntigenLM/LLM/Result_microLM_300M/0406_034622_rank0")
    ap.add_argument("--pathogLM_path", default="/home/dataset-local/AntigenLM/LLM/Result_PathogLM_300M_SS")
    ap.add_argument("--esm2_model_name", default="esm2_t33_650M_UR50D")
    ap.add_argument("--esmc_model_name", default="esmc_300m")

    args = ap.parse_args()
    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = Mymodel_HLA().to(device)

    emb_cfg = {
        "backend": args.embed_backend,
        "antigenLM_path": args.antigenLM_path,
        "antigenLM_withoutSS_path": args.antigenLM_withoutSS_path,
        "antigenLM_withoutSlidingwindow_path": args.antigenLM_withoutSlidingwindow_path,
        "microLM_path": args.microLM_path,
        "pathogLM_path": args.pathogLM_path,
        "esm2_model_name": args.esm2_model_name,
        "esmc_model_name": args.esmc_model_name,
    }

    # 数据与命名后缀
    df_eval = pd.read_csv(args.data_csv)
    suffix = "micro"
    loader, _ = build_loader_from_df(
        df_eval,
        suffix,
        args.batch_size,
        args.seed,
        device,
        args.emb_cache_dir,
        emb_cfg,
    )

    fold_preds, all_y_true = [], None
    metrics_rows = []

    for fold in range(1, 6):
        ckpt = P(args.weights_dir, f"model_fold{fold}_seed22_{args.embed_backend}.pt")
        if not os.path.exists(ckpt):
            print(f"[Skip] 未找到权重：{ckpt}")
            continue

        base_state = torch.load(ckpt, map_location=device)
        model.load_state_dict(base_state)

        y_true, y_prob, y_bin, pep_strs, hla_strs, feats = eval_on_loader(model, loader, device, args.threshold)

        df_fold = pd.DataFrame({
            "fold": [fold] * len(y_true),
            "HLA": hla_strs,
            "peptide": pep_strs,
            "label_true": y_true,
            "label_pred": y_bin,
            "label_prob": y_prob,
        })
        df_fold.to_csv(P(args.out_dir, f"preds_fold{fold}_{suffix}_{args.embed_backend}.csv"), index=False)

        if all_y_true is None:
            all_y_true = y_true

        print(f"\n===== Fold {fold} | EVAL =====")
        m = performance(y_true, y_prob, y_bin)
        fold_preds.append(np.asarray(y_prob, dtype=np.float32))
        metrics_rows.append({
            "fold": f"fold{fold}",
            "auc": m[0], "accuracy": m[1], "mcc": m[2], "f1": m[3], "pr_auc": m[4],
            "Sensitivity": m[5], "Specificity": m[6], "Precision": m[7], "Recall": m[8],
        })

    if len(fold_preds) >= 2:
        avg_prob = np.mean(fold_preds, axis=0).tolist()
        avg_bin = transfer(avg_prob, args.threshold)
        print("\n===== 5-fold Ensemble | EVAL =====")
        performance(all_y_true, avg_prob, avg_bin)

    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows, columns=[
            "fold","auc","accuracy","mcc","f1","pr_auc","Sensitivity","Specificity","Precision","Recall"
        ])
        value_cols = ["auc","accuracy","mcc","f1","pr_auc","Sensitivity","Specificity","Precision","Recall"]
        avg_row = {"fold":"avg"}; sd_row  = {"fold":"SD"}
        for c in value_cols:
            avg_row[c] = df_metrics[c].mean()
            sd_row[c]  = df_metrics[c].std(ddof=1) if len(df_metrics) >= 2 else 0.0
        df_metrics = pd.concat([df_metrics, pd.DataFrame([avg_row, sd_row])], ignore_index=True)
        df_metrics.to_csv(P(args.out_dir, f"cv_metrics_{suffix}_{args.embed_backend}.csv"), index=False)
        print(f"\n📄 已保存: {P(args.out_dir, f'cv_metrics_{suffix}_{args.embed_backend}.csv')}")

    print("\n===== 评测完成 =====")

if __name__ == "__main__":
    main()
