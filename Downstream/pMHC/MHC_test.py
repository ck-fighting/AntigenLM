import os
import json
import hashlib
import argparse
from collections import Counter
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, auc, accuracy_score, f1_score,
    precision_recall_curve, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef
)

from MHC_model import *
from feature_extractors import antigenLM_extract, extract_esm2_embeddings, extract_esmc_embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
P = os.path.join

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def transfer(probs, thr):
    return [1 if p >= thr else 0 for p in probs]

def performance(y_true, y_prob, y_bin):
    acc = accuracy_score(y_true, y_pred=y_bin)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_bin, zero_division=0)
    recall = recall_score(y_true, y_bin, zero_division=0)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float('nan')
        
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec) if len(reca) > 1 else float('nan')
    mcc = matthews_corrcoef(y_true, y_bin) if (tp + tn + fp + fn) > 0 else 0.0

    print(f'tn={tn}, fp={fp}, fn={fn}, tp={tp}')
    print(f'y_pred: 0={Counter(y_bin)[0]} | 1={Counter(y_bin)[1]}')
    print(f'y_true: 0={Counter(y_true)[0]} | 1={Counter(y_true)[1]}')
    print(f'auc={roc_auc:.4f}|sensitivity={sensitivity:.4f}|specificity={specificity:.4f}|acc={acc:.4f}|mcc={mcc:.4f}')
    print(f'precision={precision:.4f}|recall={recall:.4f}|f1={f1:.4f}|aupr={aupr:.4f}')
    
    return {
        'auc': roc_auc, 'acc': acc, 'mcc': mcc, 'f1': f1, 'aupr': aupr,
        'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'recall': recall,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'pred_0': Counter(y_bin)[0], 'pred_1': Counter(y_bin)[1],
        'true_0': Counter(y_true)[0], 'true_1': Counter(y_true)[1]
    }

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Peptide": "peptide", "antigen": "peptide", "HLA": "HLA", "hla": "HLA", "Label": "label"})
    if not {"peptide", "HLA", "label"}.issubset(df.columns):
        raise ValueError(f"Missing columns: {set(df.columns)}")
        
    df["peptide"] = df["peptide"].astype(str).str.strip()
    df["HLA"] = df["HLA"].astype(str).str.strip()
    return df[["peptide", "HLA", "label"]].copy()

def _checksum_of_list(str_list):
    h = hashlib.md5()
    for s in str_list:
        h.update(s.encode('utf-8', errors='ignore') + b'\n')
    return h.hexdigest()

def save_npz(feats, labels, out_path):
    feats = np.asarray(feats, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    np.savez(out_path, feats=feats, labels=labels)
    print(f"📦 保存 t-SNE npz：{out_path}  (feats={feats.shape}, labels={labels.shape})")

def _extract_pep_embeddings(pep_list, emb_cfg, device, pep_max_len):
    backend = emb_cfg["backend"]
    if backend == "AntigenLM":
        return antigenLM_extract(pep_list, model_name_or_path=emb_cfg["AntigenLM_path"], device=device, max_len=pep_max_len)
    elif backend == "esm2":
        return extract_esm2_embeddings(pep_list, model_name=emb_cfg.get("esm2_model_name", "esm2_t33_650M_UR50D"), device=device, max_len=pep_max_len)
    elif backend == "esmc":
        from esm.models.esmc import ESMC
        client = ESMC.from_pretrained(emb_cfg.get("esmc_model_name", "esmc_300m")).to(device)
        return extract_esmc_embeddings(pep_list, client=client, device=device, batch_size=512, max_len=15, model_max_len=256)
        
    raise ValueError(f"未知嵌入后端：{backend}")

def data_process_hla(data: pd.DataFrame, type_tag: str, seed: int, device: torch.device, emb_cache_dir: str, pep_max_len=15, hla_max_len_=34, emb_cfg=None, use_cache=True):
    data = normalize_columns(data)
    data = data.loc[(data["peptide"].str.len() > 0) & (data["HLA"].str.len() > 0)].reset_index(drop=True)
    
    pep_list = [p.ljust(hla_max_len_, '-') for p in data.peptide]
    pep_raw = data.peptide.tolist()
    
    os.makedirs(emb_cache_dir, exist_ok=True)
    emb_cache_path = P(emb_cache_dir, f"cached_pep_embeddings_{type_tag}_{seed}_{emb_cfg['backend']}.pt")
    
    if use_cache and os.path.isfile(emb_cache_path):
        print(f"[Cache] Use peptide embeddings: {emb_cache_path}")
        pep_embeddings = torch.load(emb_cache_path, map_location='cpu')
    else:
        print(f"[Embed] Extracting {len(pep_list)} peptides via {emb_cfg['backend']} ...")
        pep_embeddings = _extract_pep_embeddings(pep_list, emb_cfg, "cuda" if torch.cuda.is_available() else "cpu", pep_max_len)
        torch.save(pep_embeddings.detach().cpu(), emb_cache_path)
        
    hla_raw = data.HLA.tolist()
    hla_inputs = [[vocab.get(n, vocab.get('-', 0)) for n in seq.ljust(hla_max_len_, '-')] for seq in hla_raw]
    labels = [int(y) if str(y).isdigit() else 0 for y in data.label]
    
    return pep_embeddings, torch.LongTensor(hla_inputs), torch.LongTensor(labels), pep_raw, hla_raw

class EvalDataSet_HLA(Dataset):
    def __init__(self, pep_embeds, hla_idx_tensor, labels_tensor, pep_raw, hla_raw):
        self.pep_embeds = pep_embeds
        self.hla_idx = hla_idx_tensor
        self.labels = labels_tensor
        self.pep_raw = pep_raw
        self.hla_raw = hla_raw
        
    def __len__(self):
        return self.labels.size(0)
        
    def __getitem__(self, i):
        return self.pep_embeds[i], self.hla_idx[i], self.labels[i], self.pep_raw[i], self.hla_raw[i]

def build_loader_from_df(df_sub: pd.DataFrame, type_tag: str, batch_size: int, seed: int, device: torch.device, emb_cache_dir: str, emb_cfg):
    pep_inputs, hla_inputs, labels, pep_raw, hla_raw = data_process_hla(
        df_sub, type_tag, seed, device, emb_cache_dir, 15, hla_max_len, emb_cfg
    )
    ds = EvalDataSet_HLA(pep_inputs, hla_inputs, labels, pep_raw, hla_raw)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    return loader, ds

@torch.no_grad()
def eval_on_loader(model, loader, device, threshold):
    model.eval()
    y_true, y_prob, y_bin = [], [], []
    pep_list, hla_list, feat_chunks = [], [], []
    
    for anti_inputs, hla_inputs, labels, pep_strs, hla_strs in tqdm(loader, colour='blue'):
        logits, _, features = model(anti_inputs.to(device), hla_inputs.to(device))
        probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
        
        y_prob.extend(probs.tolist())
        y_true.extend(labels.tolist())
        y_bin.extend((probs >= threshold).astype(np.int32).tolist())
        
        pep_list.extend(list(pep_strs))
        hla_list.extend(list(hla_strs))
        feat_chunks.append(features.detach().cpu().numpy())
        
    feats = np.concatenate(feat_chunks, axis=0) if feat_chunks else np.zeros((0, 1), np.float32)
    return y_true, y_prob, y_bin, pep_list, hla_list, feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="./data/micro_test_set.csv")
    ap.add_argument("--weights_dir", default="../trained_model/pMHC/")
    ap.add_argument("--out_dir", default="../result/pMHC/")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--emb_cache_dir", default="./data_cached/")
    ap.add_argument("--embed_backend", choices=["AntigenLM", "esm2", "esmc"], default="AntigenLM")
    ap.add_argument("--AntigenLM_path", default="../../LLM/AntigenLM_300M_SS")
    ap.add_argument("--esm2_model_name", default="esm2_t33_650M_UR50D")
    ap.add_argument("--esmc_model_name", default="esmc_300m")
    args = ap.parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    emb_cfg = {
        "backend": args.embed_backend,
        "AntigenLM_path": args.AntigenLM_path,
        "esm2_model_name": args.esm2_model_name,
        "esmc_model_name": args.esmc_model_name
    }
    
    model = Mymodel_HLA().to(device)
    loader, _ = build_loader_from_df(
        pd.read_csv(args.data_csv), "micro_test", args.batch_size, args.seed, device, args.emb_cache_dir, emb_cfg
    )

    fold_preds, all_y_true, metrics_rows = [], None, []
    for fold in range(1, 6):
        ckpt = P(args.weights_dir, f"fold{fold}_seed22_{args.embed_backend}.pt")
        if not os.path.exists(ckpt):
            print(f"[Skip] 未找到权重：{ckpt}")
            continue
            
        model.load_state_dict(torch.load(ckpt, map_location=device))
        y_true, y_prob, y_bin, pep_strs, hla_strs, _ = eval_on_loader(model, loader, device, args.threshold)
        
        df_fold = pd.DataFrame({
            "fold": [fold] * len(y_true),
            "HLA": hla_strs,
            "peptide": pep_strs,
            "label_true": y_true,
            "label_pred": y_bin,
            "label_prob": y_prob
        })
        df_fold.to_csv(P(args.out_dir, f"{args.embed_backend}_micro_test_pred_results_fold{fold}.csv"), index=False)
        
        all_y_true = all_y_true if all_y_true is not None else y_true
        
        print(f"\n===== Fold {fold} | EVAL =====")
        m = performance(y_true, y_prob, y_bin)
        fold_preds.append(np.asarray(y_prob, dtype=np.float32))
        
        metrics_rows.append({
            "fold": f"fold{fold}", "auc": m['auc'], "accuracy": m['acc'], "mcc": m['mcc'], 
            "f1": m['f1'], "pr_auc": m['aupr'], "Sensitivity": m['sensitivity'], 
            "Specificity": m['specificity'], "Precision": m['precision'], "Recall": m['recall'], 
            "tn": m['tn'], "fp": m['fp'], "fn": m['fn'], "tp": m['tp'], 
            "pred_0": m['pred_0'], "pred_1": m['pred_1'], "true_0": m['true_0'], "true_1": m['true_1']
        })

    if len(fold_preds) >= 2:
        print("\n===== 5-fold Average | EVAL =====")
        avg_m = pd.DataFrame(metrics_rows).mean(numeric_only=True)
        print(
            f"tn={avg_m['tn']:.0f}, fp={avg_m['fp']:.0f}, "
            f"fn={avg_m['fn']:.0f}, tp={avg_m['tp']:.0f}"
        )
        print(f"y_pred: 0={avg_m['pred_0']:.0f} | 1={avg_m['pred_1']:.0f}")
        print(f"y_true: 0={avg_m['true_0']:.0f} | 1={avg_m['true_1']:.0f}")
        print(
            f"auc={avg_m['auc']:.4f}|sensitivity={avg_m['Sensitivity']:.4f}|"
            f"specificity={avg_m['Specificity']:.4f}|acc={avg_m['accuracy']:.4f}|mcc={avg_m['mcc']:.4f}"
        )
        print(
            f"precision={avg_m['Precision']:.4f}|recall={avg_m['Recall']:.4f}|"
            f"f1={avg_m['f1']:.4f}|aupr={avg_m['pr_auc']:.4f}"
        )

    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows, columns=[
            "fold","auc","accuracy","mcc","f1","pr_auc","Sensitivity","Specificity","Precision","Recall"
        ])
        avg_row, sd_row = {"fold":"avg"}, {"fold":"SD"}
        
        for c in df_metrics.columns[1:]:
            avg_row[c] = df_metrics[c].mean()
            sd_row[c] = df_metrics[c].std(ddof=1) if len(df_metrics) >= 2 else 0.0
            
        df_metrics = pd.concat([df_metrics, pd.DataFrame([avg_row, sd_row])], ignore_index=True)
        out_csv_path = P(args.out_dir, f"{args.embed_backend}_micro_test_metrics.csv")
        df_metrics.to_csv(out_csv_path, index=False)
        print(f"\n📄 已保存: {out_csv_path}")

    print("\n===== 评测完成 =====")

if __name__ == "__main__":
    main()
