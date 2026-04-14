# Two modes:
#   1) --mode eval    : evaluate on a single CSV using 5-fold checkpoints (no fine-tuning)
#   2) --mode fewshot : fine-tune on support set -> evaluate on query set

import os
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

from TCR_model import Mymodel_TCR, vocab, tcr_max_len
from feature_extractors import antigenLM_extract, extract_esm2_embeddings, extract_esmc_embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
P = os.path.join

# ===================== Utilities =====================

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def remap_state_dict(state_dict):
    """Remap old checkpoint keys (encoder_H -> encoder_T) for backward compatibility."""
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k.replace("encoder_H.", "encoder_T.")
        new_sd[new_key] = v
    return new_sd

def binarize(probs, thr):
    return [1 if p >= thr else 0 for p in probs]

def performance(y_true, y_prob, y_bin):
    accuracy = accuracy_score(y_true, y_bin)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_bin, zero_division=0)
    recall = recall_score(y_true, y_bin, zero_division=0)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    mcc = matthews_corrcoef(y_true, y_bin)

    print(f'tn={tn}, fp={fp}, fn={fn}, tp={tp}')
    print(f'y_pred: 0={Counter(y_bin)[0]} | 1={Counter(y_bin)[1]}')
    print(f'y_true: 0={Counter(y_true)[0]} | 1={Counter(y_true)[1]}')
    print(f'auc={roc_auc:.4f}|sensitivity={sensitivity:.4f}|specificity={specificity:.4f}|acc={accuracy:.4f}|mcc={mcc:.4f}')
    print(f'precision={precision:.4f}|recall={recall:.4f}|f1={f1:.4f}|aupr={aupr:.4f}')
    return (roc_auc, accuracy, mcc, f1, aupr, sensitivity, specificity, precision, recall)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    if "Peptide" in df.columns: col_map["Peptide"] = "antigen"
    if "CDR3" in df.columns:    col_map["CDR3"] = "TCR"
    if "Label" in df.columns:   col_map["Label"] = "label"
    df = df.rename(columns=col_map)
    need = {"antigen", "TCR", "label"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns: {need}; actual: {set(df.columns)}")
    df["antigen"] = df["antigen"].astype(str).str.strip()
    df["TCR"] = df["TCR"].astype(str).str.strip()
    return df[["antigen", "TCR", "label"]].copy()

# ===================== Embedding Pipeline =====================

def _extract_pep_embeddings(pep_list, emb_cfg, device, pep_max_len):
    backend = emb_cfg["backend"]
    if backend == "AntigenLM":
        if not emb_cfg.get("AntigenLM_path"):
            raise ValueError("AntigenLM_path cannot be empty")
        return antigenLM_extract(
            pep_list, model_name_or_path=emb_cfg["AntigenLM_path"],
            device=("cuda" if torch.cuda.is_available() else "cpu"), max_len=pep_max_len
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
    raise ValueError(f"Unknown embedding backend: {backend}")


def data_process_tcr(data, type_tag, seed, device, emb_cache_dir, pep_max_len=15, tcr_max_len_=34, emb_cfg=None, use_cache=True):
    if emb_cfg is None:
        raise ValueError("emb_cfg cannot be None")
    data = normalize_columns(data)
    data = data[(data["antigen"].str.len() > 0) & (data["TCR"].str.len() > 0)].reset_index(drop=True)

    pep_list = [pep.ljust(tcr_max_len_, '-') for pep in data.antigen]

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

    # TCR tokenization & labels
    tcr_inputs, labels, pep_raw, tcr_raw = [], [], [], []
    for pep_seq, tcr_seq, label in zip(data.antigen, data.TCR, data.label):
        pep_raw.append(pep_seq)
        tcr_raw.append(tcr_seq)
        tcr_inputs.append([vocab.get(n, vocab.get('-', 0)) for n in tcr_seq.ljust(tcr_max_len_, '-')])
        labels.append(int(label) if str(label).isdigit() else 0)

    return pep_embeddings, torch.LongTensor(tcr_inputs), torch.LongTensor(labels), pep_raw, tcr_raw

# ===================== Dataset & Loader =====================

class EvalDataSet_TCR(Dataset):
    def __init__(self, pep_embeds, tcr_idx_tensor, labels_tensor, pep_raw, tcr_raw):
        self.pep_embeds, self.tcr_idx, self.labels = pep_embeds, tcr_idx_tensor, labels_tensor
        self.pep_raw, self.tcr_raw = pep_raw, tcr_raw
        assert len(self.labels) == len(self.tcr_raw) == len(self.pep_raw) == self.tcr_idx.size(0)

    def __len__(self): return self.labels.size(0)
    def __getitem__(self, i): return (self.pep_embeds[i], self.tcr_idx[i], self.labels[i], self.pep_raw[i], self.tcr_raw[i])


def build_loader_from_df(df_sub, type_tag, batch_size, seed, device, emb_cache_dir, emb_cfg):
    pep_inputs, tcr_inputs, labels, pep_raw, tcr_raw = data_process_tcr(
        df_sub, type_tag, seed, device, emb_cache_dir,
        pep_max_len=15, tcr_max_len_=tcr_max_len, emb_cfg=emb_cfg
    )
    ds = EvalDataSet_TCR(pep_inputs, tcr_inputs, labels, pep_raw, tcr_raw)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(type_tag == "few_shot_support"), num_workers=2, pin_memory=True, drop_last=False)
    return loader, ds

# ===================== Inference =====================

@torch.no_grad()
def eval_on_loader(model, loader, device, threshold):
    model.eval()
    y_true, y_prob, y_bin, pep_list, tcr_list, feat_chunks = [], [], [], [], [], []

    for anti_inputs, tcr_inputs, labels, pep_strs, tcr_strs in tqdm(loader, colour='blue'):
        logits, _, features = model(anti_inputs.to(device), tcr_inputs.to(device))
        probs = torch.sigmoid(logits.view(-1)).cpu().numpy()

        y_prob.extend(probs.tolist())
        y_true.extend(labels.cpu().numpy().tolist())
        y_bin.extend((probs >= threshold).astype(np.int32).tolist())
        pep_list.extend(list(pep_strs))
        tcr_list.extend(list(tcr_strs))
        feat_chunks.append(features.detach().cpu().numpy())

    feats = np.concatenate(feat_chunks, axis=0) if feat_chunks else np.zeros((0, 1), np.float32)
    return y_true, y_prob, y_bin, pep_list, tcr_list, feats

# ===================== Few-shot Helpers =====================

def get_pos_weight(labels_tensor):
    y = labels_tensor.view(-1).cpu().numpy()
    pos, neg = (y == 1).sum(), (y == 0).sum()
    return torch.tensor(1.0) if pos == 0 or neg == 0 else torch.tensor(neg / max(pos, 1.0), dtype=torch.float32)


def set_train_mode_for_fewshot(model, train_head_only=True, freeze_bn=True):
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze head layers
    heads = []
    if hasattr(model, "classifier"): heads.append(model.classifier)
    if hasattr(model, "fc"):         heads.append(model.fc)
    if not heads:
        for name, m in model.named_modules():
            if any(k in name.lower() for k in ["head", "cls", "classifier"]):
                heads.append(m)
    for head in heads:
        for p in head.parameters():
            p.requires_grad = True
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
    model.train()
    return model


def build_optimizer(model, lr=1e-3, weight_decay=1e-4):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 1 or any(nd in n.lower() for nd in ["bias", "norm", "bn", "layernorm", "batchnorm"]):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=lr
    )


def finetune_on_support_then_eval_query(
    model, base_state, device, df_support_all, df_query_all,
    batch_size, lr, steps, weight_decay, max_grad_norm,
    do_finetune, threshold, emb_cache_dir, emb_cfg, seed
):
    model.load_state_dict(base_state)

    sup_loader, sup_ds = build_loader_from_df(df_support_all, 'few_shot_support', batch_size, seed, device, emb_cache_dir, emb_cfg)
    if do_finetune and len(sup_ds) > 0:
        set_train_mode_for_fewshot(model, train_head_only=True, freeze_bn=True)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=get_pos_weight(sup_ds.labels).to(device))
        opt = build_optimizer(model, lr=lr, weight_decay=weight_decay)

        for step in range(steps):
            running, n = 0.0, 0
            for anti_inputs, tcr_inputs, labels, _, _ in sup_loader:
                logits, _, _ = model(anti_inputs.to(device), tcr_inputs.to(device))
                loss = loss_fn(logits.view(-1), labels.float().to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                running += loss.item() * labels.size(0)
                n += labels.size(0)
            print(f"[fewshot] step {step+1}/{steps} loss={running/max(n,1):.4f}")

    # Query inference
    qry_loader, _ = build_loader_from_df(df_query_all, 'few_shot_query', batch_size, seed, device, emb_cache_dir, emb_cfg)
    y_true, y_prob, y_bin, pep_list, tcr_list, feats = eval_on_loader(model, qry_loader, device, threshold)

    pred_df = pd.DataFrame({"antigen": pep_list, "TCR": tcr_list, "label_true": y_true, "label_prob": y_prob, "label_pred": y_bin})

    has_pos, has_neg = any(v == 1 for v in y_true), any(v == 0 for v in y_true)
    if has_pos and has_neg:
        prec, reca, _ = precision_recall_curve(y_true, y_prob)
        aupr_, auc_ = auc(reca, prec), roc_auc_score(y_true, y_prob)
    else:
        aupr_, auc_ = float("nan"), float("nan")

    acc_ = accuracy_score(y_true, y_bin)
    f1_ = f1_score(y_true, y_bin, zero_division=0)
    mcc_ = matthews_corrcoef(y_true, y_bin) if (has_pos or has_neg) else float("nan")
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel().tolist()
        sens, spec = tp / (tp + fn) if (tp + fn) > 0 else 0.0, tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except Exception:
        sens = spec = float("nan")

    metrics = {
        "auc": auc_, "aupr": aupr_, "accuracy": acc_, "f1": f1_, "mcc": mcc_,
        "Sensitivity": sens, "Specificity": spec,
        "Precision": precision_score(y_true, y_bin, zero_division=0),
        "Recall": recall_score(y_true, y_bin, zero_division=0),
    }
    return pred_df, metrics, feats, np.asarray(y_true, dtype=np.int64)

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "fewshot"], default="eval")

    # eval mode
    ap.add_argument("--data_csv", default="./data/covid_set.csv")
    # fewshot mode
    ap.add_argument("--support_csv", default="./data/few_shot_support_set.csv")
    ap.add_argument("--query_csv", default="./data/few_shot_query_set.csv")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # common
    ap.add_argument("--weights_dir", default="../trained_model/pTCR/test")
    ap.add_argument("--out_dir", default="../result/pTCR/test")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=22)

    # embedding
    ap.add_argument("--emb_cache_dir", default="./data_cached")
    ap.add_argument("--embed_backend", choices=["AntigenLM", "esm2", "esmc"], default="AntigenLM")
    ap.add_argument("--AntigenLM_path", default="../../LLM/AntigenLM_300M_SS")
    ap.add_argument("--esm2_model_name", default="esm2_t33_650M_UR50D")
    ap.add_argument("--esmc_model_name", default="esmc_300m")

    args = ap.parse_args()
    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = Mymodel_TCR().to(device)

    emb_cfg = {
        "backend": args.embed_backend,
        "AntigenLM_path": args.AntigenLM_path,
        "esm2_model_name": args.esm2_model_name,
        "esmc_model_name": args.esmc_model_name,
    }

    if args.mode == "eval":
        df_eval = pd.read_csv(args.data_csv)
        suffix = os.path.splitext(os.path.basename(args.data_csv))[0]
        loader, _ = build_loader_from_df(df_eval, suffix, args.batch_size, args.seed, device, args.emb_cache_dir, emb_cfg)

        fold_preds, all_y_true, metrics_rows = [], None, []

        for fold in range(1, 6):
            ckpt = P(args.weights_dir, f"fold{fold}_seed{args.seed}_{args.embed_backend}.pt")
            if not os.path.exists(ckpt):
                print(f"[Skip] Weights not found: {ckpt}")
                continue

            model.load_state_dict(remap_state_dict(torch.load(ckpt, map_location=device)))
            y_true, y_prob, y_bin, pep_strs, tcr_strs, feats = eval_on_loader(model, loader, device, args.threshold)

            df_fold = pd.DataFrame({
                "fold": [fold] * len(y_true), "TCR": tcr_strs, "antigen": pep_strs,
                "label_true": y_true, "label_pred": y_bin, "label_prob": y_prob
            })
            df_fold.to_csv(P(args.out_dir, f"{args.embed_backend}_{suffix}_pred_results_fold{fold}.csv"), index=False)

            if all_y_true is None:
                all_y_true = y_true

            print(f"\n===== Fold {fold} | EVAL =====")
            m = performance(y_true, y_prob, y_bin)
            fold_preds.append(np.asarray(y_prob, dtype=np.float32))
            metrics_rows.append({
                "fold": f"fold{fold}",
                "auc": m[0], "accuracy": m[1], "mcc": m[2], "f1": m[3], "aupr": m[4],
                "Sensitivity": m[5], "Specificity": m[6], "Precision": m[7], "Recall": m[8],
            })

        # 5-fold average
        if len(fold_preds) >= 2:
            avg_prob = np.mean(fold_preds, axis=0).tolist()
            avg_bin = binarize(avg_prob, args.threshold)
            print("\n===== 5-fold Average | EVAL =====")
            performance(all_y_true, avg_prob, avg_bin)

        # Summary metrics
        if metrics_rows:
            df_metrics = pd.DataFrame(metrics_rows)
            value_cols = ["auc", "accuracy", "mcc", "f1", "aupr", "Sensitivity", "Specificity", "Precision", "Recall"]
            avg_row, sd_row = {"fold": "avg"}, {"fold": "sd"}
            for c in value_cols:
                avg_row[c] = df_metrics[c].mean()
                sd_row[c] = df_metrics[c].std(ddof=1) if len(df_metrics) >= 2 else 0.0
            df_metrics = pd.concat([df_metrics, pd.DataFrame([avg_row, sd_row])], ignore_index=True)
            metrics_path = P(args.out_dir, f"{args.embed_backend}_{suffix}_metrics.csv")
            df_metrics.to_csv(metrics_path, index=False)
            print(f"\nFinal metrics saved to: {metrics_path}")

    else:
        # FEWSHOT mode
        suffix = "few_shot" if args.finetune else "fewshot_noft"
        df_sup_all = normalize_columns(pd.read_csv(args.support_csv))
        df_qry_all = normalize_columns(pd.read_csv(args.query_csv))
        all_metrics = []

        for fold in range(1, 6):
            ckpt = P(args.weights_dir, f"fold{fold}_seed{args.seed}_{args.embed_backend}.pt")
            if not os.path.exists(ckpt):
                print(f"[Skip] Weights not found: {ckpt}")
                continue

            base_state = remap_state_dict(torch.load(ckpt, map_location=device))
            model.load_state_dict(base_state)

            print(f"\n===== Fold {fold} | FEWSHOT | finetune={args.finetune} =====")
            pred_df, metrics, feats, y_true_arr = finetune_on_support_then_eval_query(
                model=model, base_state=base_state, device=device,
                df_support_all=df_sup_all, df_query_all=df_qry_all,
                batch_size=args.batch_size, lr=args.lr, steps=args.steps,
                weight_decay=args.weight_decay, max_grad_norm=1.0,
                do_finetune=args.finetune, threshold=args.threshold,
                emb_cache_dir=args.emb_cache_dir, emb_cfg=emb_cfg, seed=args.seed,
            )

            out_csv = P(args.out_dir, f"{args.embed_backend}_{suffix}_pred_results_fold{fold}.csv")
            pred_df.insert(0, "fold", fold)
            pred_df.to_csv(out_csv, index=False)
            print(f"Predictions saved to: {out_csv}")

            if metrics is not None:
                print(f"auc={metrics['auc']:.4f}|aupr={metrics['aupr']:.4f}|acc={metrics['accuracy']:.4f}|"
                      f"f1={metrics['f1']:.4f}|mcc={metrics['mcc']:.4f}")
                print(f"Sensitivity={metrics['Sensitivity']:.4f}|Specificity={metrics['Specificity']:.4f}|"
                      f"Precision={metrics['Precision']:.4f}|Recall={metrics['Recall']:.4f}")
                row = {"fold": f"fold{fold}"}
                row.update(metrics)
                all_metrics.append(row)

        if all_metrics:
            df_metrics = pd.DataFrame(all_metrics)
            mean_row = {"fold": "avg"}
            for col in ["auc", "aupr", "accuracy", "f1", "mcc", "Sensitivity", "Specificity", "Precision", "Recall"]:
                mean_row[col] = df_metrics[col].mean()
            df_metrics = pd.concat([df_metrics, pd.DataFrame([mean_row])], ignore_index=True)
            metrics_csv = P(args.out_dir, f"{args.embed_backend}_{suffix}_metrics.csv")
            df_metrics.to_csv(metrics_csv, index=False)
            print(f"Few-shot metrics saved to: {metrics_csv}")

    print("\n===== Done =====")


if __name__ == "__main__":
    main()
