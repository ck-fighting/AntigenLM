# eval_or_fewshot.py
# 两种模式：
#   1) --mode eval      : 单CSV不微调评测（逐fold评测、导出预测/指标/TSNE）
#   2) --mode fewshot   : 用support(+可选微调) -> 在query上评测（逐fold导出预测/指标/TSNE）

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

# 你的工程内组件（需提供: vocab, Mymodel_TCR, antigenLM_extract, extract_esm2_embeddings, extract_esmc_embeddings）
from TCR_model import Mymodel_TCR, vocab, tcr_max_len
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
    if "Peptide" in df.columns: col_map["Peptide"] = "antigen"
    if "CDR3" in df.columns:    col_map["CDR3"]    = "TCR"
    if "Label" in df.columns:   col_map["Label"]   = "label"
    df = df.rename(columns=col_map)
    need = {"antigen","TCR","label"}
    if not need.issubset(df.columns):
        raise ValueError(f"缺少列：{need}；实际列：{set(df.columns)}")
    df["antigen"] = df["antigen"].astype(str).str.strip()
    df["TCR"]     = df["TCR"].astype(str).str.strip()
    return df[["antigen","TCR","label"]].copy()

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

def data_process_tcr(
    data: pd.DataFrame,
    type_tag: str,
    seed: int,
    device: torch.device,
    emb_cache_dir: str,
    pep_max_len: int = 15,
    tcr_max_len_: int = 34,
    emb_cfg=None,  # {"backend": "...", "antigenLM_path": "...", ...}
    use_cache: bool = True,
):
    if emb_cfg is None:
        raise ValueError("emb_cfg 不能为空")
    data = normalize_columns(data)
    # 过滤空串
    mask = (data["antigen"].str.len()>0) & (data["TCR"].str.len()>0)
    data = data.loc[mask].reset_index(drop=True)

    pep_list, pep_raw = [], []
    for pep in data.antigen:
        pep_raw.append(pep)
        pep_list.append(pep.ljust(tcr_max_len_, '-'))  # 兼容你原逻辑

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

    # TCR索引 + 标签
    tcr_inputs, labels, tcr_raw = [], [], []
    for tcr_seq, label in zip(data.TCR, data.label):
        tcr_raw.append(tcr_seq)
        idxs = [vocab.get(n, vocab.get('-', 0)) for n in tcr_seq.ljust(tcr_max_len_, '-')]
        tcr_inputs.append(idxs)
        labels.append(int(label) if str(label).isdigit() else 0)

    return (
        pep_embeddings,
        torch.LongTensor(tcr_inputs),
        torch.LongTensor(labels),
        pep_raw, tcr_raw
    )

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
        return (self.pep_embeds[i], self.tcr_idx[i], self.labels[i], self.pep_raw[i], self.tcr_raw[i])

def build_loader_from_df(
    df_sub: pd.DataFrame, 
    type_tag: str, 
    batch_size: int, 
    seed: int, 
    device: torch.device,
    emb_cache_dir: str,
    emb_cfg,
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
    shuffle = (type_tag in {"few_shot_support"})
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True, drop_last=False)
    return loader, ds

# ------------------ 推理（不微调） ------------------
@torch.no_grad()
def eval_on_loader(model, loader, device, threshold):
    model.eval()
    y_true, y_prob, y_bin = [], [], []
    pep_list, tcr_list, feat_chunks = [], [], []

    for anti_inputs, tcr_inputs, labels, pep_strs, tcr_strs in tqdm(loader, colour='blue'):
        anti_inputs = anti_inputs.to(device)
        tcr_inputs  = tcr_inputs.to(device)
        labels      = labels.to(device)

        logits, _, features = model(anti_inputs, tcr_inputs)
        probs = torch.sigmoid(logits.view(-1)).cpu().numpy()

        y_prob.extend(probs.tolist())
        y_true.extend(labels.cpu().numpy().tolist())
        y_bin.extend((probs >= threshold).astype(np.int32).tolist())
        pep_list.extend(list(pep_strs))
        tcr_list.extend(list(tcr_strs))
        feat_chunks.append(features.detach().cpu().numpy())

    feats = np.concatenate(feat_chunks, axis=0) if feat_chunks else np.zeros((0, 1), np.float32)
    return y_true, y_prob, y_bin, pep_list, tcr_list, feats

# ------------------ few-shot 微调辅助 ------------------
def get_pos_weight(labels_tensor: torch.LongTensor) -> torch.Tensor:
    y = labels_tensor.view(-1).cpu().numpy()
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0 or neg == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / max(pos, 1.0), dtype=torch.float32)

def set_train_mode_for_fewshot(model, train_head_only=True, freeze_bn=True):
    # 全冻结
    for p in model.parameters():
        p.requires_grad = False
    # 只开头部（根据你模型的命名尝试找头）
    heads = []
    if hasattr(model, "classifier"): heads.append(model.classifier)
    if hasattr(model, "fc"):         heads.append(model.fc)
    if not heads:
        for name, m in model.named_modules():
            if any(k in name.lower() for k in ["head","cls","classifier"]):
                heads.append(m)
    for head in heads:
        for p in head.parameters():
            p.requires_grad = True
    # 冻结 BN
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
        if p.dim()==1 or any(nd in n.lower() for nd in ["bias","norm","bn","layernorm","batchnorm"]):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr
    )

def finetune_on_support_then_eval_query(
    model, base_state, device,
    df_support_all, df_query_all,
    batch_size, lr, steps, weight_decay, max_grad_norm,
    do_finetune, threshold,
    emb_cache_dir, emb_cfg, seed,
):
    # 重置为该fold初始权重
    model.load_state_dict(base_state)

    # 1) support（可选微调）
    sup_loader, sup_ds = build_loader_from_df(
        df_support_all,
        'few_shot_support',
        batch_size,
        seed,
        device,
        emb_cache_dir,
        emb_cfg,
    )
    if do_finetune and len(sup_ds) > 0:
        set_train_mode_for_fewshot(model, train_head_only=True, freeze_bn=True)
        pos_w   = get_pos_weight(sup_ds.labels).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt     = build_optimizer(model, lr=lr, weight_decay=weight_decay)

        for step in range(steps):
            running, n = 0.0, 0
            for anti_inputs, tcr_inputs, labels, _, _ in sup_loader:
                anti_inputs = anti_inputs.to(device)
                tcr_inputs  = tcr_inputs.to(device)
                labels      = labels.float().to(device)

                logits, _, _ = model(anti_inputs, tcr_inputs)
                loss = loss_fn(logits.view(-1), labels)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                running += loss.item() * labels.size(0); n += labels.size(0)
            print(f"[fewshot] step {step+1}/{steps} loss={running/max(n,1):.4f}")

    # 2) query 推理（收集 feats+labels 用于 t-SNE）
    qry_loader, _ = build_loader_from_df(
        df_query_all,
        'few_shot_query',
        batch_size,
        seed,
        device,
        emb_cache_dir,
        emb_cfg,
    )
    y_true, y_prob, y_bin, pep_list, tcr_list, feats = eval_on_loader(model, qry_loader, device, threshold)

    # 3) 组装 dataframe + 计算整体指标
    pred_df = pd.DataFrame({
        "antigen": pep_list,
        "TCR": tcr_list,
        "label_true": y_true,
        "label_prob": y_prob,
        "label_pred": y_bin,
    })

    has_pos = any(v == 1 for v in y_true)
    has_neg = any(v == 0 for v in y_true)
    if has_pos and has_neg:
        prec, reca, _ = precision_recall_curve(y_true, y_prob)
        aupr_ = auc(reca, prec)
        auc_  = roc_auc_score(y_true, y_prob)
    else:
        aupr_, auc_ = float("nan"), float("nan")

    acc_ = accuracy_score(y_true, y_bin)
    f1_  = f1_score(y_true, y_bin, zero_division=0)
    mcc_ = matthews_corrcoef(y_true, y_bin) if (has_pos or has_neg) else float("nan")
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel().tolist()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    except Exception:
        sens = spec = float("nan")
    prc  = precision_score(y_true, y_bin, zero_division=0)
    rcl  = recall_score(y_true, y_bin, zero_division=0)

    metrics = {
        "auc": auc_, "aupr": aupr_, "accuracy": acc_, "f1": f1_,
        "mcc": mcc_, "Sensitivity": sens, "Specificity": spec,
        "Precision": prc, "Recall": rcl,
    }
    return pred_df, metrics, feats, np.asarray(y_true, dtype=np.int64)

# ------------------ 主流程 ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval","fewshot"], default="eval", help="eval: 单CSV不微调评测; fewshot: 用support(可微调)在query上评测")
    # eval
    ap.add_argument("--data_csv", default="./data/covid_set.csv")
    # fewshot
    ap.add_argument("--support_csv", default="./data/few_shot_support_set.csv")
    ap.add_argument("--query_csv",   default="./data/few_shot_query_set.csv")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # 通用
    ap.add_argument("--weights_dir", default="../trained_model/TCR_micro/Ablation/")
    ap.add_argument("--out_dir", default="../Result/TCR/Ablation")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=22)

    # 嵌入缓存 & 后端
    ap.add_argument("--emb_cache_dir", default="./data_cached/Majority")
    ap.add_argument("--embed_backend", choices=["antigenLM","antigenLM_withoutSS","antigenLM_withoutSlidingwindow","esm2","esmc"], default="antigenLM_withoutSS")
    ap.add_argument("--antigenLM_path", default="/home/dataset-local/AntigenLM/LLM/Result_antigenLM_300M_SS_2")
    ap.add_argument("--antigenLM_withoutSS_path", default="/home/dataset-local/AntigenLM/Other_LLM/AntigenLM_300M_withoutSS")
    ap.add_argument("--antigenLM_withoutSlidingwindow_path", default="/home/dataset-local/AntigenLM/Other_LLM/AntigenLM_300M_withoutSlidingwindow")
    ap.add_argument("--esm2_model_name", default="esm2_t33_650M_UR50D")
    ap.add_argument("--esmc_model_name", default="esmc_300m")

    args = ap.parse_args()
    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = Mymodel_TCR().to(device)

    emb_cfg = {
        "backend": args.embed_backend,
        "antigenLM_path": args.antigenLM_path,
        "antigenLM_withoutSS_path": args.antigenLM_withoutSS_path,
        "antigenLM_withoutSlidingwindow_path": args.antigenLM_withoutSlidingwindow_path,
        "esm2_model_name": args.esm2_model_name,
        "esmc_model_name": args.esmc_model_name,
    }

    if args.mode == "eval":
        # 数据与命名后缀
        df_eval = pd.read_csv(args.data_csv)
        suffix  = "covid"  # 例如 majority_test_set
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

            y_true, y_prob, y_bin, pep_strs, tcr_strs, feats = eval_on_loader(model, loader, device, args.threshold)

            # 保存 t-SNE npz（feats+labels）
            # tsne_npz = P(args.out_dir, f"tsne_fold{fold}_{suffix}.npz")
            # save_npz(feats, y_true, tsne_npz)

            # 保存每折逐样本预测
            df_fold = pd.DataFrame({
                "fold": [fold]*len(y_true),
                "TCR": tcr_strs,
                "antigen": pep_strs,
                "label_true": y_true,
                "label_pred": y_bin,
                "label_prob": y_prob
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

        # 5折平均
        if len(fold_preds) >= 2:
            avg_prob = np.mean(fold_preds, axis=0).tolist()
            avg_bin  = transfer(avg_prob, args.threshold)
            print("\n===== 5-fold Ensemble | EVAL =====")
            performance(all_y_true, avg_prob, avg_bin)

        # 汇总指标
        if metrics_rows:
            df_metrics = pd.DataFrame(metrics_rows, columns=[
                "fold","auc","accuracy","mcc","f1","pr_auc","Sensitivity","Specificity","Precision","Recall"
            ])
            value_cols = ["auc","accuracy","mcc","f1","pr_auc","Sensitivity","Specificity","Precision","Recall"]
            avg_row = {"fold":"avg"}; sd_row  = {"fold":"SD"}
            for c in value_cols:
                avg_row[c] = df_metrics[c].mean()
                sd_row[c]  = df_metrics[c].std(ddof=1) if len(df_metrics)>=2 else 0.0
            df_metrics = pd.concat([df_metrics, pd.DataFrame([avg_row, sd_row])], ignore_index=True)
            metrics_path = P(args.out_dir, f"cv_metrics_{suffix}_{args.embed_backend}.csv")
            df_metrics.to_csv(metrics_path, index=False)
            print(f"\n📄 已保存: {metrics_path}")

    else:
        # FEWSHOT
        suffix = "few_shot" if args.finetune else "fewshot_global_noft"
        df_sup_all = normalize_columns(pd.read_csv(args.support_csv))
        df_qry_all = normalize_columns(pd.read_csv(args.query_csv))
        all_metrics = []

        for fold in range(1, 5+1):
            ckpt = P(args.weights_dir, f"model_TCR_fold{fold}_seed22_antigenLM.pth")
            if not os.path.exists(ckpt):
                print(f"[Skip] 未找到权重：{ckpt}")
                continue

            base_state = torch.load(ckpt, map_location=device)
            model.load_state_dict(base_state)

            print(f"\n===== Fold {fold} | FEWSHOT-GLOBAL | finetune={args.finetune} =====")
            pred_df, metrics, feats, y_true_arr = finetune_on_support_then_eval_query(
                model=model,
                base_state=base_state,
                device=device,
                df_support_all=df_sup_all,
                df_query_all=df_qry_all,
                batch_size=args.batch_size,
                lr=args.lr,
                steps=args.steps,
                weight_decay=args.weight_decay,
                max_grad_norm=1.0,
                do_finetune=args.finetune,
                threshold=args.threshold,
                emb_cache_dir=args.emb_cache_dir,
                emb_cfg=emb_cfg,
                seed=args.seed,
            )

            # 保存 t-SNE npz（feats+labels）
            tsne_npz = P(args.out_dir, f"tsne_fold{fold}_{suffix}.npz")
            save_npz(feats, y_true_arr, tsne_npz)

            # 保存逐样本预测
            out_csv = P(args.out_dir, f"preds_fold{fold}_{suffix}.csv")
            pred_df.insert(0, "fold", fold)
            pred_df.to_csv(out_csv, index=False)
            print(f"📄 已保存：{out_csv}")

            if metrics is not None:
                row = {"fold": f"fold{fold}"}; row.update(metrics); all_metrics.append(row)

        if all_metrics:
            df_metrics = pd.DataFrame(all_metrics)
            mean_row = {"fold":"avg_all"}
            for col in ["auc","aupr","accuracy","f1","mcc","Sensitivity","Specificity","Precision","Recall"]:
                mean_row[col] = df_metrics[col].mean()
            df_metrics = pd.concat([df_metrics, pd.DataFrame([mean_row])], ignore_index=True)
            metrics_csv = P(args.out_dir, f"cv_metrics_{suffix}.csv")
            df_metrics.to_csv(metrics_csv, index=False)
            print(f"📄 已保存 few-shot 指标：{metrics_csv}")

    print("\n===== 评测完成 =====")

if __name__ == "__main__":
    main()
