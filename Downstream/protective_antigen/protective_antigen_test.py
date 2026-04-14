import os
import argparse
import random
import contextlib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)

from protective_antigen_model import SoluModel

P = os.path.join

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def _is_cuda(dev):
    return (isinstance(dev, torch.device) and dev.type == "cuda") or (isinstance(dev, str) and str(dev).startswith("cuda"))

def extract_esmc_embeddings(sequences, client, device, batch_size=16, max_len=512, model_max_len=4096, autocast_dtype=torch.bfloat16):
    from esm.sdk.api import ESMProtein, LogitsConfig
    client.eval()
    all_batches = []

    def _to_bld(emb: torch.Tensor) -> torch.Tensor:
        return emb.unsqueeze(0) if emb.ndim == 2 else emb

    def _pad_trunc(emb: torch.Tensor, max_len_: int) -> np.ndarray:
        out_np = []
        for e in emb:
            if e.size(0) >= max_len_:
                out_np.append(e[:max_len_].numpy())
            else:
                pad = np.zeros((max_len_ - e.size(0), e.size(1)), dtype=np.float32)
                out_np.append(np.vstack([e.numpy(), pad]))
        return np.stack(out_np)

    amp_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if (autocast_dtype is not None and _is_cuda(device)) else contextlib.nullcontext()

    for i in tqdm(range(0, len(sequences), batch_size), desc="ESMC Extract"):
        seqs = [s[: model_max_len - 2] for s in sequences[i:i + batch_size]]
        proteins = [ESMProtein(sequence=s) for s in seqs]

        with torch.no_grad(), amp_ctx:
            ok = False
            try:
                pt = client.encode(proteins)
                out = client.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
                batch_np = _pad_trunc(_to_bld(out.embeddings).to("cpu"), max_len)
                ok = True
            except Exception:
                pass

            if not ok:
                np_list = []
                for p in proteins:
                    pt1 = client.encode(p)
                    out1 = client.logits(pt1, LogitsConfig(sequence=True, return_embeddings=True))
                    np_list.append(_pad_trunc(_to_bld(out1.embeddings).to("cpu"), max_len)[0])
                batch_np = np.stack(np_list)

        all_batches.append(batch_np)

    return torch.from_numpy(np.concatenate(all_batches, axis=0)).float()

def extract_esm_embeddings(sequences, model, batch_converter, device, layer=33, batch_size=64, max_len=512, model_max_len=1024):
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="ESM Extract"):
            batch = [seq[:model_max_len-2] for seq in sequences[i:i+batch_size]]
            batch_data = [(f"seq{j}", seq) for j, seq in enumerate(batch)]
            _, _, tokens = batch_converter(batch_data)
            results = model(tokens.to(device), repr_layers=[layer], return_contacts=False)
            token_embeds = results["representations"][layer][:, 1:, :]
            arr = []
            for emb in token_embeds:
                if emb.shape[0] >= max_len:
                    arr.append(emb[:max_len].cpu().numpy())
                else:
                    pad = np.zeros((max_len - emb.shape[0], emb.shape[1]), dtype=np.float32)
                    arr.append(np.vstack([emb.cpu().numpy(), pad]))
            all_embeds.append(np.stack(arr))
            
    return torch.from_numpy(np.concatenate(all_embeds, axis=0)).float()

def extract_hf_embeddings(sequences, tokenizer, model, device, batch_size=16, max_len=512):
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="HF Extract"):
            batch = [" ".join(list(seq[:max_len])) for seq in sequences[i:i+batch_size]]
            inputs = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
            outputs = model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            token_embeds = hidden[:, 1:, :]
            arr = []
            for emb in token_embeds:
                if emb.shape[0] >= max_len:
                    arr.append(emb[:max_len].cpu().numpy())
                else:
                    pad = np.zeros((max_len - emb.shape[0], emb.shape[1]), dtype=np.float32)
                    arr.append(np.vstack([emb.cpu().numpy(), pad]))
            all_embeds.append(np.stack(arr))
            
    return torch.from_numpy(np.concatenate(all_embeds, axis=0)).float()

def get_model_and_extract_func(model_type, backend_path, device, max_len=512):
    if model_type == "esmc_300m":
        from esm.models.esmc import ESMC
        client = ESMC.from_pretrained(backend_path).to(device)
        return lambda seqs: extract_esmc_embeddings(
            seqs, client, device, batch_size=16, max_len=max_len, autocast_dtype=(torch.bfloat16 if _is_cuda(device) else None)
        ), 960

    elif model_type == "esm2":
        import esm
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        model = model.to(device)
        return lambda seqs: extract_esm_embeddings(
            seqs, model, alphabet.get_batch_converter(), device, layer=30, batch_size=16, max_len=max_len
        ), 640

    else:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(backend_path)
        model = AutoModel.from_pretrained(backend_path).to(device)
        return lambda seqs: extract_hf_embeddings(
            seqs, tokenizer, model, device, batch_size=8, max_len=max_len
        ), 768

class AntigenDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences, self.labels = list(sequences), list(labels)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]
    def get_data(self): return list(range(len(self.sequences))), self.sequences, self.labels

def collate_fn(batch, max_len=512):
    x, y = zip(*batch)
    x = [xi[:max_len] for xi in x]
    x_padded = torch.stack(x, dim=0) if isinstance(x[0], torch.Tensor) else torch.tensor(x)
    return x_padded, torch.tensor(y)

def test(test_dataset, extract_emb_func, emb_dim, model_path, output_dir, fold, batch_size=32, device="cuda", model_type="AntigenLM"):
    _, test_sequences, test_labels = test_dataset.get_data()
    print("Extracting test embeddings ...")
    test_embeddings = extract_emb_func(test_sequences)
    
    test_loader = DataLoader(list(zip(test_embeddings, test_labels)), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    state_dict = torch.load(model_path, map_location=device)
    # Remove module prefix if it was saved via DataParallel
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    
    model_clf = SoluModel(seq_len=512, in_dim=emb_dim, sa_out=emb_dim, conv_out=emb_dim).to(device)
    model_clf.load_state_dict(state_dict)
    model_clf.eval()

    all_indices, test_preds, test_trues, test_probs = [], [], [], []
    with torch.no_grad():
        for idx, (x_batch, y_batch) in enumerate(test_loader):
            probs = torch.sigmoid(model_clf(x_batch.to(device))[0]).squeeze(-1)
            test_preds.extend((probs >= 0.5).long().cpu().tolist())
            test_trues.extend(y_batch.tolist())
            test_probs.extend(probs.cpu().tolist())
            all_indices.extend(range(idx * batch_size, idx * batch_size + x_batch.shape[0]))

    acc = accuracy_score(test_trues, test_preds)
    prec = precision_score(test_trues, test_preds, zero_division=0)
    rec = recall_score(test_trues, test_preds, zero_division=0)
    f1 = f1_score(test_trues, test_preds, zero_division=0)
    mcc = matthews_corrcoef(test_trues, test_preds)
    
    try: auc = roc_auc_score(test_trues, test_probs)
    except Exception: auc = float("nan")
    
    try: aupr = average_precision_score(test_trues, test_probs)
    except Exception: aupr = float("nan")

    print(f"\n[Fold {fold} EVAL] AUC: {auc:.4f} | AUPR: {aupr:.4f} | Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")

    df_result = pd.DataFrame({'ids': [f"seq{i+1}" for i in all_indices], 'y_true': test_trues, 'y_pred': test_preds, 'y_score': test_probs})
    result_path = P(output_dir, f"{model_type}_test_pred_results_fold{fold}.csv")
    df_result.to_csv(result_path, index=False)
    print(f"Predictions saved to {result_path}")
    
    return {"fold": fold, "auc": auc, "aupr": aupr, "acc": acc, "precision": prec, "recall": rec, "f1": f1, "mcc": mcc}

def main():
    p = argparse.ArgumentParser(description="Eval for Protective Antigen Classification")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--weights_dir", type=str, default="../trained_model/protective_antigen/test")
    p.add_argument("--out_dir", type=str, default="../Result/protective_antigen_bacteria/test")
    p.add_argument("--embed_backend", type=str, choices=["AntigenLM", "esm2", "esmc_300m"], default="AntigenLM")
    p.add_argument("--backend_path", type=str, default="../../LLM/AntigenLM_300M_SS")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=22)
    args = p.parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Model: {args.embed_backend} | Device: {device}")
    extract_emb_func, emb_dim = get_model_and_extract_func(args.embed_backend, args.backend_path, device)

    all_fold_metrics = []
    for fold in range(1, 6):
        test_csv = P(args.data_dir, f"fold_{fold}_test.csv")
        ckpt_path = P(args.weights_dir, f"fold{fold}_seed{args.seed}_{args.embed_backend}.pt")
        
        if not os.path.exists(ckpt_path):
            print(f"[Skip] Weights not found: {ckpt_path}")
            continue

        test_df = pd.read_csv(test_csv)
        fold_metrics = test(
            test_dataset=AntigenDataset(test_df['sequence'], test_df['label']),
            extract_emb_func=extract_emb_func,
            emb_dim=emb_dim,
            model_path=ckpt_path,
            output_dir=args.out_dir,
            fold=fold,
            batch_size=args.batch_size,
            device=device,
            model_type=args.embed_backend
        )
        all_fold_metrics.append(fold_metrics)

    if all_fold_metrics:
        metrics_df = pd.DataFrame(all_fold_metrics)
        avg_row, sd_row = {"fold": "avg"}, {"fold": "sd"}
        
        for col in ["auc", "aupr", "acc", "precision", "recall", "f1", "mcc"]:
            avg_row[col] = float(metrics_df[col].mean())
            sd_row[col] = float(metrics_df[col].std(ddof=0)) if len(metrics_df) > 1 else 0.0
            
        metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_row, sd_row])], ignore_index=True)
        out_csv = P(args.out_dir, f"{args.embed_backend}_test_metrics.csv")
        metrics_df.to_csv(out_csv, index=False)
        print(f"\nFinal CV Metrics saved to: {out_csv}")

if __name__ == '__main__':
    main()
