import os
import random
import argparse
import contextlib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
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

def split_train_val(csv_path, val_ratio=0.15, seed=22):
    df = pd.read_csv(csv_path)
    pos_df, neg_df = df[df['label'] == 1].reset_index(drop=True), df[df['label'] == 0].reset_index(drop=True)
    n_val_pos, n_val_neg = int(len(pos_df) * val_ratio), min(int(len(pos_df) * val_ratio) * 10, len(neg_df))
    
    random.seed(seed)
    val_pos_idx, val_neg_idx = random.sample(list(pos_df.index), n_val_pos), random.sample(list(neg_df.index), n_val_neg)
    
    val_df = pd.concat([pos_df.loc[val_pos_idx], neg_df.loc[val_neg_idx]]).sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = pd.concat([pos_df.drop(val_pos_idx), neg_df.drop(val_neg_idx)]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Train set: Pos={len(pos_df)-n_val_pos} Neg={len(neg_df)-n_val_neg} Total={len(train_df)}")
    print(f"Val set: Pos={n_val_pos} Neg={n_val_neg} Total={len(val_df)}")
    return train_df, val_df

class AntigenDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences, self.labels = list(sequences), list(labels)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]
    def get_data(self): return list(range(len(self.sequences))), self.sequences, self.labels

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.euc_dist = nn.PairwiseDistance(p=2)
        
    def forward(self, out, label):
        B = out.size(0)
        loss, cnt = 0.0, 0
        for i in range(B):
            for j in range(i+1, B):
                dist = self.euc_dist(out[i].unsqueeze(0), out[j].unsqueeze(0))
                if label[i] == label[j]:
                    loss += dist ** 2
                else:
                    loss += torch.clamp(self.margin - dist, min=0.0) ** 2
                cnt += 1
        return loss / (cnt + 1e-6)

def train(train_dataset, val_dataset, extract_emb_func, emb_dim, epochs, batch_size, device, save_path):
    _, train_sequences, train_labels = train_dataset.get_data()
    _, val_sequences, val_labels = val_dataset.get_data()

    print("Extracting train embeddings ...")
    train_embeddings = extract_emb_func(train_sequences) 
    print("Extracting val embeddings ...")
    val_embeddings = extract_emb_func(val_sequences)

    labels_flat = np.array(train_labels)
    class_sample_count = np.bincount(labels_flat.astype(int))
    sample_weights = (1. / class_sample_count)[labels_flat.astype(int)]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(list(zip(train_embeddings, train_labels)), batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(list(zip(val_embeddings, val_labels)), batch_size=batch_size, shuffle=False, drop_last=True)

    model_clf = SoluModel(seq_len=512, in_dim=emb_dim, sa_out=emb_dim, conv_out=emb_dim).to(device)
    if torch.cuda.device_count() > 1 and _is_cuda(device):
        model_clf = nn.DataParallel(model_clf)

    optimizer = torch.optim.Adam(model_clf.parameters(), lr=1e-3)
    contrastive_criterion = ContrastiveLoss(margin=1)
    
    pos, neg = int(np.sum(labels_flat == 1)), int(np.sum(labels_flat == 0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1)], dtype=torch.float, device=device))

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model_clf.train()
        total_loss, train_preds, train_trues = 0.0, [], []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = torch.as_tensor(y_batch, dtype=torch.float, device=device).unsqueeze(1)

            cls_out, emb_out = model_clf(x_batch)
            loss = criterion(cls_out, y_batch) + 0.2 * contrastive_criterion(emb_out, y_batch.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            train_preds.extend((torch.sigmoid(cls_out).squeeze(-1) >= 0.5).long().cpu().tolist())
            train_trues.extend(y_batch.long().cpu().tolist())

        train_acc = accuracy_score(train_trues, train_preds)
        avg_loss = total_loss / len(train_loader.dataset)

        model_clf.eval()
        val_preds, val_trues, val_probs = [], [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                probs = torch.sigmoid(model_clf(x_batch.to(device))[0]).squeeze(-1)
                val_preds.extend((probs >= 0.5).long().cpu().tolist())
                val_trues.extend(y_batch.tolist())
                val_probs.extend(probs.cpu().tolist())

        val_acc = accuracy_score(val_trues, val_preds)
        prec = precision_score(val_trues, val_preds, zero_division=0)
        rec = recall_score(val_trues, val_preds, zero_division=0)
        f1 = f1_score(val_trues, val_preds, zero_division=0)
        mcc = matthews_corrcoef(val_trues, val_preds)
        
        try:
            roc_auc = roc_auc_score(val_trues, val_probs)
        except Exception:
            roc_auc = float("nan")

        print(
            f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f} | AUC: {roc_auc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_clf.module.state_dict() if isinstance(model_clf, nn.DataParallel) else model_clf.state_dict(), save_path)
            print(f">>> New best model saved: {save_path} (Val Acc: {best_acc:.4f})")

    return save_path

def main():
    p = argparse.ArgumentParser(description="Training for Protective Antigen Classification")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="../trained_model/protective_antigen/test")
    p.add_argument("--embed_backend", type=str, choices=["AntigenLM", "esm2", "esmc_300m"], default="AntigenLM")
    p.add_argument("--backend_path", type=str, default="../../LLM/AntigenLM_300M_SS", help="Path to AntigentLM or esmc")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=22)
    args = p.parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Model: {args.embed_backend} | Device: {device}")
    extract_emb_func, emb_dim = get_model_and_extract_func(args.embed_backend, args.backend_path, device)

    for fold in range(1, 6):
        train_csv = P(args.data_dir, f"fold_{fold}_train.csv")
        if not os.path.exists(train_csv):
            print(f"[Skip] CSV not found: {train_csv}")
            continue
            
        train_df, val_df = split_train_val(train_csv, val_ratio=0.15, seed=args.seed)
        print(f"========== Fold {fold} ==========")
        
        train(
            train_dataset=AntigenDataset(train_df['sequence'], train_df['label']),
            val_dataset=AntigenDataset(val_df['sequence'], val_df['label']),
            extract_emb_func=extract_emb_func,
            emb_dim=emb_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            save_path=P(args.save_dir, f"fold{fold}_seed{args.seed}_{args.embed_backend}.pt")
        )

if __name__ == '__main__':
    main()
