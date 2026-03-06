import random
import contextlib
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
)
from tqdm import tqdm

from my_model import SoluModel
import esm  # esm需单独安装


# ===================== 模型配置 =====================
SUPPORTED_MODELS = {
    "esmc_300m": {
        "load_func": lambda: None,            # 在 get_model_and_extract_func 里实例化
        "extract_fn": "extract_esmc_embeddings",
        "emb_dim": 960,                       # 已确认，固定 960
        "max_len": 512,
    },
    "esm2": {
        "load_func": lambda: esm.pretrained.esm2_t30_150M_UR50D(),
        "extract_fn": "extract_esm_embeddings",
        "emb_dim": 640,
        "layer": 30
    },
    "esm1b": {
        "load_func": lambda: esm.pretrained.esm1b_t33_650M_UR50S(),
        "extract_fn": "extract_esm_embeddings",
        "emb_dim": 1280,
        "layer": 33
    },
    "esm1v": {
        "load_func": lambda: esm.pretrained.esm1v_t33_650M_UR90S_1(),
        "extract_fn": "extract_esm_embeddings",
        "emb_dim": 1280,
        "layer": 33,
        "max_len": 512,
    },
    "protbert": {
        "model_name": "/data0/chenkai/data/microLM-main/LLM/ProtBert",
        "extract_fn": "extract_hf_embeddings",
        "emb_dim": 1024
    },
    "prott5": {
        "model_name": "/data0/chenkai/data/microLM-main/LLM/ProtT5",
        "extract_fn": "extract_hf_embeddings",
        "emb_dim": 1024
    },
    "antigenLM_withoutSlidingwindow": {
        "model_name": "/home/dataset-local/AntigenLM/LLM/Result_antigenLM_300M_SS_2",
        "extract_fn": "extract_hf_embeddings",
        "emb_dim": 768,
        "max_len": 512,
    },
    "AntigenLM": {
        "model_name": "/home/dataset-local/AntigenLM/LLM/Result_antigenLM_300M_SS_2",
        "extract_fn": "extract_hf_embeddings",
        "emb_dim": 768,
        "max_len": 512,
    },
    "antigenLM_withoutSS_SW": {
        "model_name": "/home/dataset-local/AntigenLM/Other_LLM/AntigenLM_300M_withoutSS_SW",
        "extract_fn": "extract_hf_embeddings",
        "emb_dim": 768,
        "max_len": 512,
    },
    "microLM": {
        "model_name": "/home/dataset-local/AntigenLM/LLM/Result_microLM_300M/0406_034622_rank0",
        "extract_fn": "extract_hf_embeddings",
        "emb_dim": 768,
        "max_len": 512,
    },
    "PathogLM": {
        "model_name": "/home/dataset-local/AntigenLM/LLM/Result_PathogLM_300M_SS",
        "extract_fn": "extract_hf_embeddings",
        "emb_dim": 768,
        "max_len": 512,
    }
}


# ===================== 小工具 =====================
def _is_cuda(dev):
    return (isinstance(dev, torch.device) and dev.type == "cuda") or (isinstance(dev, str) and str(dev).startswith("cuda"))


# ===================== 大模型与抽取函数工厂 =====================
def get_model_and_extract_func(model_type, device):
    info = SUPPORTED_MODELS[model_type]

    # —— ESM C —— #
    if info["extract_fn"] == "extract_esmc_embeddings":
        from esm.models.esmc import ESMC

        client = ESMC.from_pretrained("esmc_300m").to(device)

        def extract_emb_func(seqs):
            return extract_esmc_embeddings(
                seqs,
                client=client,
                device=device if isinstance(device, str) else ("cuda" if torch.cuda.is_available() else "cpu"),
                batch_size=16,
                max_len=info.get("max_len", 512),
                model_max_len=4096,
                autocast_dtype=(torch.bfloat16 if _is_cuda(device) else None),
            )

        return extract_emb_func, info["emb_dim"]

    # —— ESM 系列 —— #
    if info["extract_fn"] == "extract_esm_embeddings":
        import esm as _esm
        model, alphabet = info["load_func"]()
        model = model.to(device)
        batch_converter = alphabet.get_batch_converter()
        def extract_emb_func(seqs):
            return extract_esm_embeddings(
                seqs, model, batch_converter, device,
                layer=info.get("layer", None),
                batch_size=16, max_len=info.get("max_len", 512)
            )
        return extract_emb_func, info["emb_dim"]

    # —— HF 系列 —— #
    elif model_type == "prott5":
        from transformers import T5Tokenizer, T5EncoderModel
        tokenizer = T5Tokenizer.from_pretrained(info["model_name"])
        model = T5EncoderModel.from_pretrained(info["model_name"]).to(device)
        def extract_emb_func(seqs):
            return extract_hf_embeddings(
                seqs, tokenizer, model, device, batch_size=8, max_len=info.get("max_len", 512)
            )
        return extract_emb_func, info["emb_dim"]
    else:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(info["model_name"])
        model = AutoModel.from_pretrained(info["model_name"]).to(device)
        def extract_emb_func(seqs):
            return extract_hf_embeddings(
                seqs, tokenizer, model, device, batch_size=8, max_len=info.get("max_len", 512)
            )
        return extract_emb_func, info["emb_dim"]


# ===================== ESM C (ESMC) 真·批量嵌入提取 =====================
def extract_esmc_embeddings(
    sequences,
    client,                     # ESMC 模型实例
    device,
    batch_size=16,
    max_len=512,
    model_max_len=4096,
    autocast_dtype=torch.bfloat16,  # 不支持 bfloat16 的卡可改 torch.float16 或 None
):
    from esm.sdk.api import ESMProtein, LogitsConfig
    client.eval()
    all_batches = []

    def _to_bld(emb: torch.Tensor) -> torch.Tensor:
        # [L,D] -> [1,L,D]；[B,L,D] 保持
        return emb.unsqueeze(0) if emb.ndim == 2 else emb

    def _pad_trunc_to_maxlen(emb_batch_cpu: torch.Tensor, max_len_: int) -> np.ndarray:
        """emb_batch_cpu: [B, L, D] -> np [B, max_len, D]"""
        B, L, D = emb_batch_cpu.shape
        out_np = []
        for b in range(B):
            e = emb_batch_cpu[b]  # [L, D]
            if e.size(0) >= max_len_:
                out_np.append(e[:max_len_].numpy())
            else:
                pad = np.zeros((max_len_ - e.size(0), D), dtype=np.float32)
                out_np.append(np.vstack([e.numpy(), pad]))
        return np.stack(out_np)

    amp_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if (autocast_dtype is not None and _is_cuda(device)) else contextlib.nullcontext()

    for i in tqdm(range(0, len(sequences), batch_size), desc="ESMC Extract (batched)"):
        # 1) 截断并组 batch
        seqs = [s[: model_max_len - 2] for s in sequences[i:i + batch_size]]
        proteins = [ESMProtein(sequence=s) for s in seqs]

        with torch.no_grad(), amp_ctx:
            ok = False
            # 2) 理想路径：真正 batch encode+logits
            try:
                pt = client.encode(proteins)
                out = client.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
                emb = _to_bld(out.embeddings).to("cpu")   # [B,L,D]
                batch_np = _pad_trunc_to_maxlen(emb, max_len)
                ok = True
            except Exception:
                pass

            # 3) 回退：逐条 encode+logits
            if not ok:
                np_list = []
                for p in proteins:
                    pt1 = client.encode(p)
                    out1 = client.logits(pt1, LogitsConfig(sequence=True, return_embeddings=True))
                    e = _to_bld(out1.embeddings).to("cpu")  # [1,L,D]
                    np_list.append(_pad_trunc_to_maxlen(e, max_len)[0])  # [max_len,D]
                batch_np = np.stack(np_list)  # [B,max_len,D]

        all_batches.append(batch_np)

    all_np = np.concatenate(all_batches, axis=0)  # [N, max_len, D]
    return torch.from_numpy(all_np).float()


# ===================== ESM 系列嵌入提取（沿用） =====================
def extract_esm_embeddings(sequences, model, batch_converter, device, layer=33, batch_size=64, max_len=512, model_max_len=1024):
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="ESM Extract"):
            batch = sequences[i:i+batch_size]
            batch = [seq[:model_max_len-2] for seq in batch]
            batch_data = [("seq%d" % j, seq) for j, seq in enumerate(batch)]
            _, _, tokens = batch_converter(batch_data)
            tokens = tokens.to(device)
            results = model(tokens, repr_layers=[layer], return_contacts=False)
            token_embeds = results["representations"][layer][:, 1:, :]
            arr = []
            for emb in token_embeds:
                cur_len = emb.shape[0]
                if cur_len >= max_len:
                    arr.append(emb[:max_len].cpu().numpy())
                else:
                    pad = np.zeros((max_len - cur_len, emb.shape[1]), dtype=np.float32)
                    arr.append(np.vstack([emb.cpu().numpy(), pad]))
            arr = np.stack(arr)
            all_embeds.append(arr)
    all_embeds = np.concatenate(all_embeds, axis=0)
    return torch.from_numpy(all_embeds).float()


# ------------------ HuggingFace 模型嵌入提取（沿用小修） ------------------
def extract_hf_embeddings(sequences, tokenizer, model, device, batch_size=16, max_len=512):
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="HF Extract"):
            batch = [" ".join(list(seq[:max_len])) for seq in sequences[i:i+batch_size]]
            inputs = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            token_embeds = hidden[:, 1:, :]  # [B, L-1, D]
            arr = []
            for emb in token_embeds:
                cur_len = emb.shape[0]
                if cur_len >= max_len:
                    arr.append(emb[:max_len].cpu().numpy())
                else:
                    pad = np.zeros((max_len-cur_len, emb.shape[1]), dtype=np.float32)
                    arr.append(np.vstack([emb.cpu().numpy(), pad]))
            arr = np.stack(arr)
            all_embeds.append(arr)
    all_embeds = np.concatenate(all_embeds, axis=0)
    return torch.from_numpy(all_embeds).float()


# ===================== 数据分割 =====================
def split_train_val(csv_path, val_ratio=0.15, seed=22):
    df = pd.read_csv(csv_path)
    pos_df = df[df['label'] == 1].reset_index(drop=True)
    neg_df = df[df['label'] == 0].reset_index(drop=True)
    n_pos = len(pos_df)
    n_val_pos = int(n_pos * val_ratio)
    n_val_neg = min(n_val_pos * 10, len(neg_df))
    random.seed(seed)
    val_pos_idx = random.sample(list(pos_df.index), n_val_pos)
    val_neg_idx = random.sample(list(neg_df.index), n_val_neg)
    val_df = pd.concat([pos_df.loc[val_pos_idx], neg_df.loc[val_neg_idx]]).sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = pd.concat([pos_df.drop(val_pos_idx), neg_df.drop(val_neg_idx)]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"训练集: 正{len(pos_df)-n_val_pos} 负{len(neg_df)-n_val_neg} 总{len(train_df)}")
    print(f"验证集: 正{n_val_pos} 负{n_val_neg} 总{len(val_df)}")
    print(f"验证集正负样本比例: {n_val_pos}:{n_val_neg} = {n_val_pos/max(n_val_neg,1):.2f}")
    return train_df, val_df


# ===================== Dataset =====================
class AntigenDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = list(sequences)
        self.labels = list(labels)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]
    def get_data(self): return list(range(len(self.sequences))), self.sequences, self.labels


# ===================== Losses =====================
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


# ===================== 训练主流程 =====================
def train(train_dataset, val_dataset, extract_emb_func, emb_dim, epochs=15, batch_size=32, device="cuda", model_type="esmc_300m", fold=1):
    _, train_sequences, train_labels = train_dataset.get_data()
    _, val_sequences, val_labels = val_dataset.get_data()
    save_path = f"/home/dataset-local/AntigenLM/Downstream/trained_model/protective_antigen/Ablation/{model_type}_best_model_fold{fold}.pt"

    # 1) 预计算 embeddings
    print("Extracting train embeddings ...")
    train_embeddings = extract_emb_func(train_sequences) 
    print(train_embeddings.shape)  # [N, L, 960]
    print("Extracting val embeddings ...")
    val_embeddings = extract_emb_func(val_sequences)      # [M, L, 960]

    train_data = list(zip(train_embeddings, train_labels))
    val_data = list(zip(val_embeddings, val_labels))

    labels_flat = np.array(train_labels)
    class_sample_count = np.bincount(labels_flat.astype(int))
    weights = 1. / class_sample_count
    sample_weights = weights[labels_flat.astype(int)]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

    # 2) 构建模型（固定 in_dim=960；优先传入 num_attention_heads=20）
    model_clf = None
    for param_name in ("num_attention_heads", "n_heads", "heads"):
        try:
            kw = {param_name: 20}  # 960/20 = 48
            model_clf = SoluModel(seq_len=512, in_dim=emb_dim, sa_out=emb_dim, conv_out=emb_dim, **kw).to(device)
            break
        except TypeError:
            continue
    if model_clf is None:
        # 没有可用的 heads 参数名，就按默认构建（确保 my_model.py 默认 heads 能整除 960）
        model_clf = SoluModel(seq_len=512, in_dim=emb_dim, sa_out=emb_dim, conv_out=emb_dim).to(device)

    if torch.cuda.device_count() > 1 and _is_cuda(device):
        model_clf = nn.DataParallel(model_clf)

    optimizer = torch.optim.Adam(model_clf.parameters(), lr=1e-3)
    contrastive_criterion = ContrastiveLoss(margin=1)
    pos = int(np.sum(labels_flat == 1)); neg = int(np.sum(labels_flat == 0))
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # —— 训练 —— #
        model_clf.train()
        total_loss = 0.0
        train_preds, train_trues = [], []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)                           # [B, L, 960]
            y_batch = torch.as_tensor(y_batch, dtype=torch.float, device=device).unsqueeze(1)

            cls_out, emb_out = model_clf(x_batch)                  # logits, embeddings
            bce_loss = criterion(cls_out, y_batch)
            contrastive_loss = contrastive_criterion(emb_out, y_batch.squeeze())
            loss = bce_loss + 0.2 * contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

            probs = torch.sigmoid(cls_out).squeeze(-1)             # 在 sigmoid 后阈值
            preds = (probs >= 0.5).long()
            train_preds.extend(preds.detach().cpu().tolist())
            train_trues.extend(y_batch.long().cpu().tolist())

        train_acc = accuracy_score(train_trues, train_preds)
        avg_loss = total_loss / len(train_loader.dataset)

        # —— 验证 —— #
        model_clf.eval()
        val_preds, val_trues, val_probs = [], [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = torch.as_tensor(y_batch, dtype=torch.float, device=device).unsqueeze(1)
                cls_out, _ = model_clf(x_batch)
                probs = torch.sigmoid(cls_out).squeeze(-1)
                preds = (probs >= 0.5).long()
                val_preds.extend(preds.cpu().tolist())
                val_trues.extend(y_batch.cpu().tolist())
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
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f} | AUC: {roc_auc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_clf.state_dict(), save_path)
            print(f">>> New best model saved: {save_path} (Val Acc: {best_acc:.4f})")

    return save_path


# ===================== 主程序入口 =====================
if __name__ == '__main__':
    model_type = "AntigenLM"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用模型: {model_type} | 设备: {device}")

    data_dir = "/home/dataset-local/AntigenLM/Downstream/protective_antigen/data/protective_antigen_bacteria_overall"
    extract_emb_func, emb_dim = get_model_and_extract_func(model_type, device)  # emb_dim 将是 960

    for fold in range(1, 6):
        train_csv = f"{data_dir}/fold_{fold}_train.csv"
        train_df, val_df = split_train_val(train_csv, val_ratio=0.15)
        print(f"[Fold {fold}] 训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}")
        print(len(train_df[train_df['label'] == 1]), len(val_df[val_df['label'] == 1]))
        print(len(train_df[train_df['label'] == 0]), len(val_df[val_df['label'] == 0]))

        train_dataset = AntigenDataset(train_df['sequence'], train_df['label'])
        val_dataset = AntigenDataset(val_df['sequence'], val_df['label'])

        train(
            train_dataset,
            val_dataset,
            extract_emb_func=extract_emb_func,
            emb_dim=emb_dim,
            epochs=50,
            batch_size=32,
            device=device,
            model_type=model_type,
            fold=fold
        )
