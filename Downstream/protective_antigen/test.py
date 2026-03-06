import os
import torch
import contextlib
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
from my_model import SoluModel
import numpy as np
import esm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

# ========= 你的大模型配置（与训练保持一致） =========
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
    "antigenLM_withoutSS": {
        "model_name": "/home/dataset-local/AntigenLM/Other_LLM/AntigenLM_300M_withoutSS",
        "extract_fn": "extract_hf_embeddings",
        "emb_dim": 768,
        "max_len": 512,
    },
    "antigenLM_withoutSlidingwindow": {
        "model_name": "/home/dataset-local/AntigenLM/Other_LLM/AntigenLM_300M_withoutSlidingwindow",
        "extract_fn": "extract_hf_embeddings",
        "emb_dim": 768,
        "max_len": 512,
    },
    "AntigenLM": {
        "model_name": "/home/dataset-local/AntigenLM/Other_LLM/AntigenLM_300M_withoutSlidingwindow",
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
    "antigenLM_withoutSS_SW": {
        "model_name": "/home/dataset-local/AntigenLM/Other_LLM/AntigenLM_300M_withoutSS_SW",
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

# ========= 通用Embedding提取函数 =========
def extract_hf_embeddings(sequences, tokenizer, model, device, batch_size=16, max_len=512):
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="HF Extract"):
            batch = [ " ".join(list(seq[:max_len])) for seq in sequences[i:i+batch_size] ]
            inputs = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
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

# ========= 自动选择模型和Embedding提取 =========
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
            )

        return extract_emb_func, info["emb_dim"]

    # —— ESM 系列 —— #
    if info["extract_fn"] == "extract_esm_embeddings":
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


    for i in tqdm(range(0, len(sequences), batch_size), desc="ESMC Extract (batched)"):
        # 1) 截断并组 batch
        seqs = [s[: model_max_len - 2] for s in sequences[i:i + batch_size]]
        proteins = [ESMProtein(sequence=s) for s in seqs]

        with torch.no_grad():
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

# ========= Dataset & collate =========
class AntigenDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = list(sequences)
        self.labels = list(labels)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
    def get_data(self):
        return list(range(len(self.sequences))), self.sequences, self.labels

def collate_fn(batch, max_len=512):
    x, y = zip(*batch)
    x = [xi[:max_len] for xi in x]
    x_padded = torch.stack(x, dim=0) if isinstance(x[0], torch.Tensor) else torch.tensor(x)
    y = torch.tensor(y)
    return x_padded, y

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# ========= 推理与评估 =========
def test(test_dataset, extract_emb_func, emb_dim, model_path, test_df, output_dir, batch_size=32, device="cuda", model_type="prott5", fold=1):
    _, test_sequences, test_labels = test_dataset.get_data()
    print("提取测试集embedding ...")
    test_embeddings = extract_emb_func(test_sequences)
    print("测试集embedding提取完成:", test_embeddings.shape)
    test_data = list(zip(test_embeddings, test_labels))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    state_dict = torch.load(model_path, map_location=device)
    state_dict = remove_module_prefix(state_dict)
    model_clf = SoluModel(seq_len=512, in_dim=emb_dim, sa_out=emb_dim, conv_out=emb_dim).to(device)
    model_clf.load_state_dict(state_dict)
    model_clf.eval()
    all_indices = []
    test_preds, test_trues, test_probs, features = [], [], [], []
    with torch.no_grad():
        for idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.float().to(device).unsqueeze(1)
            cls_out, out = model_clf(x_batch)
            features.append(out.detach().cpu().numpy())

            probs = torch.sigmoid(cls_out).squeeze(-1)
            preds = (probs >= 0.5).long()
            batch_size_now = x_batch.shape[0]
            test_preds.extend(preds.cpu().tolist())
            test_trues.extend(y_batch.cpu().tolist())
            test_probs.extend(probs.cpu().tolist())
            # 记录索引，便于回溯
            all_indices.extend(range(idx * batch_size, idx * batch_size + batch_size_now))
        # feature_matrix = np.concatenate(features, axis=0)
        # np.save(f"{model_type}_features.npy", feature_matrix)
        # print(f"✅ 保存特征 {model_type}:", feature_matrix.shape)

    acc = accuracy_score(test_trues, test_preds)
    prec = precision_score(test_trues, test_preds, zero_division=0)
    rec = recall_score(test_trues, test_preds, zero_division=0)
    f1 = f1_score(test_trues, test_preds, zero_division=0)
    mcc = matthews_corrcoef(test_trues, test_preds)
    try:
        auc = roc_auc_score(test_trues, test_probs)
    except Exception:
        auc = float("nan")
    try:
        aupr = average_precision_score(test_trues, test_probs)
    except Exception:
        aupr = float("nan")
    print("【Test集评估】")
    print(f" AUC: {auc:.4f} | AUPR: {aupr:.4f} | Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f} | ")
    ids = [f"seq{i+1}" for i in all_indices]

    df_result = pd.DataFrame({
        'ids'  : ids,
        'y_true': test_trues,
        'y_pred': test_preds,
        'y_score': test_probs
    })
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{model_type}_test_pred_results_fold{fold}.csv")
    df_result.to_csv(result_path, index=False)
    print(f"✅ 已保存预测结果到 {result_path}")
    return {
        "fold": fold,
        "auc": auc,
        "aupr": aupr,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
    }


def main():
    model_type = "AntigenLM"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前模型: {model_type}")

    extract_emb_func, emb_dim = get_model_and_extract_func(model_type, device)

    data_dir = "/home/dataset-local/AntigenLM/Downstream/protective_antigen/data/protective_antigen_bacteria_overall"
    model_dir = "/home/dataset-local/AntigenLM/Downstream/trained_model/protective_antigen/Ablation"
    output_dir = "/home/dataset-local/AntigenLM/Downstream/Result/protective_antigen_bacteria/Ablation"

    all_fold_metrics = []
    for fold in range(1, 6):
        test_csv = f"{data_dir}/fold_{fold}_test.csv"
        model_ckpt_path = f"{model_dir}/{model_type}_best_model_fold{fold}.pt"

        test_df = pd.read_csv(test_csv)
        test_dataset = AntigenDataset(test_df['sequence'], test_df['label'])

        fold_metrics = test(
            test_dataset,
            extract_emb_func=extract_emb_func,
            emb_dim=emb_dim,
            model_path=model_ckpt_path,
            test_df=test_df,
            output_dir=output_dir,
            batch_size=32,
            device=device,
            model_type=model_type,
            fold=fold
        )
        if fold_metrics is not None:
            all_fold_metrics.append(fold_metrics)

    if all_fold_metrics:
        metrics_df = pd.DataFrame(all_fold_metrics)
        avg_row = {"fold": "avg"}
        sd_row = {"fold": "sd"}
        for col in ["auc", "aupr", "acc", "precision", "recall", "f1", "mcc"]:
            avg_row[col] = float(metrics_df[col].mean())
            sd_row[col] = float(metrics_df[col].std(ddof=0))
        metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_row, sd_row])], ignore_index=True)
        os.makedirs(output_dir, exist_ok=True)
        out_csv = os.path.join(output_dir, f"{model_type}_test_metrics.csv")
        metrics_df.to_csv(out_csv, index=False)
        print(f"✅ 已保存折间评估到 {out_csv}")

if __name__ == '__main__':
    main()
