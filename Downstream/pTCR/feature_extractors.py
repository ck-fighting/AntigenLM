from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def antigenLM_extract(sequences, model_name_or_path, device, max_len, batch_size=64):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Extract {model_name_or_path.split('/')[-1]}"):
            batch_seqs = sequences[i:i + batch_size]
            batch_spaced = [" ".join(list(seq)) for seq in batch_seqs]
            tokens = tokenizer(batch_spaced, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (B, L, D)
            for j in range(last_hidden.size(0)):
                valid_len = attention_mask[j].sum().item()
                emb = last_hidden[j, :valid_len].cpu()
                embeddings.append(emb)
    maxL = max_len
    dim = embeddings[0].shape[1]
    emb_padded = torch.stack([F.pad(e, (0, 0, 0, maxL - e.size(0))) for e in embeddings])
    return emb_padded


def _pick_amp_dtype(prefer=torch.bfloat16):
    if prefer is torch.bfloat16 and (
        not hasattr(torch.cuda, "is_bf16_supported") or not torch.cuda.is_bf16_supported()
    ):
        return torch.float16
    return prefer


def autocast_cuda(dtype=None):
    if dtype is None:
        dtype = _pick_amp_dtype()

    try:
        return torch.autocast("cuda", dtype=dtype)
    except Exception:
        pass

    try:
        return torch.cuda.amp.autocast(dtype=dtype)
    except Exception:
        class _NullCtx:
            def __enter__(self): return None
            def __exit__(self, *args): return False
        return _NullCtx()


@torch.inference_mode()
def extract_esm2_embeddings(
    sequences: List[str],
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cuda",
    max_len: int = 256,
    batch_size: int = 512,
    use_amp: bool = True,
    amp_dtype=torch.bfloat16,
    deduplicate: bool = True,
    compile_model: bool = False,
) -> torch.Tensor:
    seqs = [s[:max_len] for s in sequences]
    if deduplicate:
        uniq2idx, uniq_list, rev_index = {}, [], []
        for s in seqs:
            if s not in uniq2idx:
                uniq2idx[s] = len(uniq_list)
                uniq_list.append(s)
            rev_index.append(uniq2idx[s])
        work_seqs = uniq_list
    else:
        work_seqs = seqs
        rev_index = list(range(len(seqs)))

    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if compile_model:
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception:
            pass

    batch_converter = alphabet.get_batch_converter()
    repr_layer = getattr(model, "num_layers", None)
    if repr_layer is None:
        repr_layer = 33 if "t33" in model_name else 36

    out_cpu = None
    D_dim = None

    N_work = len(work_seqs)
    idx = 0
    while idx < N_work:
        sub = work_seqs[idx: idx + batch_size]
        idx += len(sub)

        batch = [(str(j), s) for j, s in enumerate(sub)]
        labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.pin_memory().to(device, non_blocking=True)

        amp_dtype = _pick_amp_dtype(torch.bfloat16) if use_amp else None
        if amp_dtype is None:
            out = model(batch_tokens, repr_layers=[repr_layer])
        else:
            with autocast_cuda(dtype=amp_dtype):
                out = model(batch_tokens, repr_layers=[repr_layer])

        reps = out["representations"][repr_layer]
        B, Ltok, D_now = reps.shape

        if D_dim is None:
            D_dim = D_now
            total_N = len(work_seqs)
            out_cpu = torch.empty((total_N, max_len, D_dim), dtype=torch.float32, pin_memory=True)

        emb_list = []
        for j in range(B):
            L = min(len(batch_strs[j]), max_len)
            e = reps[j, 1:1 + L, :]
            if L < max_len:
                e = F.pad(e, (0, 0, 0, max_len - L))
            emb_list.append(e)
        emb_batch = torch.stack(emb_list, dim=0)

        emb_cpu = emb_batch.to("cpu", non_blocking=True)
        if emb_cpu.dtype != torch.float32:
            emb_cpu = emb_cpu.float()
        start = idx - B
        out_cpu[start: start + B].copy_(emb_cpu)

    if deduplicate:
        final = torch.empty((len(seqs), max_len, D_dim), dtype=torch.float32)
        for orig_i, uniq_i in enumerate(rev_index):
            final[orig_i].copy_(out_cpu[uniq_i])
        return final
    else:
        return out_cpu


def extract_esmc_embeddings(
    sequences,
    client,
    device="cuda",
    batch_size=128,
    max_len=15,
    model_max_len=256,
    deduplicate=True,
    use_amp=True,
    amp_dtype=torch.float16,
):
    from esm.sdk.api import ESMProtein, LogitsConfig

    client.eval()

    seqs = [s[: max(0, model_max_len - 2)] for s in sequences]
    if deduplicate:
        uniq2idx, uniq_list, rev_index = {}, [], []
        for s in seqs:
            if s not in uniq2idx:
                uniq2idx[s] = len(uniq_list)
                uniq_list.append(s)
            rev_index.append(uniq2idx[s])
        work_seqs = uniq_list
    else:
        work_seqs = seqs
        rev_index = list(range(len(seqs)))

    N_work = len(work_seqs)
    proteins = [ESMProtein(sequence=s) for s in work_seqs]

    out_cpu = None
    D_dim = None

    idx = 0
    while idx < N_work:
        sub = proteins[idx: idx + batch_size]
        idx += len(sub)

        emb_list = []
        for p in sub:
            pt = client.encode(p)
            if use_amp:
                with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                    out = client.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
            else:
                out = client.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))

            e = out.embeddings
            if e.ndim == 2:
                e = e.unsqueeze(0)
            if str(e.device) != str(device):
                e = e.to(device)

            if use_amp and e.dtype != amp_dtype:
                e = e.to(amp_dtype)

            emb_list.append(e)

        emb_cat = torch.cat(emb_list, dim=0)
        B, Lvar, D_now = emb_cat.shape

        if D_dim is None:
            D_dim = D_now
            total_N = len(work_seqs)
            out_cpu = torch.empty((total_N, max_len, D_dim), dtype=torch.float32, pin_memory=True)

        if Lvar >= max_len:
            emb_pad = emb_cat[:, :max_len, :]
        else:
            emb_pad = F.pad(emb_cat, (0, 0, 0, max_len - Lvar))

        emb_cpu = emb_pad.to("cpu", non_blocking=True)
        if emb_cpu.dtype != torch.float32:
            emb_cpu = emb_cpu.float()

        start = idx - B
        out_cpu[start: start + B].copy_(emb_cpu)

    if deduplicate:
        final = torch.empty((len(seqs), max_len, D_dim), dtype=torch.float32)
        for orig_i, uniq_i in enumerate(rev_index):
            final[orig_i].copy_(out_cpu[uniq_i])
        return final
    else:
        return out_cpu
