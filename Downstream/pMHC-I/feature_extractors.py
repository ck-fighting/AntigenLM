import contextlib
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM


def load_antigenlm_tokenizer(model_name_or_path):
    return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


def load_antigenlm_model(model_name_or_path, device):
    try:
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    return model.to(device)


def antigenlm_last_hidden(model, input_ids, attention_mask):
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_logits=False,
        )
    except TypeError:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is not None:
        return last_hidden

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]

    if isinstance(outputs, (tuple, list)):
        if outputs and torch.is_tensor(outputs[0]) and outputs[0].dim() == 3:
            return outputs[0]
        for item in reversed(outputs):
            if isinstance(item, (tuple, list)) and item and torch.is_tensor(item[-1]):
                return item[-1]

    raise ValueError("Could not find token hidden states in AntigenLM model output")


def antigenLM_extract(sequences, model_name_or_path, device, max_len, batch_size=64):
    tokenizer = load_antigenlm_tokenizer(model_name_or_path)
    model = load_antigenlm_model(model_name_or_path, device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Extract {model_name_or_path.split('/')[-1]}"):
            batch_seqs = [seq.rstrip("-")[:max_len] for seq in sequences[i:i + batch_size]]
            tokens = tokenizer(
                batch_seqs,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_len + 2,
                add_special_tokens=True,
                return_special_tokens_mask=True,
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            last_hidden = antigenlm_last_hidden(model, input_ids, attention_mask)  # (B, L, D)
            special_tokens_mask = tokens["special_tokens_mask"].to(device).bool()
            for j, seq in enumerate(batch_seqs):
                residue_mask = attention_mask[j].bool() & (~special_tokens_mask[j])
                emb = last_hidden[j][residue_mask][:max_len].cpu()
                valid_len = min(len(seq), max_len, emb.size(0))
                emb = emb[:valid_len]
                if valid_len < max_len:
                    emb = F.pad(emb, (0, 0, 0, max_len - valid_len))
                embeddings.append(emb)
    return torch.stack([F.pad(e, (0, 0, 0, max_len - e.size(0))) for e in embeddings])


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


def _is_cuda(dev):
    return (isinstance(dev, torch.device) and dev.type == "cuda") or (
        isinstance(dev, str) and str(dev).startswith("cuda")
    )


def _pad_trunc_token_embeddings(emb, max_len):
    emb = emb[:max_len].detach().cpu().float()
    if emb.size(0) >= max_len:
        return emb.numpy()
    pad = torch.zeros((max_len - emb.size(0), emb.size(1)), dtype=emb.dtype)
    return torch.cat([emb, pad], dim=0).numpy()


def esmc_forward_embeddings(client, sequence_tokens):
    sequence_id = sequence_tokens == client.tokenizer.pad_token_id
    chain_id = torch.ones(sequence_tokens.shape, dtype=torch.int64, device=sequence_tokens.device)
    x = client.embed(sequence_tokens)
    for block in client.transformer.blocks:
        x = block(x, sequence_id, None, None, chain_id)
    return client.transformer.norm(x)


def _resolve_local_esm2_path(model_name: str) -> str:
    candidates = []
    if model_name:
        candidates.append(model_name)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.extend([
        os.path.join(repo_root, "LLM", "esm2_650M"),
    ])
    for path in candidates:
        if path and os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            return path
    return ""


@torch.inference_mode()
def extract_hf_esm2_embeddings(
    sequences: List[str],
    model_name_or_path: str,
    device: str = "cuda",
    max_len: int = 15,
    batch_size: int = 64,
    use_amp: bool = True,
    amp_dtype=torch.bfloat16,
    deduplicate: bool = True,
) -> torch.Tensor:
    seqs = [s.rstrip("-")[:max_len] for s in sequences]
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

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, add_pooling_layer=False).to(device).eval()
    out_cpu = None
    d_dim = None

    iterator = range(0, len(work_seqs), batch_size)
    for start in tqdm(iterator, desc=f"Extract {os.path.basename(model_name_or_path.rstrip('/'))}"):
        batch_seqs = work_seqs[start:start + batch_size]
        tokens = tokenizer(
            batch_seqs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_len + 2,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        model_inputs = {
            key: value.to(device)
            for key, value in tokens.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        amp = _pick_amp_dtype(amp_dtype) if use_amp else None
        if amp is None:
            outputs = model(**model_inputs)
        else:
            with autocast_cuda(dtype=amp):
                outputs = model(**model_inputs)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        attention_mask = model_inputs["attention_mask"].bool()
        special_tokens_mask = tokens["special_tokens_mask"].to(device).bool()

        batch_embeddings = []
        for j, seq in enumerate(batch_seqs):
            residue_mask = attention_mask[j] & (~special_tokens_mask[j])
            emb = hidden[j][residue_mask][:max_len]
            valid_len = min(len(seq), max_len, emb.size(0))
            emb = emb[:valid_len]
            if valid_len < max_len:
                emb = F.pad(emb, (0, 0, 0, max_len - valid_len))
            batch_embeddings.append(emb)
        emb_batch = torch.stack(batch_embeddings, dim=0).detach().cpu().float()
        if d_dim is None:
            d_dim = int(emb_batch.shape[-1])
            out_cpu = torch.empty((len(work_seqs), max_len, d_dim), dtype=torch.float32)
        out_cpu[start:start + len(batch_seqs)].copy_(emb_batch)
        del tokens, model_inputs, outputs, hidden, attention_mask, special_tokens_mask, emb_batch

    if deduplicate:
        final = torch.empty((len(seqs), max_len, d_dim), dtype=torch.float32)
        for orig_i, uniq_i in enumerate(rev_index):
            final[orig_i].copy_(out_cpu[uniq_i])
        return final
    return out_cpu


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
    local_path = _resolve_local_esm2_path(model_name)
    if local_path:
        return extract_hf_esm2_embeddings(
            sequences,
            model_name_or_path=local_path,
            device=device,
            max_len=max_len,
            batch_size=batch_size,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            deduplicate=deduplicate,
        )

    seqs = [s.rstrip("-")[:max_len] for s in sequences]
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

    try:
        import esm.pretrained as esm_pretrained
    except Exception as e:
        raise ImportError(
            "ESM2 backend needs a local HuggingFace ESM2 path or fair-esm. "
            "Default local ESM2 candidates were not found."
        ) from e
    if not hasattr(esm_pretrained, "load_model_and_alphabet"):
        raise ImportError(
            "Current `esm` package does not expose fair-esm `load_model_and_alphabet`. "
            "Use a local HuggingFace ESM2 path or a separate fair-esm environment."
        )
    model, alphabet = esm_pretrained.load_model_and_alphabet(model_name)
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
        _, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.pin_memory().to(device, non_blocking=True)

        amp_dtype = _pick_amp_dtype(torch.bfloat16) if use_amp else None
        if amp_dtype is None:
            out = model(batch_tokens, repr_layers=[repr_layer])
        else:
            with autocast_cuda(dtype=amp_dtype):
                out = model(batch_tokens, repr_layers=[repr_layer])

        reps = out["representations"][repr_layer]
        B, _, D_now = reps.shape

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
    device,
    batch_size=16,
    max_len=512,
    model_max_len=512,
    autocast_dtype=torch.bfloat16,
):
    from esm.sdk.api import ESMProtein, LogitsConfig

    client.eval()
    seqs = [seq[: model_max_len - 2] for seq in sequences]
    all_batches = []

    amp_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if (
        autocast_dtype is not None and _is_cuda(device)
    ) else contextlib.nullcontext()

    if hasattr(client, "_tokenize"):
        with torch.no_grad(), amp_ctx:
            for i in tqdm(range(0, len(seqs), batch_size), desc="ESMC Extract"):
                batch_seqs = seqs[i:i + batch_size]
                tokens = client._tokenize(batch_seqs).to(device)
                embeddings = esmc_forward_embeddings(client, tokens)
                batch_embeds = []
                for j, seq in enumerate(batch_seqs):
                    token_embeds = embeddings[j]
                    if token_embeds.size(0) >= len(seq) + 2:
                        token_embeds = token_embeds[1:1 + len(seq)]
                    else:
                        token_embeds = token_embeds[:len(seq)]
                    batch_embeds.append(_pad_trunc_token_embeddings(token_embeds, max_len))
                all_batches.append(np.stack(batch_embeds))
                del tokens, embeddings
        return torch.from_numpy(np.concatenate(all_batches, axis=0)).float()

    with torch.no_grad(), amp_ctx:
        for i in tqdm(range(0, len(seqs), batch_size), desc="ESMC Extract"):
            batch_embeds = []
            for seq in seqs[i:i + batch_size]:
                protein_tensor = client.encode(ESMProtein(sequence=seq))
                output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True),
                )
                token_embeds = output.embeddings
                if token_embeds.ndim == 3:
                    token_embeds = token_embeds[0]
                if token_embeds.size(0) >= len(seq) + 2:
                    token_embeds = token_embeds[1:1 + len(seq)]
                else:
                    token_embeds = token_embeds[:len(seq)]
                batch_embeds.append(_pad_trunc_token_embeddings(token_embeds, max_len))
            all_batches.append(np.stack(batch_embeds))

    return torch.from_numpy(np.concatenate(all_batches, axis=0)).float()
