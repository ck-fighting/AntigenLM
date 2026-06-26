import json
import os
from contextlib import nullcontext
from glob import glob
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ESM2_MODEL_PATH = os.path.join(REPO_ROOT, "LLM", "esm2_650M")
DEFAULT_ESMC_MODEL_PATH = os.path.join(REPO_ROOT, "LLM", "ESMC_300M")


def _safe_cache_id(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def embedding_model_cache_id(backend: str, model_name_or_path: str = "") -> str:
    value = str(model_name_or_path or "").rstrip("/")
    if not value:
        return _safe_cache_id(backend)
    expanded = os.path.expanduser(value)
    if os.path.exists(expanded) or os.path.sep in value:
        return _safe_cache_id(os.path.basename(value) or backend)
    return _safe_cache_id(value)


def infer_peptide_embedding_dim(
    backend: str,
    AntigenLM_path: str = "",
    esm2_model_name: str = DEFAULT_ESM2_MODEL_PATH,
    esmc_model_name: str = DEFAULT_ESMC_MODEL_PATH,
) -> int:
    if backend == "AntigenLM":
        config_path = os.path.join(AntigenLM_path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for key in ("hidden_size", "d_model", "embedding_size"):
                if key in cfg:
                    return int(cfg[key])
        return 768

    if backend == "esmc":
        name = str(esmc_model_name).lower()
        return 1152 if "600" in name else 960

    if backend == "esm2":
        config_path = os.path.join(str(esm2_model_name), "config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "hidden_size" in cfg:
                return int(cfg["hidden_size"])

        esm2_dims = {
            "esm2_t6_8M_UR50D": 320,
            "esm2_t12_35M_UR50D": 480,
            "esm2_t30_150M_UR50D": 640,
            "esm2_t33_650M_UR50D": 1280,
            "esm2_t36_3B_UR50D": 2560,
            "esm2_t48_15B_UR50D": 5120,
        }
        return esm2_dims.get(esm2_model_name, 1280)

    raise ValueError(f"Unknown embedding backend: {backend}")


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


def antigenlm_needs_spaced_input(tokenizer) -> bool:
    probe = "ACDEFGHIKLMNPQRSTVWY"
    tokens = tokenizer(
        probe,
        truncation=True,
        padding=False,
        max_length=len(probe) + 2,
        add_special_tokens=True,
        return_special_tokens_mask=True,
    )
    special_mask = tokens.get("special_tokens_mask")
    if special_mask is None:
        return False
    residue_token_count = sum(1 for value in special_mask if value == 0)
    return residue_token_count < len(probe) // 2


def antigenLM_extract(sequences, model_name_or_path, device, max_len, batch_size=64, deduplicate=True):
    tokenizer = load_antigenlm_tokenizer(model_name_or_path)
    model = load_antigenlm_model(model_name_or_path, device)
    model.eval()
    use_spaced_input = antigenlm_needs_spaced_input(tokenizer)

    seqs = [str(seq).replace(" ", "").rstrip("-")[:max_len] for seq in sequences]
    if deduplicate:
        uniq2idx, work_seqs, rev_index = {}, [], []
        for seq in seqs:
            if seq not in uniq2idx:
                uniq2idx[seq] = len(work_seqs)
                work_seqs.append(seq)
            rev_index.append(uniq2idx[seq])
    else:
        work_seqs = seqs
        rev_index = list(range(len(seqs)))

    if deduplicate and len(work_seqs) < len(seqs):
        print(f"[Embed] Deduplicated peptides: {len(seqs)} rows -> {len(work_seqs)} unique", flush=True)

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(work_seqs), batch_size), desc=f"Extract {model_name_or_path.split('/')[-1]}"):
            batch_seqs = work_seqs[i:i + batch_size]
            batch_inputs = [" ".join(seq) if use_spaced_input else seq for seq in batch_seqs]
            tokens = tokenizer(
                batch_inputs,
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

    maxL = max_len
    emb_padded = torch.stack([F.pad(e, (0, 0, 0, maxL - e.size(0))) for e in embeddings])
    if deduplicate:
        return emb_padded[torch.LongTensor(rev_index)].contiguous()
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


def _is_local_transformer_model(model_name_or_path: str) -> bool:
    model_path = os.path.expanduser(str(model_name_or_path))
    return os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "config.json"))


def _extract_transformers_esm2_embeddings(
    sequences: List[str],
    model_name_or_path: str,
    device: str = "cuda",
    max_len: int = 256,
    batch_size: int = 512,
    use_amp: bool = True,
    deduplicate: bool = True,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        add_pooling_layer=False,
    ).to(device).eval()
    dev = torch.device(device)

    seqs = [str(s).replace(" ", "")[:max_len] for s in sequences]
    if deduplicate:
        uniq2idx, work_seqs, rev_index = {}, [], []
        for seq in seqs:
            if seq not in uniq2idx:
                uniq2idx[seq] = len(work_seqs)
                work_seqs.append(seq)
            rev_index.append(uniq2idx[seq])
    else:
        work_seqs = seqs
        rev_index = list(range(len(seqs)))

    out_cpu = None
    D_dim = None
    pin_memory = dev.type == "cuda"

    with torch.no_grad():
        for start in tqdm(range(0, len(work_seqs), batch_size), desc=f"Extract {os.path.basename(str(model_name_or_path))}"):
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
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            amp_ctx = autocast_cuda(dtype=_pick_amp_dtype(torch.bfloat16)) if use_amp and dev.type == "cuda" else nullcontext()
            with amp_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is not None:
                    last_hidden = hidden_states[-1]
            if last_hidden is None and isinstance(outputs, (tuple, list)) and outputs:
                last_hidden = outputs[0]
            if last_hidden is None:
                raise ValueError("Could not find token hidden states in ESM2 model output")

            if D_dim is None:
                D_dim = last_hidden.shape[-1]
                out_cpu = torch.empty((len(work_seqs), max_len, D_dim), dtype=torch.float32, pin_memory=pin_memory)

            special_tokens_mask = tokens["special_tokens_mask"].to(device).bool()
            emb_list = []
            for j, seq in enumerate(batch_seqs):
                residue_mask = attention_mask[j].bool() & (~special_tokens_mask[j])
                emb = last_hidden[j][residue_mask][:max_len]
                valid_len = min(len(seq), max_len, emb.size(0))
                emb = emb[:valid_len]
                if valid_len < max_len:
                    emb = F.pad(emb, (0, 0, 0, max_len - valid_len))
                emb_list.append(emb)

            emb_cpu = torch.stack(emb_list, dim=0).to("cpu", non_blocking=pin_memory)
            if emb_cpu.dtype != torch.float32:
                emb_cpu = emb_cpu.float()
            out_cpu[start:start + len(batch_seqs)].copy_(emb_cpu)

    if deduplicate:
        final = torch.empty((len(seqs), max_len, D_dim), dtype=torch.float32)
        for orig_i, uniq_i in enumerate(rev_index):
            final[orig_i].copy_(out_cpu[uniq_i])
        return final
    return out_cpu


@torch.inference_mode()
def extract_esm2_embeddings(
    sequences: List[str],
    model_name: str = DEFAULT_ESM2_MODEL_PATH,
    device: str = "cuda",
    max_len: int = 256,
    batch_size: int = 512,
    use_amp: bool = True,
    amp_dtype=torch.bfloat16,
    deduplicate: bool = True,
    compile_model: bool = False,
) -> torch.Tensor:
    if _is_local_transformer_model(model_name):
        return _extract_transformers_esm2_embeddings(
            sequences,
            model_name_or_path=model_name,
            device=device,
            max_len=max_len,
            batch_size=batch_size,
            use_amp=use_amp,
            deduplicate=deduplicate,
        )

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

    try:
        from esm.pretrained import load_model_and_alphabet
    except Exception as exc:
        raise ImportError(
            "Cannot load ESM2 by model name because this `esm` package does not provide "
            "`load_model_and_alphabet`; pass a local HuggingFace ESM2 directory instead."
        ) from exc
    model, alphabet = load_model_and_alphabet(model_name)
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
            out_cpu = torch.empty((total_N, max_len, D_dim), dtype=torch.float32, pin_memory=torch.device(device).type == "cuda")

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
    return out_cpu


def _resolve_esmc_checkpoint(model_name_or_path: str):
    path = os.path.expanduser(str(model_name_or_path))
    if os.path.isdir(path):
        candidates = sorted(glob(os.path.join(path, "*.pth")))
        if not candidates:
            raise FileNotFoundError(f"No .pth checkpoint found in ESMC directory: {model_name_or_path}")
        preferred = [p for p in candidates if "esmc" in os.path.basename(p).lower()]
        return preferred[0] if preferred else candidates[0]
    if os.path.isfile(path):
        return path
    return None


def load_esmc_model(model_name_or_path: str = DEFAULT_ESMC_MODEL_PATH, device: str = "cuda"):
    checkpoint_path = _resolve_esmc_checkpoint(model_name_or_path)
    dev = torch.device(device)

    if checkpoint_path is None:
        from esm.models.esmc import ESMC

        return ESMC.from_pretrained(model_name_or_path, device=dev)

    from esm.models.esmc import ESMC
    from esm.pretrained import get_esmc_model_tokenizers

    ckpt_name = os.path.basename(checkpoint_path).lower()
    if "600" in ckpt_name:
        d_model, n_heads, n_layers = 1152, 18, 36
    else:
        d_model, n_heads, n_layers = 960, 15, 30

    with torch.device(dev):
        model = ESMC(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            tokenizer=get_esmc_model_tokenizers(),
        ).eval()
    state_dict = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(state_dict)
    if dev.type != "cpu":
        model = model.to(torch.bfloat16)
    return model.to(dev).eval()


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
    dev = torch.device(device)

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

    out_cpu = None
    D_dim = None

    idx = 0
    while idx < len(work_seqs):
        sub_seqs = work_seqs[idx: idx + batch_size]
        idx += len(sub_seqs)

        emb_list = []
        for seq in sub_seqs:
            p = ESMProtein(sequence=seq)
            pt = client.encode(p)
            if use_amp and dev.type == "cuda":
                with autocast_cuda(dtype=amp_dtype):
                    out = client.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
            else:
                out = client.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))

            e = out.embeddings
            if e.ndim == 2:
                e = e.unsqueeze(0)
            if str(e.device) != str(device):
                e = e.to(device)

            if use_amp and dev.type == "cuda" and e.dtype != amp_dtype:
                e = e.to(amp_dtype)

            e = e[0]
            if e.size(0) >= len(seq) + 2:
                e = e[1:1 + len(seq)]
            else:
                e = e[:len(seq)]
            e = e[:max_len]
            if e.size(0) < max_len:
                e = F.pad(e, (0, 0, 0, max_len - e.size(0)))
            emb_list.append(e)

        emb_batch = torch.stack(emb_list, dim=0)
        B, _, D_now = emb_batch.shape

        if D_dim is None:
            D_dim = D_now
            total_N = len(work_seqs)
            out_cpu = torch.empty((total_N, max_len, D_dim), dtype=torch.float32, pin_memory=dev.type == "cuda")

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
