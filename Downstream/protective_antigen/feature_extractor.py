import contextlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


P = os.path.join
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

EMBED_BACKENDS = (
    "AntigenLM",
    "AntigenLM_4layers",
    "AntigenLM_12layers",
    "Fine-esm3",
    "ESMC_300M_two_stage_last8",
    "AntigenLM_random-mlm20",
    "AntigenLM_random-mlm15",
    "AntigenLM_no_window",
    "esm2",
    "esmc_300m",
    "PathogLM",
    "MicroLM",
)

HF_BACKEND_PATHS = {
    "AntigenLM": P(PROJECT_ROOT, "LLM", "AntigenLM"),
    "AntigenLM_4layers": "../../LLM/AntigenLM_last4_layers",
    "AntigenLM_12layers": "../../LLM/AntigenLM_last12_layers",
    "Fine-esm3": P(PROJECT_ROOT, "LLM", "stage2_antigen_epoch3_step1548.pt"),
    "ESMC_300M_two_stage_last8": "../../LLM/ESMC_300M_two_stage_fine_last8",
    "AntigenLM_random-mlm20": "../../LLM/AntigenLM_random_mlm20_last8_layers",
    "AntigenLM_random-mlm15": "../../LLM/AntigenLM_random_mlm15_last8_layers",
    "AntigenLM_no_window":"../../LLM/AntigenLM_no_window_last8_layers",
    "PathogLM":"../../LLM/PathogLM",
    "MicroLM":"../../LLM/MicroLM"
}

ESM2_MODEL_PATH = P(PROJECT_ROOT, "LLM", "esm2_650M")
LOCAL_ESMC_300M_WEIGHT = P(PROJECT_ROOT, "LLM", "ESMC_300M", "esmc_300m_2024_12_v0.pth")

ESMC_HF_BACKENDS = {
    "ESMC_300M_two_stage",
    "ESMC_300M_two_stage_last8",
    "ESMC_300M_two_stage_last8_hf",
}

ESM3_TORCH_BACKENDS = {
    "Fine-esm3",
}

ANTIGENLM_DISTILLED_BACKENDS = {
    "AntigenLM_distilled",
    "AntigenLM",
    "AntigenLM_distilled_origin",
}

ANTIGENLM_HF_BACKENDS = {
    "AntigenLM",
    "AntigenLM_4layers",
    "AntigenLM_8layers",
    "AntigenLM_12layers",
    "AntigenLM_random-mlm20",
    "AntigenLM_random-mlm15",
    "AntigenLM_no_window",
    "PathogLM",
    "MicroLM",
    "MicroLM_pathogen_random_mlm20_last8",
    *ANTIGENLM_DISTILLED_BACKENDS,
}


def _is_cuda(dev):
    return (isinstance(dev, torch.device) and dev.type == "cuda") or (
        isinstance(dev, str) and str(dev).startswith("cuda")
    )


def resolve_backend_path(model_type, backend_path=None):
    if backend_path:
        return os.path.abspath(os.path.expanduser(backend_path))
    return HF_BACKEND_PATHS.get(model_type, backend_path)


def resolve_hf_extract_dtype(dtype_name, device):
    if not _is_cuda(device):
        return None
    if dtype_name in (None, "none", "fp32"):
        return None
    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported --hf_extract_dtype: {dtype_name}")


def patch_esmc_tokenizer():
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    def make_token_property(private_name):
        def getter(self):
            value = getattr(self, private_name, None)
            return None if value is None else str(value)

        def setter(self, value):
            setattr(self, private_name, value)

        return property(getter, setter)

    for token_name in ("cls_token", "eos_token", "mask_token", "pad_token"):
        prop = getattr(EsmSequenceTokenizer, token_name, None)
        if isinstance(prop, property) and prop.fset is None:
            setattr(EsmSequenceTokenizer, token_name, make_token_property(f"_{token_name}"))


def import_esmc():
    try:
        from esm.models.esmc import ESMC

        patch_esmc_tokenizer()
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("esm"):
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            raise ImportError(
                "The ESM-C backend requires EvolutionaryScale's `esm` package. "
                f"Current Python is {py_version}; `esm` releases require Python >=3.10. "
                "Use the AntigenLM environment and install it with `python -m pip install -U esm`. "
                "If you have `fair-esm` installed, remove it or use a separate environment because "
                "it provides `esm.pretrained` but not `esm.models.esmc`."
            ) from exc
        raise
    return ESMC


def resolve_esmc_model_name(backend_path):
    if not backend_path or "AntigenLM" in backend_path:
        return "esmc_300m"
    return backend_path


def load_esmc_client(ESMC, model_name, device):
    if model_name != "esmc_300m":
        raise ValueError(f"Unsupported local ESM-C model: {model_name}")
    if not os.path.exists(LOCAL_ESMC_300M_WEIGHT):
        raise FileNotFoundError(f"ESM-C weight not found: {LOCAL_ESMC_300M_WEIGHT}")

    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    model = ESMC(
        d_model=960,
        n_heads=15,
        n_layers=30,
        tokenizer=EsmSequenceTokenizer(),
    ).eval()
    state_dict = torch.load(LOCAL_ESMC_300M_WEIGHT, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)


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
    all_batches = []

    amp_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if (
        autocast_dtype is not None and _is_cuda(device)
    ) else contextlib.nullcontext()

    seqs = [seq[: model_max_len - 2] for seq in sequences]
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


def tokenizer_requires_space_separated_input(tokenizer):
    probe = "ACDE"
    encoded = tokenizer(probe, add_special_tokens=True)
    special_ids = {
        token_id
        for token_id in [
            getattr(tokenizer, "cls_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "sep_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        ]
        if token_id is not None
    }
    aa_ids = [token_id for token_id in encoded["input_ids"] if token_id not in special_ids]
    return len(aa_ids) != len(probe) or getattr(tokenizer, "unk_token_id", None) in aa_ids


def _amp_context(device, autocast_dtype):
    if autocast_dtype is not None and _is_cuda(device):
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
    return contextlib.nullcontext()


def _sorted_indices_by_length(sequences, residue_budget):
    return sorted(
        range(len(sequences)),
        key=lambda idx: min(len(sequences[idx]), residue_budget),
        reverse=True,
    )


def _batch_texts(sequences, batch_indices, residue_budget, space_separated):
    batch = [sequences[idx][:residue_budget] for idx in batch_indices]
    return [" ".join(seq) for seq in batch] if space_separated else batch


def _special_tokens_mask(inputs, attention_mask, device):
    mask = inputs.get("special_tokens_mask")
    if mask is None:
        return torch.zeros_like(attention_mask, dtype=torch.bool)
    return mask.to(device).bool()


def _finalize_token_embeddings(
    hidden,
    attention_mask,
    special_tokens_mask,
    add_special_tokens,
    max_len,
):
    residue_mask = attention_mask.bool() & (~special_tokens_mask)
    token_embeds = hidden[:, 1:, :] if add_special_tokens else hidden
    token_mask = residue_mask[:, 1:] if add_special_tokens else residue_mask

    token_embeds = token_embeds[:, :max_len, :]
    token_mask = token_mask[:, :max_len]
    if token_embeds.size(1) < max_len:
        pad_len = max_len - token_embeds.size(1)
        token_embeds = F.pad(token_embeds, (0, 0, 0, pad_len))
        token_mask = F.pad(token_mask, (0, pad_len), value=False)
    return token_embeds * token_mask.unsqueeze(-1).to(token_embeds.dtype)


def extract_hf_embeddings(
    sequences,
    tokenizer,
    model,
    device,
    batch_size=8,
    max_len=512,
    add_special_tokens=True,
    space_separated_input=True,
    output_hidden_states=False,
    autocast_dtype=None,
):
    model.eval()
    model_max_len = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    token_budget = max_len + (2 if add_special_tokens else 0)
    input_max_len = min(token_budget, int(model_max_len)) if model_max_len else token_budget
    residue_budget = max(input_max_len - 2, 0) if add_special_tokens else input_max_len
    sorted_indices = _sorted_indices_by_length(sequences, residue_budget)
    embeds_by_index = [None] * len(sequences)

    with torch.inference_mode(), _amp_context(device, autocast_dtype):
        for i in tqdm(range(0, len(sorted_indices), batch_size), desc="HF Extract"):
            batch_indices = sorted_indices[i:i + batch_size]
            inputs = tokenizer(
                _batch_texts(sequences, batch_indices, residue_budget, space_separated_input),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=input_max_len,
                add_special_tokens=add_special_tokens,
                return_special_tokens_mask=True,
            )
            model_inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            if output_hidden_states and hasattr(model, "backbone"):
                model_inputs["return_logits"] = False
            outputs = model(**model_inputs, output_hidden_states=output_hidden_states)
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            elif getattr(outputs, "hidden_states", None) is not None:
                hidden = outputs.hidden_states[-1]
            else:
                hidden = outputs[0]
            attention_mask = model_inputs["attention_mask"].bool()
            token_embeds = _finalize_token_embeddings(
                hidden,
                attention_mask,
                _special_tokens_mask(inputs, attention_mask, device),
                add_special_tokens,
                max_len,
            )
            batch_embeds = token_embeds.detach().cpu().float().numpy()
            for seq_idx, emb in zip(batch_indices, batch_embeds):
                embeds_by_index[seq_idx] = emb
    return torch.from_numpy(np.stack(embeds_by_index, axis=0)).float()


def extract_esmc_hf_embeddings(
    sequences,
    tokenizer,
    model,
    device,
    batch_size=8,
    max_len=512,
    add_special_tokens=True,
    space_separated_input=False,
    autocast_dtype=None,
):
    model.eval()
    token_budget = max_len + (2 if add_special_tokens else 0)
    residue_budget = max(token_budget - 2, 0) if add_special_tokens else token_budget
    sorted_indices = _sorted_indices_by_length(sequences, residue_budget)
    embeds_by_index = [None] * len(sequences)

    with torch.inference_mode(), _amp_context(device, autocast_dtype):
        for i in tqdm(range(0, len(sorted_indices), batch_size), desc="ESMC HF Extract"):
            batch_indices = sorted_indices[i:i + batch_size]
            inputs = tokenizer(
                _batch_texts(sequences, batch_indices, residue_budget, space_separated_input),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=token_budget,
                add_special_tokens=add_special_tokens,
                return_special_tokens_mask=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = input_ids.ne(getattr(model.config, "pad_token_id", 1))
            else:
                attention_mask = attention_mask.to(device)

            x = model.esmc.embed(input_ids)
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(model.config, "pad_token_id", 1)
            sequence_id = input_ids.eq(pad_token_id)
            chain_id = torch.ones(x.shape[:-1], dtype=torch.int64, device=x.device)
            for block in model.esmc.transformer.blocks:
                x = block(x, sequence_id, None, None, chain_id)
            hidden = model.esmc.transformer.norm(x)

            token_embeds = _finalize_token_embeddings(
                hidden,
                attention_mask,
                _special_tokens_mask(inputs, attention_mask, device),
                add_special_tokens,
                max_len,
            )
            batch_embeds = token_embeds.detach().cpu().float().numpy()
            for seq_idx, emb in zip(batch_indices, batch_embeds):
                embeds_by_index[seq_idx] = emb
    return torch.from_numpy(np.stack(embeds_by_index, axis=0)).float()


def _infer_esm3_config(state_dict):
    d_model = int(state_dict["encoder.sequence_embed.weight"].shape[1])
    n_layers = max(
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("transformer.blocks.")
    ) + 1
    if d_model == 1536 and n_layers == 48:
        return {
            "d_model": 1536,
            "n_heads": 24,
            "v_heads": 256,
            "n_layers": 48,
        }
    raise ValueError(
        "Unsupported Fine-esm3 checkpoint architecture: "
        f"d_model={d_model}, n_layers={n_layers}"
    )


def load_fine_esm3_from_checkpoint(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Fine-esm3 checkpoint not found: {checkpoint_path}")

    try:
        from esm.models.esm3 import ESM3
        from esm.pretrained import (
            ESM3_function_decoder_v0,
            ESM3_structure_decoder_v0,
            ESM3_structure_encoder_v0,
        )
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("esm"):
            raise ImportError(
                "The Fine-esm3 backend requires EvolutionaryScale's `esm` package. "
                "Use an environment where the `esm` package is installed."
            ) from exc
        raise

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Fine-esm3 checkpoint has no model state dict: {checkpoint_path}")

    config = _infer_esm3_config(state_dict)
    model = ESM3(
        **config,
        structure_encoder_fn=ESM3_structure_encoder_v0,
        structure_decoder_fn=ESM3_structure_decoder_v0,
        function_decoder_fn=ESM3_function_decoder_v0,
        tokenizers=SimpleNamespace(sequence=EsmSequenceTokenizer()),
    ).eval()
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to load Fine-esm3 weights: "
            f"missing keys={missing_keys}, unexpected keys={unexpected_keys}"
        )
    model = model.to(device)
    if _is_cuda(device):
        model = model.to(torch.bfloat16)
    return model, config["d_model"]


def extract_esm3_embeddings(
    sequences,
    model,
    device,
    batch_size=1,
    max_len=512,
    add_special_tokens=True,
    autocast_dtype=None,
):
    model.eval()
    tokenizer = model.tokenizers.sequence
    token_budget = max_len + (2 if add_special_tokens else 0)
    residue_budget = max(token_budget - 2, 0) if add_special_tokens else token_budget
    sorted_indices = _sorted_indices_by_length(sequences, residue_budget)
    embeds_by_index = [None] * len(sequences)

    with torch.inference_mode(), _amp_context(device, autocast_dtype):
        for i in tqdm(range(0, len(sorted_indices), batch_size), desc="ESM3 Extract"):
            batch_indices = sorted_indices[i:i + batch_size]
            inputs = tokenizer(
                _batch_texts(sequences, batch_indices, residue_budget, False),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=token_budget,
                add_special_tokens=add_special_tokens,
                return_special_tokens_mask=True,
            )
            sequence_tokens = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device).bool()
            output = model(sequence_tokens=sequence_tokens, sequence_id=~attention_mask)
            token_embeds = _finalize_token_embeddings(
                output.embeddings,
                attention_mask,
                _special_tokens_mask(inputs, attention_mask, device),
                add_special_tokens,
                max_len,
            )
            batch_embeds = token_embeds.detach().cpu().float().numpy()
            for seq_idx, emb in zip(batch_indices, batch_embeds):
                embeds_by_index[seq_idx] = emb
    return torch.from_numpy(np.stack(embeds_by_index, axis=0)).float()


def _resolve_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype, None)
    return None


def patch_esm_pretrained_tokenizer_factory():
    try:
        import esm.pretrained as esm_pretrained
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    except ImportError:
        return
    if not hasattr(esm_pretrained, "get_esmc_model_tokenizers"):
        esm_pretrained.get_esmc_model_tokenizers = lambda: EsmSequenceTokenizer()


def load_masked_lm_from_pretrained(backend_path, trust_remote_code=False):
    from transformers import AutoConfig, AutoModelForMaskedLM

    patch_esm_pretrained_tokenizer_factory()
    try:
        return AutoModelForMaskedLM.from_pretrained(
            backend_path,
            trust_remote_code=trust_remote_code,
        )
    except AttributeError as exc:
        if "'NoneType' object has no attribute 'get'" not in str(exc):
            raise

    safetensors_path = P(backend_path, "model.safetensors")
    if not os.path.isfile(safetensors_path):
        raise FileNotFoundError(
            "Transformers failed while reading safetensors metadata, and no "
            f"model.safetensors file was found at {safetensors_path}"
        )

    from safetensors.torch import load_file

    config = AutoConfig.from_pretrained(
        backend_path,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForMaskedLM.from_config(
        config,
        trust_remote_code=trust_remote_code,
    )
    dtype = _resolve_torch_dtype(getattr(config, "torch_dtype", None))
    if dtype is not None:
        model = model.to(dtype=dtype)
    state_dict = load_file(safetensors_path, device="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to load HuggingFace masked-LM weights from "
            f"{safetensors_path}: missing keys={missing_keys}, "
            f"unexpected keys={unexpected_keys}"
        )
    return model


def load_hf_tokenizer(backend_path, trust_remote_code=False):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            backend_path,
            trust_remote_code=trust_remote_code,
        )
    except ValueError as exc:
        if "Tokenizer class EsmSequenceTokenizer" not in str(exc):
            raise
        tokenizer_json = P(backend_path, "tokenizer.json")
        if not os.path.isfile(tokenizer_json):
            raise

    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        cls_token="<cls>",
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        mask_token="<mask>",
        additional_special_tokens=["|"],
    )


def get_model_and_extract_func(
    model_type,
    backend_path,
    device,
    max_len=512,
    hf_add_special_tokens=True,
    hf_extract_batch_size=8,
    hf_autocast_dtype=None,
):
    backend_path = resolve_backend_path(model_type, backend_path)

    if model_type == "esmc_300m":
        ESMC = import_esmc()
        client = load_esmc_client(ESMC, resolve_esmc_model_name(backend_path), device)
        return lambda seqs: extract_esmc_embeddings(
            seqs,
            client,
            device,
            batch_size=16,
            max_len=max_len,
            autocast_dtype=(torch.bfloat16 if _is_cuda(device) else None),
        ), 960

    if model_type == "esm2":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_PATH)
        model = AutoModel.from_pretrained(ESM2_MODEL_PATH).to(device)
        space_separated = tokenizer_requires_space_separated_input(tokenizer)
        return lambda seqs: extract_hf_embeddings(
            seqs,
            tokenizer,
            model,
            device,
            batch_size=hf_extract_batch_size,
            max_len=max_len,
            add_special_tokens=hf_add_special_tokens,
            space_separated_input=space_separated,
            autocast_dtype=hf_autocast_dtype,
        ), int(getattr(model.config, "hidden_size", 1280))

    if model_type in ESM3_TORCH_BACKENDS:
        if not backend_path:
            raise ValueError(f"--backend_path is required for Fine-esm3 backend: {model_type}")
        model, emb_dim = load_fine_esm3_from_checkpoint(backend_path, device)
        esm3_batch_size = max(1, hf_extract_batch_size)
        if not _is_cuda(device):
            esm3_batch_size = 1
        return lambda seqs: extract_esm3_embeddings(
            seqs,
            model,
            device,
            batch_size=esm3_batch_size,
            max_len=max_len,
            add_special_tokens=hf_add_special_tokens,
            autocast_dtype=hf_autocast_dtype,
        ), emb_dim

    from transformers import AutoModel

    if not backend_path:
        raise ValueError(f"--backend_path is required for HuggingFace backend: {model_type}")
    if not os.path.isdir(backend_path):
        raise FileNotFoundError(f"HuggingFace backend path not found: {backend_path}")

    trust_remote_code = model_type in ANTIGENLM_HF_BACKENDS or model_type in ESMC_HF_BACKENDS
    tokenizer = load_hf_tokenizer(backend_path, trust_remote_code=trust_remote_code)
    if model_type in ESMC_HF_BACKENDS:
        model = load_masked_lm_from_pretrained(
            backend_path,
            trust_remote_code=True,
        ).to(device)
        space_separated = tokenizer_requires_space_separated_input(tokenizer)
        return lambda seqs: extract_esmc_hf_embeddings(
            seqs,
            tokenizer,
            model,
            device,
            batch_size=hf_extract_batch_size,
            max_len=max_len,
            add_special_tokens=hf_add_special_tokens,
            space_separated_input=space_separated,
            autocast_dtype=hf_autocast_dtype,
        ), int(getattr(model.config, "hidden_size", getattr(model.config, "d_model", 960)))

    if model_type in ANTIGENLM_HF_BACKENDS:
        model = load_masked_lm_from_pretrained(
            backend_path,
            trust_remote_code=True,
        ).to(device)
        output_hidden_states = True
    else:
        model = AutoModel.from_pretrained(backend_path).to(device)
        output_hidden_states = False
    space_separated = tokenizer_requires_space_separated_input(tokenizer)
    return lambda seqs: extract_hf_embeddings(
        seqs,
        tokenizer,
        model,
        device,
        batch_size=hf_extract_batch_size,
        max_len=max_len,
        add_special_tokens=hf_add_special_tokens,
        space_separated_input=space_separated,
        output_hidden_states=output_hidden_states,
        autocast_dtype=hf_autocast_dtype,
    ), int(getattr(model.config, "hidden_size", getattr(model.config, "d_model", 768)))
