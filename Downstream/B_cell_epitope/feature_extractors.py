import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def first_existing_hf_encoder_path(*paths):
    for path in paths:
        if os.path.exists(os.path.join(path, "config.json")):
            return path
    return paths[0]


DEFAULT_ESM2_ENCODER_PATH = first_existing_hf_encoder_path(
    os.path.join(PROJECT_ROOT, "LLM", "esm2_650M"),
    os.path.join(PROJECT_ROOT, "LLM_MR", "esm2_650M"),
)
DEFAULT_ANTIGENLM_ENCODER_PATH = first_existing_hf_encoder_path(
    os.path.join(PROJECT_ROOT, "LLM", "AntigenLM"),
    os.path.join(PROJECT_ROOT, "LLM_MR", "AntigenLM"),
)

ENCODER_CONFIGS = {
    "esm2": {
        "backend": "hf",
        "display_name": "ESM2",
        "base_stem": "bcell_epitope_esm2",
        "encoder_path": DEFAULT_ESM2_ENCODER_PATH,
        "space_separated": False,
    },
    "antigenlm": {
        "backend": "hf",
        "display_name": "AntigenLM",
        "base_stem": "bcell_epitope_antigenlm_distilled",
        "encoder_path": DEFAULT_ANTIGENLM_ENCODER_PATH,
        "space_separated": False,
    },
    "esm2_antigenlm": {
        "backend": "fusion",
        "display_name": "ESM2+AntigenLM",
        "base_stem": "bcell_epitope_esm2_antigenlm",
        "components": ("esm2", "antigenlm"),
    },
}

MODEL_ALIASES = {
    "esm2": "esm2",
    "antigenlm": "antigenlm",
    "antigenlm_distilled": "antigenlm",
    "esm2_antigenlm": "esm2_antigenlm",
    "esm2+antigenlm": "esm2_antigenlm",
    "esm2-antigenlm": "esm2_antigenlm",
    "fusion": "esm2_antigenlm",
}


@dataclass
class FeatureExtractor:
    name: str
    backend: str
    display_name: str
    encoder_ref: object
    encoder: object
    tokenizer: object = None
    input_dim: int = None
    space_separated: bool = False
    sub_extractors: tuple = None


def normalize_model_name(model_name):
    name = str(model_name).strip().lower()
    if name not in MODEL_ALIASES:
        valid_names = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {valid_names}.")
    return MODEL_ALIASES[name]


def get_encoder_config(model_name):
    return ENCODER_CONFIGS[normalize_model_name(model_name)]


def checkpoint_stem(model_name, classifier_type):
    config = get_encoder_config(model_name)
    return f"{config['base_stem']}_{classifier_type}"


def parse_residue_feature_groups(value):
    if value is None:
        return ()
    if isinstance(value, (tuple, list, set)):
        groups = tuple(str(group).strip().lower() for group in value if str(group).strip())
    else:
        normalized = str(value).strip().lower()
        if normalized in {"", "none", "off", "false", "0"}:
            return ()
        groups = tuple(part.strip() for part in normalized.split(",") if part.strip())

    if not groups or all(group in {"none", "off", "false", "0"} for group in groups):
        return ()
    raise ValueError("Extra residue feature groups were removed from the final model; use --residue-features none.")


def residue_feature_dim(groups):
    parse_residue_feature_groups(groups)
    return 0


def augmented_input_dim(base_dim, groups):
    parse_residue_feature_groups(groups)
    return int(base_dim)


def format_sequences_for_encoder(sequences, space_separated=False):
    if space_separated:
        return [" ".join(list(sequence)) for sequence in sequences]
    return sequences


def load_transformers_classes():
    try:
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required for ESM2/AntigenLM encoders."
        ) from exc
    return AutoModel, AutoModelForMaskedLM, AutoTokenizer


def validate_hf_encoder_ref(encoder_ref):
    if os.path.isabs(encoder_ref) or os.path.sep in encoder_ref:
        if not os.path.isdir(encoder_ref):
            raise FileNotFoundError(f"HF encoder directory not found: {encoder_ref}")
        config_path = os.path.join(encoder_ref, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"HF encoder config not found: {config_path}")


def load_feature_extractor(
    model_name,
    device,
    encoder_path=None,
    esm2_path=None,
    antigenlm_path=None,
):
    name = normalize_model_name(model_name)
    config = ENCODER_CONFIGS[name]

    if config["backend"] == "fusion":
        component_paths = {}
        if isinstance(encoder_path, dict):
            component_paths.update({key: value for key, value in encoder_path.items() if value})
        elif encoder_path:
            raise ValueError("ESM2+AntigenLM expects encoder_path as {'esm2': path, 'antigenlm': path}.")
        if esm2_path:
            component_paths["esm2"] = esm2_path
        if antigenlm_path:
            component_paths["antigenlm"] = antigenlm_path

        sub_extractors = tuple(
            load_feature_extractor(component_name, device, encoder_path=component_paths.get(component_name))
            for component_name in config["components"]
        )
        input_dim = sum(extractor.input_dim for extractor in sub_extractors)
        encoder_ref = {extractor.name: extractor.encoder_ref for extractor in sub_extractors}
        return FeatureExtractor(
            name=name,
            backend="fusion",
            display_name=config["display_name"],
            encoder_ref=encoder_ref,
            encoder=None,
            input_dim=input_dim,
            sub_extractors=sub_extractors,
        )

    AutoModel, AutoModelForMaskedLM, AutoTokenizer = load_transformers_classes()
    encoder_ref = encoder_path or config["encoder_path"]
    validate_hf_encoder_ref(encoder_ref)
    tokenizer = AutoTokenizer.from_pretrained(encoder_ref, trust_remote_code=True)
    try:
        model_kwargs = {"trust_remote_code": True}
        if name == "esm2":
            model_kwargs["add_pooling_layer"] = False
        encoder = AutoModel.from_pretrained(encoder_ref, **model_kwargs).to(device)
    except ValueError:
        encoder = AutoModelForMaskedLM.from_pretrained(encoder_ref, trust_remote_code=True).to(device)
    input_dim = getattr(encoder.config, "hidden_size", None) or getattr(encoder.config, "d_model", None)
    if input_dim is None:
        raise ValueError(f"Could not infer hidden size for encoder: {encoder_ref}")

    return FeatureExtractor(
        name=name,
        backend="hf",
        display_name=config["display_name"],
        encoder_ref=encoder_ref,
        encoder=encoder,
        tokenizer=tokenizer,
        input_dim=input_dim,
        space_separated=config.get("space_separated", False),
    )


def freeze_encoder(feature_extractor):
    if feature_extractor.backend == "fusion":
        for sub_extractor in feature_extractor.sub_extractors:
            freeze_encoder(sub_extractor)
        return

    for param in feature_extractor.encoder.parameters():
        param.requires_grad = False


def unfreeze_last_hf_layers(feature_extractor, num_layers):
    if feature_extractor.backend == "fusion":
        return sum(unfreeze_last_hf_layers(sub_extractor, num_layers) for sub_extractor in feature_extractor.sub_extractors)

    freeze_encoder(feature_extractor)
    if feature_extractor.backend != "hf" or num_layers <= 0:
        return 0

    encoder = feature_extractor.encoder
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    elif hasattr(encoder, "backbone") and hasattr(encoder.backbone, "transformer"):
        blocks = getattr(encoder.backbone.transformer, "blocks", None)
        if blocks is not None:
            for block in blocks[-num_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
            for param in encoder.backbone.transformer.norm.parameters():
                param.requires_grad = True
    else:
        print("Warning: Could not identify encoder layers to unfreeze. Keeping encoder frozen.")

    return sum(param.numel() for param in encoder.parameters() if param.requires_grad)


def set_encoder_mode(feature_extractor, is_train):
    if feature_extractor.backend == "fusion":
        for sub_extractor in feature_extractor.sub_extractors:
            set_encoder_mode(sub_extractor, is_train)
        return

    has_trainable_params = any(param.requires_grad for param in feature_extractor.encoder.parameters())
    if is_train and has_trainable_params:
        feature_extractor.encoder.train()
    else:
        feature_extractor.encoder.eval()


def collect_trainable_encoder_state_dict(feature_extractor):
    if feature_extractor.backend == "fusion":
        state_by_component = {}
        for sub_extractor in feature_extractor.sub_extractors:
            component_state = collect_trainable_encoder_state_dict(sub_extractor)
            if component_state:
                state_by_component[sub_extractor.name] = component_state
        return state_by_component

    return {
        name: param.detach().cpu()
        for name, param in feature_extractor.encoder.named_parameters()
        if param.requires_grad
    }


def apply_encoder_state_dict_delta(feature_extractor, state_delta):
    if not state_delta:
        return 0

    if feature_extractor.backend == "fusion":
        loaded_count = 0
        for sub_extractor in feature_extractor.sub_extractors:
            loaded_count += apply_encoder_state_dict_delta(
                sub_extractor,
                state_delta.get(sub_extractor.name, {}),
            )
        return loaded_count

    current_state = feature_extractor.encoder.state_dict()
    compatible_state = {}
    for name, value in state_delta.items():
        if name not in current_state:
            continue
        compatible_state[name] = value.to(dtype=current_state[name].dtype)

    if compatible_state:
        feature_extractor.encoder.load_state_dict(compatible_state, strict=False)
    return len(compatible_state)


def encode_residue_batch(
    feature_extractor,
    samples,
    device,
    max_length,
    include_metadata=False,
    residue_feature_groups=None,
):
    parse_residue_feature_groups(residue_feature_groups)
    if feature_extractor.backend == "fusion":
        return encode_fusion_residue_batch(feature_extractor, samples, device, max_length, include_metadata)
    return encode_hf_residue_batch(feature_extractor, samples, device, max_length, include_metadata)


def encode_fusion_residue_batch(feature_extractor, samples, device, max_length, include_metadata=False):
    encoded_batches = [
        encode_residue_batch(sub_extractor, samples, device, max_length, include_metadata=False)
        for sub_extractor in feature_extractor.sub_extractors
    ]

    residue_features = []
    residue_labels = []
    metadata = []
    truncated_sequences = 0
    reference_batch = encoded_batches[0]

    for sample_index, sample in enumerate(samples):
        residue_counts = [
            int(encoded_batch["residue_mask"][sample_index].sum().item())
            for encoded_batch in encoded_batches
        ]
        residue_count = min(residue_counts)
        if residue_count == 0:
            continue
        if residue_count < len(sample["sequence"]):
            truncated_sequences += 1

        fused_features = torch.cat(
            [encoded_batch["features"][sample_index, :residue_count] for encoded_batch in encoded_batches],
            dim=-1,
        )
        residue_features.append(fused_features)
        residue_labels.append(reference_batch["labels"][sample_index, :residue_count])
        if include_metadata:
            metadata.extend(sample_metadata(sample, residue_count))

    return pad_residue_batch(residue_features, residue_labels, metadata, truncated_sequences)


def encode_hf_residue_batch(feature_extractor, samples, device, max_length, include_metadata=False):
    sequences = [sample["sequence"] for sample in samples]
    encoder_sequences = format_sequences_for_encoder(sequences, feature_extractor.space_separated)
    model_max_length = getattr(getattr(feature_extractor.encoder, "config", None), "max_position_embeddings", None)
    input_max_length = min(max_length, int(model_max_length)) if model_max_length else max_length
    encoded = feature_extractor.tokenizer(
        encoder_sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=input_max_length,
        return_special_tokens_mask=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model_inputs = {
        key: value
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }

    outputs = feature_extractor.encoder(**model_inputs, output_hidden_states=True)
    hidden_states = getattr(outputs, "last_hidden_state", None)
    if hidden_states is None:
        output_hidden_states = getattr(outputs, "hidden_states", None)
        if not output_hidden_states:
            raise ValueError(f"Encoder did not return hidden states: {feature_extractor.encoder_ref}")
        hidden_states = output_hidden_states[-1]
    attention_mask = model_inputs["attention_mask"].bool()
    special_tokens_mask = encoded["special_tokens_mask"].bool()

    residue_features = []
    residue_labels = []
    metadata = []
    truncated_sequences = 0

    for i, sample in enumerate(samples):
        residue_token_mask = attention_mask[i] & (~special_tokens_mask[i])
        features = hidden_states[i, residue_token_mask]
        residue_count = min(features.size(0), len(sample["sequence"]))
        if residue_count == 0:
            continue
        if residue_count < len(sample["sequence"]):
            truncated_sequences += 1

        residue_features.append(features[:residue_count])
        residue_labels.append(torch.tensor(sample["labels"][:residue_count], dtype=torch.long, device=device))
        if include_metadata:
            metadata.extend(sample_metadata(sample, residue_count))

    return pad_residue_batch(residue_features, residue_labels, metadata, truncated_sequences)


def sample_metadata(sample, residue_count):
    return [
        {
            "sample_id": sample["id"],
            "residue_index": residue_index + 1,
            "residue": sample["sequence"][residue_index],
            "true_label": sample["labels"][residue_index],
        }
        for residue_index in range(residue_count)
    ]


def pad_residue_batch(residue_features, residue_labels, metadata, truncated_sequences):
    if not residue_features:
        raise ValueError("No residue features were extracted. Check tokenizer/model settings.")

    max_residue_count = max(features.size(0) for features in residue_features)
    device = residue_features[0].device
    padded_features = []
    padded_labels = []
    masks = []

    for sample_features, sample_labels in zip(residue_features, residue_labels):
        residue_count = sample_features.size(0)
        pad_count = max_residue_count - residue_count
        padded_features.append(F.pad(sample_features, (0, 0, 0, pad_count)))
        padded_labels.append(F.pad(sample_labels[:residue_count], (0, pad_count)))
        masks.append(
            F.pad(
                torch.ones(residue_count, dtype=torch.bool, device=device),
                (0, pad_count),
            )
        )

    return {
        "features": torch.stack(padded_features, dim=0).float(),
        "labels": torch.stack(padded_labels, dim=0),
        "residue_mask": torch.stack(masks, dim=0),
        "metadata": metadata,
        "truncated_sequences": truncated_sequences,
    }
