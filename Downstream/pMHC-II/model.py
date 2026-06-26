import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceConvBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim)
        self.pointwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        residual = tokens
        tokens = self.norm(tokens).transpose(1, 2)
        tokens = self.depthwise(tokens)
        tokens = self.pointwise(tokens).transpose(1, 2)
        return residual + self.dropout(tokens)


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [SequenceConvBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, embeddings):
        tokens = self.input(embeddings)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, tokens, mask=None):
        scores = self.score(tokens).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(tokens * weights, dim=1)


class AntigenLMESM2PMHCIIModel(nn.Module):
    """Cached peptide tokens + ESM2 MHC-II matching classifier."""

    def __init__(
        self,
        peptide_dim=960,
        hla_dim=1152,
        hidden_dim=256,
        peptide_layers=2,
        hla_layers=2,
        hla_fusion_layers=1,
        cross_attention_layers=0,
        attention_heads=8,
        dropout=0.2,
        projection_dim=128,
        aggregation="pool",
        window_size=9,
        window_representation="mean",
        mil_pooling="softmax",
        mil_temperature=1.0,
    ):
        super().__init__()
        if aggregation not in {"pool", "mil_window"}:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        if window_representation not in {"mean", "position_aware", "dsca"}:
            raise ValueError(f"Unsupported window_representation: {window_representation}")
        if mil_pooling not in {"softmax", "max"}:
            raise ValueError(f"Unsupported mil_pooling: {mil_pooling}")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        self.config = {
            "peptide_dim": peptide_dim,
            "hla_dim": hla_dim,
            "hidden_dim": hidden_dim,
            "peptide_layers": peptide_layers,
            "hla_layers": hla_layers,
            "hla_fusion_layers": hla_fusion_layers,
            "cross_attention_layers": cross_attention_layers,
            "attention_heads": attention_heads,
            "dropout": dropout,
            "projection_dim": projection_dim,
            "aggregation": aggregation,
            "window_size": window_size,
            "window_representation": window_representation,
            "mil_pooling": mil_pooling,
            "mil_temperature": mil_temperature,
        }
        self.aggregation = aggregation
        self.window_size = int(window_size)
        self.window_representation = window_representation
        self.mil_pooling = mil_pooling
        self.mil_temperature = float(mil_temperature)
        self.register_buffer("hla_embedding_table", torch.empty(0), persistent=False)

        self.peptide_encoder = SequenceEncoder(peptide_dim, hidden_dim, peptide_layers, dropout)
        self.hla_chain_encoder = SequenceEncoder(hla_dim, hidden_dim, hla_layers, dropout)
        self.hla_fusion_blocks = nn.ModuleList(
            [SequenceConvBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(hla_fusion_layers)]
        )

        self.pool = AttentionPool(hidden_dim)
        if self.aggregation == "mil_window" and self.window_representation == "position_aware":
            self.window_position_embedding = nn.Parameter(torch.empty(self.window_size, hidden_dim))
            self.window_slot_norm = nn.LayerNorm(hidden_dim)
            self.window_position_head = nn.Sequential(
                nn.LayerNorm(self.window_size * hidden_dim),
                nn.Linear(self.window_size * hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
            )
            nn.init.normal_(self.window_position_embedding, mean=0.0, std=0.02)
        if self.aggregation == "mil_window" and self.window_representation == "dsca":
            self.peptide_local_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=self.window_size)
            self.hla_local_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
            self.peptide_local_norm = nn.LayerNorm(hidden_dim)
            self.hla_local_norm = nn.LayerNorm(hidden_dim)
            self.dsca_query = nn.Linear(hidden_dim, hidden_dim)
            self.dsca_key = nn.Linear(hidden_dim, hidden_dim)
            self.dsca_value = nn.Linear(hidden_dim, hidden_dim)
            self.dsca_context_norm = nn.LayerNorm(hidden_dim)
            self.dsca_window_fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
            )
        pair_feature_dim = hidden_dim * 8
        classifier_hidden = hidden_dim * 2
        self.interaction = nn.Sequential(
            nn.LayerNorm(pair_feature_dim),
            nn.Linear(pair_feature_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, hidden_dim),
            nn.GELU(),
        )
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.contrastive_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def set_hla_embeddings(self, embeddings):
        device = next(self.parameters()).device
        self.hla_embedding_table = torch.as_tensor(embeddings, dtype=torch.float32, device=device)

    def forward(
        self,
        peptide_embedding,
        peptide_mask,
        alpha_index,
        beta_index,
        return_embedding=False,
        return_core=False,
    ):
        if self.hla_embedding_table.numel() == 0:
            raise RuntimeError("HLA ESM2 embedding table is not set. Call model.set_hla_embeddings(...) first.")
        peptide_mask = peptide_mask.bool()
        peptide_tokens = self.peptide_encoder(peptide_embedding)

        alpha_embedding = self.hla_embedding_table[alpha_index.long()]
        beta_embedding = self.hla_embedding_table[beta_index.long()]
        alpha_tokens = self.hla_chain_encoder(alpha_embedding)
        beta_tokens = self.hla_chain_encoder(beta_embedding)

        hla_tokens = torch.cat([alpha_tokens, beta_tokens], dim=1)
        for block in self.hla_fusion_blocks:
            hla_tokens = block(hla_tokens)

        alpha_vector = self.pool(alpha_tokens)
        beta_vector = self.pool(beta_tokens)
        hla_vector = self.pool(hla_tokens)

        if self.aggregation == "mil_window":
            return self._forward_mil_window(
                peptide_tokens,
                peptide_mask,
                hla_tokens,
                hla_vector,
                alpha_vector,
                beta_vector,
                return_embedding=return_embedding,
                return_core=return_core,
            )

        peptide_vector = self.pool(peptide_tokens, peptide_mask)

        classifier_features = torch.cat(
            [
                peptide_vector,
                hla_vector,
                torch.abs(peptide_vector - hla_vector),
                peptide_vector * hla_vector,
                alpha_vector,
                beta_vector,
                torch.abs(alpha_vector - beta_vector),
                alpha_vector * beta_vector,
            ],
            dim=-1,
        )
        interaction_embedding = self.interaction(classifier_features)
        classifier_input = self.classifier_dropout(interaction_embedding)
        logits = self.classifier(classifier_input).squeeze(-1)

        if return_embedding:
            contrastive_embedding = F.normalize(self.contrastive_projection(interaction_embedding), dim=-1)
            if return_core:
                return logits, contrastive_embedding, None
            return logits, contrastive_embedding
        if return_core:
            return logits, None
        return logits

    def _residue_tokens_and_mask(self, peptide_tokens, peptide_mask):
        max_residue_length = peptide_tokens.size(1) - 2
        if max_residue_length <= 0:
            raise RuntimeError("peptide_embedding must include room for <cls>, residues, and <eos> tokens.")
        residue_tokens = peptide_tokens[:, 1:-1, :]
        token_lengths = peptide_mask.long().sum(dim=1)
        residue_lengths = (token_lengths - 2).clamp(min=0, max=max_residue_length)
        positions = torch.arange(max_residue_length, device=peptide_tokens.device).unsqueeze(0)
        residue_mask = positions < residue_lengths.unsqueeze(1)
        return residue_tokens, residue_mask, residue_lengths

    def _window_vectors_and_mask(self, peptide_tokens, peptide_mask, hla_tokens=None):
        residue_tokens, _, residue_lengths = self._residue_tokens_and_mask(peptide_tokens, peptide_mask)
        if residue_tokens.size(1) < self.window_size:
            raise RuntimeError(
                f"Configured window_size={self.window_size} exceeds available residue token slots={residue_tokens.size(1)}."
            )

        if self.window_representation == "dsca":
            if hla_tokens is None:
                raise ValueError("hla_tokens are required for dsca window representation.")
            peptide_local = self.peptide_local_conv(residue_tokens.transpose(1, 2)).transpose(1, 2)
            peptide_local = F.gelu(self.peptide_local_norm(peptide_local))
            hla_local = self.hla_local_conv(hla_tokens.transpose(1, 2)).transpose(1, 2)
            hla_local = F.gelu(self.hla_local_norm(hla_local))

            query = self.dsca_query(peptide_local)
            key = self.dsca_key(hla_local)
            value = self.dsca_value(hla_local)
            attention_scores = torch.matmul(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            hla_context = torch.matmul(attention_weights, value)
            hla_context = self.dsca_context_norm(hla_context)
            window_vectors = self.dsca_window_fusion(
                torch.cat(
                    [
                        peptide_local,
                        hla_context,
                        torch.abs(peptide_local - hla_context),
                        peptide_local * hla_context,
                    ],
                    dim=-1,
                )
            )
        else:
            # Shape after unfold is [batch, num_windows, hidden_dim, window_size].
            window_tokens = residue_tokens.unfold(dimension=1, size=self.window_size, step=1)
            if self.window_representation == "position_aware":
                window_tokens = window_tokens.permute(0, 1, 3, 2).contiguous()
                window_tokens = self.window_slot_norm(
                    window_tokens + self.window_position_embedding.view(1, 1, self.window_size, -1)
                )
                window_vectors = self.window_position_head(window_tokens.flatten(start_dim=2))
            else:
                window_vectors = window_tokens.mean(dim=-1)
        num_windows = window_vectors.size(1)
        starts = torch.arange(num_windows, device=peptide_tokens.device).unsqueeze(0)
        window_mask = starts + self.window_size <= residue_lengths.unsqueeze(1)

        # Very short peptides are rare for this task. Keep a valid fallback window
        # so BCE training/evaluation remains numerically stable.
        no_valid_window = ~window_mask.any(dim=1)
        if no_valid_window.any():
            window_mask = window_mask.clone()
            window_mask[no_valid_window, 0] = True
        return window_vectors, window_mask

    def _classifier_from_vectors(self, peptide_vector, hla_vector, alpha_vector, beta_vector):
        classifier_features = torch.cat(
            [
                peptide_vector,
                hla_vector,
                torch.abs(peptide_vector - hla_vector),
                peptide_vector * hla_vector,
                alpha_vector,
                beta_vector,
                torch.abs(alpha_vector - beta_vector),
                alpha_vector * beta_vector,
            ],
            dim=-1,
        )
        interaction_embedding = self.interaction(classifier_features)
        logits = self.classifier(self.classifier_dropout(interaction_embedding)).squeeze(-1)
        return logits, interaction_embedding

    def _forward_mil_window(
        self,
        peptide_tokens,
        peptide_mask,
        hla_tokens,
        hla_vector,
        alpha_vector,
        beta_vector,
        return_embedding=False,
        return_core=False,
    ):
        window_vectors, window_mask = self._window_vectors_and_mask(peptide_tokens, peptide_mask, hla_tokens=hla_tokens)
        batch_size, num_windows, hidden_dim = window_vectors.shape

        flat_window_vectors = window_vectors.reshape(batch_size * num_windows, hidden_dim)
        expanded_hla = hla_vector.unsqueeze(1).expand(-1, num_windows, -1).reshape(batch_size * num_windows, hidden_dim)
        expanded_alpha = alpha_vector.unsqueeze(1).expand(-1, num_windows, -1).reshape(batch_size * num_windows, hidden_dim)
        expanded_beta = beta_vector.unsqueeze(1).expand(-1, num_windows, -1).reshape(batch_size * num_windows, hidden_dim)

        flat_logits, flat_interaction = self._classifier_from_vectors(
            flat_window_vectors,
            expanded_hla,
            expanded_alpha,
            expanded_beta,
        )
        window_logits = flat_logits.view(batch_size, num_windows)
        window_interaction = flat_interaction.view(batch_size, num_windows, -1)
        masked_window_logits = window_logits.masked_fill(~window_mask, torch.finfo(window_logits.dtype).min)

        if self.mil_pooling == "max":
            logits, best_indices = masked_window_logits.max(dim=1)
            window_weights = torch.zeros_like(window_logits)
            window_weights.scatter_(1, best_indices.unsqueeze(1), 1.0)
        else:
            temperature = max(self.mil_temperature, 1e-6)
            window_weights = torch.softmax(masked_window_logits / temperature, dim=1)
            logits = torch.sum(window_weights * window_logits, dim=1)
            best_indices = masked_window_logits.argmax(dim=1)

        outputs = [logits]
        if return_embedding:
            bag_interaction = torch.sum(window_interaction * window_weights.unsqueeze(-1), dim=1)
            contrastive_embedding = F.normalize(self.contrastive_projection(bag_interaction), dim=-1)
            outputs.append(contrastive_embedding)
        if return_core:
            core_info = {
                "window_logits": window_logits,
                "window_mask": window_mask,
                "window_weights": window_weights,
                "core_start": best_indices + 1,
                "core_score": window_logits.gather(1, best_indices.unsqueeze(1)).squeeze(1),
            }
            outputs.append(core_info)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
