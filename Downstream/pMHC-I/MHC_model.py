import math
import numpy as np
import torch
from torch import nn

pep_max_len = 15
hla_max_len = 34
vocab = np.load('./data/data_dict.npy', allow_pickle=True).item()
vocab_size = len(vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        n_heads, d_k, d_v = self.n_heads, self.d_k, self.d_v
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (d_k ** 0.5)
        scores.masked_fill_(attn_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return self.layer_norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pep_inputs, hla_inputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(pep_inputs, hla_inputs, hla_inputs, dec_self_attn_mask)
        dec_outputs = self.dropout(dec_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=hla_max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model).to(device)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs1 = enc_inputs + self.dropout(enc_outputs)
        enc_outputs1 = self.layer_norm(enc_outputs1)
        enc_outputs = self.pos_ffn(enc_outputs1)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Cross_Attention(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.tgt_len = hla_max_len

    def forward(self, pep_inputs, hla_inputs):
        pep_outputs = pep_inputs.to(device)
        hla_outputs = hla_inputs.to(device)
        dec_self_attn_pad_mask = torch.zeros(
            pep_inputs.shape[0],
            pep_inputs.shape[1],
            hla_inputs.shape[1],
            dtype=torch.bool,
            device=device,
        )
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(pep_outputs, hla_outputs, dec_self_attn_pad_mask)
            pep_outputs = dec_outputs
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns


class Mymodel_HLA(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=1024, n_layers=2, dropout=0.16, pep_dim=768):
        super().__init__()
        self.pep_proj = nn.Linear(pep_dim, d_model)
        self.encoder_H = Encoder(d_model, n_layers, n_heads, d_ff, dropout).to(device)
        self.cross = Cross_Attention(d_model, n_layers, n_heads, d_ff, dropout)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pep_max_len * d_model, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.fusion_proj = nn.Linear(2 * d_model, d_model)
        for p in self.fusion_proj.parameters():
            p.requires_grad = False
        
    def forward(self, pep_emb, hla_input):
        hla_enc, _ = self.encoder_H(hla_input)
        pep = self.pep_proj(pep_emb)
        pep2hla, attn1 = self.cross(pep, hla_enc)
        fusion_flat = pep2hla.contiguous().view(pep2hla.shape[0], -1)
        logits = self.fc(fusion_flat)
        return logits, attn1, fusion_flat
