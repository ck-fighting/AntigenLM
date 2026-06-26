import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

pep_max_len = 15
tcr_max_len = 34
vocab = np.load('./data/data_dict.npy', allow_pickle=True).item()
vocab_size = len(vocab)
batch_size = 64
epochs = 50
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def nt_xent_loss(z1, z2, temperature=0.07):
    """
    Args:
        z1: [B, d]
        z2: [B, d]
        temperature: float scalar
    Returns:
        loss: scalar
    """
    # Step 1: L2 normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)        # [2B, d]

    # Step 2: 相似度矩阵
    sim_matrix = torch.matmul(z, z.T) / temperature    # [2B, 2B]

    # Step 3: 避免对角线参与分母
    mask = (~torch.eye(2 * batch_size, dtype=bool)).to(z.device)

    # Step 4: 正样本对（每i和i+B互为正对）
    pos_idx = torch.arange(batch_size, device=z.device)
    positives = torch.cat([
        sim_matrix[pos_idx, pos_idx + batch_size],
        sim_matrix[pos_idx + batch_size, pos_idx]
    ], dim=0)  # [2B]

    # Step 5: 对每个样本，分母是去掉自己之后的全部pair
    sim_matrix = sim_matrix.masked_fill(~mask, float('-inf'))   # 对角线为-inf

    # Step 6: 计算 log-sum-exp，防溢出
    denominator = torch.logsumexp(sim_matrix, dim=1)  # [2B]

    loss = - positives + denominator  # [2B]
    loss = loss.mean()
    
    return loss


class MyDataSet_TCR(Dataset):
    def __init__(self, pep_inputs, tcr_inputs, labels):
        super().__init__()
        self.pep_inputs = pep_inputs
        self.tcr_inputs = tcr_inputs
        self.labels = labels
    def __len__(self):
        return self.pep_inputs.shape[0]
    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.tcr_inputs[idx], self.labels[idx]


def transfer(y_prob, threshold):
    return np.array([[0, 1][x > threshold] for x in y_prob])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=34):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q,seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # print(pad_attn_mask.size())
    return pad_attn_mask.expand(batch_size, len_q, len_k)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # batch_size, n_heads, len_q, len_k
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # batch_size, n_heads, len_q, d_v
        return context, attn


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
        scores.masked_fill_(attn_mask, -1e9)
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
    def forward(self, pep_inputs, tcr_inputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(pep_inputs, tcr_inputs, tcr_inputs, dec_self_attn_mask)
        dec_outputs = self.dropout(dec_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=34):
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
        pe = self.pe[:x.size(0), :]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionalEncoding_padding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pad = torch.zeros(34, d_model)
        pad[:pe.shape[0], :] = pe
        pe = pad.unsqueeze(0).transpose(0, 1).to(device)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x.to(device) + self.pe[:x.size(0), :].to(device)
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model,n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff , dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model).to(device)
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,enc_self_attn_mask)
        enc_outputs1 = enc_inputs + self.dropout(enc_outputs)
        enc_outputs1 = self.layer_norm(enc_outputs1)
        enc_outputs = self.pos_ffn(enc_outputs1)
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self,d_model,n_layers, n_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model)
        self.motif_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2)
            for k in (3, 5, 7)
        ])
        self.motif_norm = nn.LayerNorm(d_model)
        self.motif_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.motif_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        valid_mask = enc_inputs.ne(0).unsqueeze(-1).type_as(enc_outputs)
        conv_in = enc_outputs.transpose(1, 2)
        motif_outputs = [
            F.gelu(conv(conv_in)).transpose(1, 2)
            for conv in self.motif_convs
        ]
        motif = torch.stack(motif_outputs, dim=0).mean(dim=0)
        motif = self.motif_norm(motif)
        gate = self.motif_gate(torch.cat([enc_outputs, motif], dim=-1))
        enc_outputs = enc_outputs + self.motif_dropout(gate * motif)
        enc_outputs = enc_outputs * valid_mask
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        # print(enc_inputs.size())
        # print(enc_self_attn_mask.size())
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: batch_size, src_len, d_model, enc_self_attn: batch_size, n_heads, src_len, src_len
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Cross_Attention(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.tgt_len = tcr_max_len
    def forward(self, pep_inputs, tcr_inputs, tcr_pad_mask=None):
        pep_outputs = pep_inputs.to(device)
        tcr_outputs = tcr_inputs.to(device)
        if tcr_pad_mask is None:
            dec_self_attn_pad_mask = torch.zeros(
                (pep_inputs.shape[0], pep_inputs.shape[1], tcr_inputs.shape[1]),
                dtype=torch.bool,
                device=pep_inputs.device,
            )
        else:
            dec_self_attn_pad_mask = tcr_pad_mask.bool().to(pep_inputs.device)
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(pep_outputs, tcr_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns


class AttentionPooling(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1, bias=False),
        )

    def forward(self, x, mask=None):
        scores = self.score(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(x * weights, dim=1)


class Mymodel_TCR(nn.Module):
    def __init__(
        self,
        pep_dim=768,
        d_model=128,
        n_heads=4,
        d_ff=1024,
        n_layers=2,
        dropout=0.16,
        pep_input_norm=False,
        pep_input_scale=1.0,
    ):
        super().__init__()
        self.pep_input_norm = pep_input_norm
        self.pep_input_scale = pep_input_scale
        self.pep_proj = nn.Linear(pep_dim, d_model)
        self.encoder_T = Encoder(d_model, n_layers, n_heads, d_ff, dropout).to(device)
        self.cross = Cross_Attention(d_model, n_layers, n_heads, d_ff, dropout)
        self.tcr_pool = AttentionPooling(d_model, dropout)
        self.pep_pool = AttentionPooling(d_model, dropout)
        fc_in_dim = pep_max_len * d_model + 4 * d_model

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.fusion_proj = nn.Linear(2 * d_model, d_model)
        
    def forward(self, pep_emb, tcr_input):
        tcr_enc, tcr_attn = self.encoder_T(tcr_input)
        if self.pep_input_norm:
            pep_emb = F.layer_norm(pep_emb, (pep_emb.size(-1),))
        if self.pep_input_scale != 1.0:
            pep_emb = pep_emb * self.pep_input_scale
        pep = self.pep_proj(pep_emb)
        tcr_valid_mask = tcr_input.ne(0)
        tcr_pad_mask = ~tcr_valid_mask.unsqueeze(1).expand(-1, pep.size(1), -1)
        pep2tcr, attn1 = self.cross(pep, tcr_enc, tcr_pad_mask=tcr_pad_mask)  # [B, Lp, D]
        # === 关键：把序列长度自适应压到 pep_max_len ===
        # [B, Lp, D] -> [B, D, Lp] -> 自适应池化到 [B, D, pep_max_len] -> [B, pep_max_len, D]
        fusion_flat = pep2tcr.contiguous().view(pep2tcr.shape[0], -1)
        tcr_vec = self.tcr_pool(tcr_enc, mask=tcr_valid_mask)
        pep_vec = self.pep_pool(pep)
        match_features = torch.cat([
            fusion_flat,
            tcr_vec,
            pep_vec,
            torch.abs(tcr_vec - pep_vec),
            tcr_vec * pep_vec,
        ], dim=-1)
        logits = self.fc(match_features)
        return logits, attn1, match_features
