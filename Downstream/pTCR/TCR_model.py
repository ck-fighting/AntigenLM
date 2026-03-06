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
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
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
    def forward(self, pep_inputs, tcr_inputs):
        pep_outputs = pep_inputs.to(device)
        tcr_outputs = tcr_inputs.to(device)
        dec_self_attn_pad_mask = torch.LongTensor(
            np.zeros((pep_inputs.shape[0], pep_inputs.shape[1], tcr_inputs.shape[1]))
        ).bool().to(device)
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(pep_outputs, tcr_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns

class Mymodel_TCR(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=1024, n_layers=2, dropout=0.16):
        super().__init__()
        self.pep_proj = nn.Linear(768, d_model)
        self.encoder_T = Encoder(d_model, n_layers, n_heads, d_ff, dropout).to(device)
        self.cross = Cross_Attention(d_model, n_layers, n_heads, d_ff, dropout)
        # FC 的第一层 in_features = pep_max_len * d_model（保持你原来的设计）
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pep_max_len * d_model, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.fusion_proj = nn.Linear(2 * d_model, d_model)
        
    def forward(self, pep_emb, tcr_input):
        tcr_enc, tcr_attn = self.encoder_T(tcr_input)
        pep = self.pep_proj(pep_emb)
        pep2tcr, attn1 = self.cross(pep, tcr_enc)                  # [B, Lp, D]
        # === 关键：把序列长度自适应压到 pep_max_len ===
        # [B, Lp, D] -> [B, D, Lp] -> 自适应池化到 [B, D, pep_max_len] -> [B, pep_max_len, D]
        fusion_flat = pep2tcr.contiguous().view(pep2tcr.shape[0], -1)
        logits = self.fc(fusion_flat)
        return logits, attn1, fusion_flat

# esmc
# class Mymodel_TCR(nn.Module):
#     def __init__(self, d_model=128, n_heads=4, d_ff=1024, n_layers=2, dropout=0.16):
#         super().__init__()
#         self.pep_proj = nn.Linear(960, d_model)
#         self.encoder_T = Encoder(d_model, n_layers, n_heads, d_ff, dropout).to(device)
#         self.cross = Cross_Attention(d_model, n_layers, n_heads, d_ff, dropout)
#         # FC 的第一层 in_features = pep_max_len * d_model（保持你原来的设计）
#         self.fc = nn.Sequential(
#             nn.Linear(pep_max_len * d_model, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
#         self.fusion_proj = nn.Linear(2 * d_model, d_model)

#     def forward(self, pep_emb, tcr_input):
#         tcr_enc, tcr_attn = self.encoder_T(tcr_input)              # [B, Lt, D]
#         pep = self.pep_proj(pep_emb)                               # [B, Lp, D]  (Lp 可能是 512)
#         pep2tcr, attn1 = self.cross(pep, tcr_enc)                  # [B, Lp, D]

#         # === 关键：把序列长度自适应压到 pep_max_len ===
#         # [B, Lp, D] -> [B, D, Lp] -> 自适应池化到 [B, D, pep_max_len] -> [B, pep_max_len, D]
#         x = pep2tcr.transpose(1, 2)
#         x = F.adaptive_avg_pool1d(x, output_size=pep_max_len)
#         x = x.transpose(1, 2)

#         fusion_flat = x.reshape(x.size(0), -1)                     # [B, pep_max_len * D]
#         logits = self.fc(fusion_flat)
#         return logits, attn1, fusion_flat
