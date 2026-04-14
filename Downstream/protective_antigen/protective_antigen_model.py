import torch.nn as nn
import torch
import math
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)
# -------------------------------
# ResNet1D 分类器
# -------------------------------
class ResNet1DClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=3):
        super().__init__()
        self.initial_conv = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.resblock1 = ResidualBlock(256, 256)
        self.resblock2 = ResidualBlock(256, 256)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.initial_conv(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
    

#CNN+MLP
class CNNMLPClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveMaxPool1d(1)  # [B, 64, 1]
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, L, C] -> [B, C, L]
        x = self.conv_layers(x)
        x = self.pool(x).squeeze(-1)  # [B, 64]
        return self.classifier(x)






class ContextPooling(nn.Module):
    def __init__(self,seq_len,in_dim=768):
        super(ContextPooling,self).__init__()
        self.seq_len=seq_len
        self.conv=nn.Sequential(
            nn.Conv1d(in_dim,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,2,3,stride=1,padding=1),
            nn.LayerNorm((2,seq_len)),
            nn.LeakyReLU(True),
        )

    def _local_normal(self,s,center,r=0.1):
        PI=3.1415926
        std_=(r*self.seq_len*s[:,center]).unsqueeze(1) #[B,1]
        mean_=center
        place=torch.arange(self.seq_len).float().repeat(std_.shape[0],1).to(device) # [B,L]

        #print(std_)

        ret=pow(2*PI,-0.5)*torch.pow(std_,-1)*torch.exp(-torch.pow(place-mean_,2)/(1e-5+2*torch.pow(std_,2)))

        #ret-=torch.max(ret,dim=1)[0].unsqueeze(1)
        #ret=torch.softmax(ret,dim=1)

        ret/=torch.max(ret,dim=1)[0].unsqueeze(1)


        return ret

    def forward(self,feats): # feats: [B,L,1024]
        feats_=feats.permute(0,2,1)
        feats_=self.conv(feats_) #output: [B,2,L]
        s,w=feats_[:,0,:].squeeze(1),feats_[:,1,:].squeeze(1) #[B,L]
        s=torch.softmax(s,1)
        w=torch.softmax(w,1)

        out=[]

        for i in range(self.seq_len):
            w_=self._local_normal(s,i)*w
            w_=w_.unsqueeze(2) # [B,L,1]
            out.append((w_*feats).sum(1,keepdim=True)) # w_ [B,L,1], feats [B,L,1024]

        out=torch.cat(out,dim=1) # [B,L,1024]
        return out

# class SelfAttention(nn.Module):
#     def __init__(self, hidden_size, num_attention_heads, output_size, dropout_prob):   
#         super(SelfAttention, self).__init__()

#         assert output_size%num_attention_heads==0

#         self.num_attention_heads = num_attention_heads
#         #self.attention_head_size = int(hidden_size / num_attention_heads)
#         self.attention_head_size= int(output_size/num_attention_heads)
#         self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        
#         self.query = nn.Linear(hidden_size, self.all_head_size)
#         self.key = nn.Linear(hidden_size, self.all_head_size)
#         self.value = nn.Linear(hidden_size, self.all_head_size)
        
#         # dropout
#         self.dropout = nn.Dropout(dropout_prob)

#     def transpose_for_scores(self, x):
#         # INPUT:  x'shape = [bs, seqlen, hid_size]
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, hidden_states):
        
#         mixed_query_layer = self.query(hidden_states)   # [bs, seqlen, hid_size]
#         mixed_key_layer = self.key(hidden_states)       # [bs, seqlen, hid_size]
#         mixed_value_layer = self.value(hidden_states)   # [bs, seqlen, hid_size]

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)   # [bs, 8, seqlen, seqlen]

#         attention_probs = nn.Softmax(dim=-1)(attention_scores)    # [bs, 8, seqlen, seqlen]

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs).to(torch.float32)
        
#         context_layer = torch.matmul(attention_probs, value_layer)   # [bs, 8, seqlen, 16]
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # [bs, seqlen, 8, 16]
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)   # [bs, seqlen, 128]
#         context_layer = context_layer.view(*new_context_layer_shape)
#         return context_layer

# class SoluModel(nn.Module):
#     def __init__(self, seq_len ,in_dim=768, sa_out=768, conv_out=768):
#         super(SoluModel, self).__init__()
        
#         #self.self_attention=SelfAttention(in_dim,4,sa_out,0.6) # input: [B,L,1024] output: [B,L,1024]
#         self.contextpooling=ContextPooling(seq_len,in_dim)

#         self.conv=nn.Sequential( #input: [B,1024,L] output: [B,1024,L]
#             nn.Conv1d(in_dim,in_dim*2,3,stride=1,padding=1),
#             nn.LayerNorm((in_dim*2,seq_len)),
#             nn.LeakyReLU(True),

#             nn.Conv1d(in_dim*2,in_dim*2,3,stride=1,padding=1),
#             nn.LayerNorm((in_dim*2,seq_len)),
#             nn.LeakyReLU(True),

#             nn.Conv1d(in_dim*2,conv_out,3,stride=1,padding=1),
#             nn.LayerNorm((conv_out,seq_len)),
#             nn.LeakyReLU(True),
#         )

#         self.cls_dim=sa_out+conv_out

#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.6),
#             nn.Linear(self.cls_dim, self.cls_dim // 4),
#             nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

#             nn.Dropout(p=0.6),
#             nn.Linear(self.cls_dim//4, self.cls_dim // 4),
#             nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

#             nn.Dropout(p=0.6),
#             nn.Linear(self.cls_dim//4, self.cls_dim // 64),
#             nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
#             nn.Linear(self.cls_dim // 64, 1),
            
#             nn.Sigmoid())


        
#         self._initialize_weights()

#     def forward(self, feats):
#         out_sa=self.contextpooling(feats)+feats

#         out_conv=self.conv(feats.permute(0,2,1))
#         out_conv=out_conv.permute(0,2,1)+feats

#         out=torch.cat([out_sa,out_conv],dim=2)
#         out=torch.max(out,dim=1)[0].squeeze()

#         cls_out = self.classifier(out)
        

#         #print(cls_out)

#         return cls_out

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, output_size, dropout_prob):   
        super(SelfAttention, self).__init__()
        assert output_size % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(output_size / num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs).to(torch.float32)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer  # [B, L, all_head_size]

# ----------------- ContextPooling -------------------
class ContextPooling(nn.Module):
    def __init__(self, seq_len, in_dim=768):
        super(ContextPooling, self).__init__()
        self.seq_len = seq_len
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, in_dim*2, 3, stride=1, padding=1),
            nn.LayerNorm((in_dim*2, seq_len)),
            nn.LeakyReLU(True),
            nn.Conv1d(in_dim*2, in_dim*2, 3, stride=1, padding=1),
            nn.LayerNorm((in_dim*2, seq_len)),
            nn.LeakyReLU(True),
            nn.Conv1d(in_dim*2, 2, 3, stride=1, padding=1),
            nn.LayerNorm((2, seq_len)),
            nn.LeakyReLU(True),
        )

    def _local_normal(self, s, center, r=0.1):
        PI = 3.1415926
        std_ = (r * self.seq_len * s[:, center]).unsqueeze(1)  # [B,1]
        mean_ = center
        place = torch.arange(self.seq_len).float().repeat(std_.shape[0],1).to(s.device)  # [B,L]
        ret = pow(2*PI,-0.5) * torch.pow(std_, -1) * torch.exp(-torch.pow(place-mean_,2)/(1e-5+2*torch.pow(std_,2)))
        ret /= torch.max(ret,dim=1)[0].unsqueeze(1)
        return ret

    def forward(self, feats):  # feats: [B, L, in_dim]
        feats_ = feats.permute(0,2,1)  # [B, D, L]
        feats_ = self.conv(feats_)     # [B, 2, L]
        s, w = feats_[:,0,:].squeeze(1), feats_[:,1,:].squeeze(1)  # [B,L]
        s = torch.softmax(s,1)
        w = torch.softmax(w,1)
        out = []
        for i in range(self.seq_len):
            w_ = self._local_normal(s, i) * w
            w_ = w_.unsqueeze(2)  # [B,L,1]
            out.append((w_*feats).sum(1, keepdim=True))
        out = torch.cat(out, dim=1)  # [B, L, in_dim]
        return out

# ----------------- SoluModel -------------------
class SoluModel(nn.Module):
    def __init__(self, seq_len, in_dim=768, sa_out=768, conv_out=768, attn_heads=4, attn_dropout=0.1):
        super(SoluModel, self).__init__()
        self.contextpooling = ContextPooling(seq_len, in_dim)

        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, in_dim*2, 3, stride=1, padding=1),
            nn.LayerNorm((in_dim*2, seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2, in_dim*2, 3, stride=1, padding=1),
            nn.LayerNorm((in_dim*2, seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2, conv_out, 3, stride=1, padding=1),
            nn.LayerNorm((conv_out, seq_len)),
            nn.LeakyReLU(True),

        )

        # ---- Self-Attention层 ----
        self.self_attention = SelfAttention(hidden_size=in_dim,
                                            num_attention_heads=attn_heads,
                                            output_size=sa_out,
                                            dropout_prob=attn_dropout)

        # 池化方式可选，默认max
        self.cls_dim = sa_out + conv_out + sa_out # sa_out: attention输出，conv_out: 卷积池化输出
        self.bn = nn.BatchNorm1d(self.cls_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ReLU(True),
            nn.Dropout(p=0.7),
            nn.Linear(self.cls_dim // 4, 1),
        )
        self._initialize_weights()

    def forward(self, feats):
        # feats: [B, L, D]
        out_sa = self.self_attention(feats)          # [B, L, sa_out]
        out_sa_pool = torch.max(out_sa, dim=1)[0]    # [B, sa_out]

        out_cp = self.contextpooling(feats)          # [B, L, D]
        out_cp_pool = torch.max(out_cp, dim=1)[0]    # [B, D]

        out_conv = self.conv(feats.permute(0, 2, 1)).permute(0, 2, 1) + feats
        out_conv_pool = torch.max(out_conv, dim=1)[0]  # [B, conv_out]

        out = torch.cat([out_sa_pool, out_conv_pool, out_cp_pool], dim=1)
        out = self.bn(out)
        cls_out = self.classifier(out)
        return cls_out, out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
