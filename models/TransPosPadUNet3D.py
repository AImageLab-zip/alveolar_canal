import torch
import torch.nn as nn
from torch import dropout, nn, Tensor
import math
import numpy as np

# def get_angles(pos, i, d_model):
#   angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / torch.tensor(d_model, dtype=torch.float32))
#   return pos * angle_rates  

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout = 0.1, max_len= 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = get_angles(torch.arange(max_len).unsqueeze(-1), torch.arange(d_model).unsqueeze(0), d_model)
#         pe[:, 0::2]  = torch.sin(pe[:, 0::2])
#         pe[:, 1::2]  = torch.cos(pe[:, 1::2])
#         self.register_buffer('pe', pe)
#         print(pe.shape)
#     def forward(self, x):
#         x = x + self.pe[:x.size(1)]
#         return self.dropout(x)     

class SqueezeExcitation(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation= torch.nn.ReLU,
        scale_activation= torch.nn.Sigmoid,
        ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool3d(1)
        self.fc1 = torch.nn.Conv3d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv3d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()
    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)
    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class PositionalEncodingTorchDoc(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)  

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.unsqueeze(torch.arange(max_len), 0)
        self.register_buffer('position', position)
        self.embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        # print(self.embedding.weight)
        pos_emb = self.embedding(self.position)
        x = x + pos_emb
        return self.dropout(x)  

# class PositionalEmbedding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=1000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         # position = torch.unsqueeze(torch.arange(max_len), 0)
#         self.embedding = nn.Parameter(torch.rand(max_len, d_model))
#         # self.embedding = nn.Embedding(max_len, d_model)
        
#     def forward(self, x):
#         print(self.embedding, self.embedding.shape)
#         pos_emb = self.embedding#(self.position)
#         x = x + pos_emb
#         return self.dropout(x)          

      

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, seq_len, rate=0.1, batch_first=True):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embed_dim, batch_first=batch_first)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(),nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm([seq_len, embed_dim], eps=1e-6)
        self.layernorm2 = nn.LayerNorm([seq_len, embed_dim], eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs, masked=None):
        if masked is not None:
            attn_output, _ = self.att(inputs, inputs, inputs, key_padding_mask=masked, average_attn_weights=False)
        else:
            attn_output, _ = self.att(inputs, inputs, inputs, average_attn_weights=False)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class TransPosPadUNet3D(nn.Module):
    def __init__(self, n_classes, emb_shape, in_ch, size=32, n_layers=4, num_head=8, max_len=15**3, pos_enc="sin"):
        self.n_classes = n_classes
        self.in_ch = in_ch
        self.size = size
        super(TransPosPadUNet3D, self).__init__()

        self.emb_shape = torch.as_tensor(emb_shape)
        # self.pos_emb_layer = nn.Linear(6, torch.prod(self.emb_shape).item())
        self.ec0 = self.conv3Dblock(self.in_ch, size, groups=1)
        self.ec1 = self.conv3Dblock(size, size*2, kernel_size=3, padding=1, groups=1)  # third dimension to even val
        self.ec2 = self.conv3Dblock(size*2, size*2, groups=1)
        self.ec3 = self.conv3Dblock(size*2, size*4, groups=1)
        self.ec4 = self.conv3Dblock(size*4, size*4, groups=1)
        self.ec5 = self.conv3Dblock(size*4, size*8, groups=1)
        self.ec6 = self.conv3Dblock(size*8, size*8, groups=1)
        self.ec7 = self.conv3Dblock(size*8, size*16, groups=1)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        if pos_enc=="sin":
            self.pos_encoder = PositionalEncodingTorchDoc(size*16, max_len=max_len)
        elif pos_enc=="emb":
            self.pos_encoder = PositionalEmbedding(size*16, max_len=max_len)
        else:
            raise ValueError(f'pos_enc \'{pos_enc}\' not found')

        self.batchnorm = nn.BatchNorm1d(size*16)
        self.encoder_layers = nn.ModuleList([TransformerBlock(embed_dim=size*16, num_heads=num_head, 
                                            ff_dim=(size*16)*2, seq_len=(self.emb_shape[0]*self.emb_shape[1]*self.emb_shape[2]), 
                                            rate=0.1, batch_first=True) for _ in range(n_layers)])

        self.dc9 = nn.ConvTranspose3d(size*16, size*16, kernel_size=2, stride=2)
        self.dc8 = self.conv3Dblock(size*8 + size*16, size*8, kernel_size=3, stride=1, padding=1)
        self.dc7 = self.conv3Dblock(size*8, size*8, kernel_size=3, stride=1, padding=1)
        self.dc6 = nn.ConvTranspose3d(size*8, size*8, kernel_size=2, stride=2)
        self.dc5 = self.conv3Dblock(size*4 + size*8, size*4, kernel_size=3, stride=1, padding=1)
        self.dc4 = self.conv3Dblock(size*4, size*4, kernel_size=3, stride=1, padding=1)
        self.dc3 = nn.ConvTranspose3d(size*4, size*4, kernel_size=2, stride=2)
        self.dc2 = self.conv3Dblock(size*2 + size*4, size*2, kernel_size=3, stride=1, padding=1)
        self.dc1 = self.conv3Dblock(size*2, size*2, kernel_size=3, stride=1, padding=1)

        self.final = nn.ConvTranspose3d(size*2, n_classes, kernel_size=3, padding=1, stride=1)
        initialize_weights(self)

    def conv3Dblock(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), groups=1, padding_mode='replicate'):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
                # nn.SiLU()
        )

    def forward(self, x, emb_codes): # batch, channel, D, H, W
        h = self.ec0(x)
        feat_0 = self.ec1(h)
        h = self.pool0(feat_0)
        h = self.ec2(h)
        feat_1 = self.ec3(h)

        h = self.pool1(feat_1)
        h = self.ec4(h)
        feat_2 = self.ec5(h)

        h = self.pool2(feat_2)
        h = self.ec6(h)
        h = self.ec7(h)

        h = h.reshape(h.size(0), h.size(1), h.size(2)*h.size(3)*h.size(4))  # (BS, CH, D, H, W) -> (BS, CH, D*H*W)
        h = self.batchnorm(h)
        h = torch.permute(h, (0,2,1))                                       # (BS, CH, D*H*W) -> (BS,  D*H*W, CH)
        # h = h * math.sqrt(self.size*16)
        h = self.pos_encoder(h)
        for enc_layer in self.encoder_layers:
            h = enc_layer(h)
        h = torch.permute(h, (0,2,1))                                       # (BS,  D*H*W, CH) -> (BS, CH, D*H*W)
        h = h.reshape(h.size(0), h.size(1), int(round(np.cbrt(h.size(2)))), 
                        int(round(np.cbrt(h.size(2)))), int(round(np.cbrt(h.size(2)))))   # (BS, CH, D*H*W) -> (BS, CH, D, H, W)

        h = torch.cat((self.dc9(h), feat_2), dim=1)

        h = self.dc8(h)
        h = self.dc7(h)

        h = torch.cat((self.dc6(h), feat_1), dim=1)
        h = self.dc5(h)
        h = self.dc4(h)

        h = torch.cat((self.dc3(h), feat_0), dim=1)
        h = self.dc2(h)
        h = self.dc1(h)
        
        h = self.final(h)
        return torch.sigmoid(h)
