import torch
import torch.nn as nn
from torch import dropout, nn, Tensor
import math
import numpy as np
from .layers.PositionalEncoding import PositionalEncoding
from .layers.PositionalEmbedding import PositionalEmbedding
from .layers.ConcatAbsolutePos import ConcatAbsolutePos

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, seq_len, rate=0.1, batch_first=True, mem_len=128, ABS=True):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embed_dim, batch_first=batch_first)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(),nn.Linear(ff_dim, embed_dim)
        )
       
        if ABS:
            self.ABS = 1
        else:
            self.ABS = 0

        self.layernorm1 = nn.LayerNorm([seq_len+mem_len+self.ABS, embed_dim], eps=1e-6)
        self.layernorm2 = nn.LayerNorm([seq_len+mem_len+self.ABS, embed_dim], eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.seq_len = seq_len
        self.mem_len = mem_len
       
        if mem_len>0:
            mem_pos = torch.unsqueeze(torch.arange(mem_len), 0)
            self.register_buffer('mem_pos', mem_pos)
            self.embedding = nn.Embedding(mem_len, embed_dim)

    def get_memory(self,):
        return self.embedding(self.mem_pos)        

    def forward(self, inputs, masked=None):
        batch_size = inputs.size(dim=0)
        if self.mem_len > 0:
            memory = self.get_memory()
            memory = memory.repeat(batch_size, 1, 1)
            inputs = torch.cat((inputs,memory), dim=1)
        if masked is not None:
            attn_output, _ = self.att(inputs, inputs, inputs, key_padding_mask=masked, average_attn_weights=False)
        else:
            attn_output, _ = self.att(inputs, inputs, inputs, average_attn_weights=False)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        output = self.layernorm2(out1 + ffn_output)
        return output[:, :self.seq_len+self.ABS, :]

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

class MemTransPosPadUNet3D(nn.Module):
    def __init__(self, n_classes, emb_shape, in_ch, size=32, n_layers=4, num_head=8, max_len=10**3, pos_enc="sin", mem_len=10, ABS=True):
        self.n_classes = n_classes
        self.in_ch = in_ch
        self.size = size
        super(MemTransPosPadUNet3D, self).__init__()
        self.concatabspos = ConcatAbsolutePos()
        self.max_len = max_len
        self.mem_len = mem_len
        self.ABS     = ABS
        self.emb_shape = torch.as_tensor(emb_shape)
        self.pos_emb_layer = nn.Linear(6, size*16)
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
            self.pos_encoder = PositionalEncoding(size*16, max_len=max_len)
        elif pos_enc=="emb":
            self.pos_encoder = PositionalEmbedding(size*16, max_len=max_len)
        else:
            raise ValueError(f'pos_enc \'{pos_enc}\' not found')

        self.batchnorm = nn.BatchNorm1d(size*16)
        self.encoder_layers = nn.ModuleList([TransformerBlock(embed_dim=size*16, num_heads=num_head, 
                                            ff_dim=(size*16)*2, seq_len=(self.emb_shape[0]*self.emb_shape[1]*self.emb_shape[2]), 
                                            rate=0.1, batch_first=True, mem_len=mem_len, ABS=self.ABS) for _ in range(n_layers)])

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

        if self.ABS:
            abs_pos = self.pos_emb_layer(emb_codes)
            h = self.concatabspos(h, abs_pos)

        for enc_layer in self.encoder_layers:
            h = enc_layer(h)

        if self.ABS:            
            h = h[:, :self.max_len, :]

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


if __name__ == "__main__":
    model = MemTransPosPadUNet3D()
