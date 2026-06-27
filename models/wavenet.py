# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def normalize_img(img):
    if torch.max(img) > 1 or torch.min(img) < 0:
        b, c, h, w = img.shape
        temp_img = img.view(b, c, h * w)
        im_max = torch.max(temp_img, dim=2)[0].view(b, c, 1)
        im_min = torch.min(temp_img, dim=2)[0].view(b, c, 1)

        temp_img = (temp_img - im_min) / (im_max - im_min + 1e-7)

        img = temp_img.view(b, c, h, w)

    return img

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False,
                 norm=True, act=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

        self.norm = nn.BatchNorm2d(out_ch) if norm else nn.Identity()
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.dim_head_q = dim_q // num_heads
        self.dim_head_kv = dim_kv // num_heads

        self.q_linear = nn.Linear(dim_q, dim_q)
        self.k_linear = nn.Linear(dim_kv, dim_q)
        self.v_linear = nn.Linear(dim_kv, dim_q)
        self.out_linear = nn.Linear(dim_q, dim_q)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        Q = self.q_linear(query)  # (batch_size, N_q, dim_q)
        K = self.k_linear(key)    # (batch_size, N_kv, dim_q)
        V = self.v_linear(value)  # (batch_size, N_kv, dim_q)

        Q = Q.view(batch_size, -1, self.num_heads, self.dim_head_q).transpose(1, 2)  # (batch_size, num_heads, N_q, dim_head_q)
        K = K.view(batch_size, -1, self.num_heads, self.dim_head_q).transpose(1, 2)  # (batch_size, num_heads, N_kv, dim_head_q)
        V = V.view(batch_size, -1, self.num_heads, self.dim_head_q).transpose(1, 2)  # (batch_size, num_heads, N_kv, dim_head_q)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_head_q ** 0.5)  # (batch_size, num_heads, N_q, N_kv)
        attn = F.softmax(scores, dim=-1)  # (batch_size, num_heads, N_q, N_kv)
        context = torch.matmul(attn, V)   # (batch_size, num_heads, N_q, dim_head_q)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_head_q)  # (batch_size, N_q, dim_q)

        out = self.out_linear(context)  # (batch_size, N_q, dim_q)

        return out

class FuseModule(nn.Module):
    def __init__(self, img_dim, text_dim, num_heads=8):
        super(FuseModule, self).__init__()
        self.cross_attn = CrossAttention(dim_q=img_dim, dim_kv=text_dim, num_heads=num_heads)
        self.linear = nn.Linear(img_dim, img_dim)
        self.norm = nn.LayerNorm(img_dim)

    def forward(self, img_features, text_features):
        batch_size, channels, H, W = img_features.shape
        img_seq = img_features.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, N, channels)
        N = img_seq.size(1)        
        text_seq = text_features  # (batch_size, seq_len, text_dim)       
        fused_seq = self.cross_attn(img_seq, text_seq, text_seq)  # (batch_size, N, channels)        
        fused_seq = self.norm(fused_seq + img_seq)  # (batch_size, N, channels)
        fused_features = fused_seq.permute(0, 2, 1).contiguous().view(batch_size, channels, H, W)  # (batch_size, channels, H, W)

        return fused_features

class CFM(nn.Module):
    def __init__(self, img_dim, text_dim, num_heads=8):
        super(CFM, self).__init__()
        self.cross_attn = CrossAttention(dim_q=img_dim, dim_kv=text_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(img_dim)
        self.gamma_beta_proj = nn.Sequential(
            nn.Linear(img_dim, img_dim * 2),
            nn.ReLU(),
            nn.Linear(img_dim * 2, img_dim * 2)
        )

    def forward(self, img_features, text_features):
        B, C, H, W = img_features.size()

        img_seq = img_features.view(B, C, -1).permute(0, 2, 1)  
        N = img_seq.size(1)
        fused_seq = self.cross_attn(img_seq, text_features, text_features)  
        fused_seq = self.norm(fused_seq + img_seq)
        pooled_feat = fused_seq.mean(dim=1) 

        gamma_beta = self.gamma_beta_proj(pooled_feat)  
        gamma, beta = gamma_beta.chunk(2, dim=-1)       

        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        modulated_feat = img_features * gamma + beta 

        return modulated_feat

class DownSimple(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DSConv(
            in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False,
            norm=True, act=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class UpSimple(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_deconv: bool = True):
        super().__init__()
        if use_deconv:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            )
        self.conv = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


def _build_haar_kernels(device, dtype):
    k = 1.0 / (2.0 ** 0.5)
    h = torch.tensor([k, k], device=device, dtype=dtype)
    g = torch.tensor([k, -k], device=device, dtype=dtype)
    LL = torch.einsum('i,j->ij', h, h)
    LH = torch.einsum('i,j->ij', h, g)
    HL = torch.einsum('i,j->ij', g, h)
    HH = torch.einsum('i,j->ij', g, g)
    return LL, LH, HL, HH


def _pad_to_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, C, H, W = x.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')  # (left,right,top,bottom)
    return x, (pad_h, pad_w)


def _unpad_even(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h:
        x = x[:, :, :-pad_h, :]
    if pad_w:
        x = x[:, :, :, :-pad_w]
    return x


def dwt_haar(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    x, pad_hw = _pad_to_even(x)
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    LL, LH, HL, HH = _build_haar_kernels(device, dtype)

    weight = torch.zeros(4 * C, 1, 2, 2, device=device, dtype=dtype)
    for c in range(C):
        weight[4 * c + 0, 0] = LL
        weight[4 * c + 1, 0] = LH
        weight[4 * c + 2, 0] = HL
        weight[4 * c + 3, 0] = HH

    y = F.conv2d(x, weight, bias=None, stride=2, padding=0, groups=C)
    return y, pad_hw


def idwt_haar(y: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    B, C4, H2, W2 = y.shape
    assert C4 % 4 == 0, "Channels must be 4*C: LL, LH, HL, HH."
    C = C4 // 4
    device, dtype = y.device, y.dtype

    LL, LH, HL, HH = _build_haar_kernels(device, dtype)

    weight_t = torch.zeros(4 * C, 1, 2, 2, device=device, dtype=dtype)
    for c in range(C):
        weight_t[4 * c + 0, 0] = LL
        weight_t[4 * c + 1, 0] = LH
        weight_t[4 * c + 2, 0] = HL
        weight_t[4 * c + 3, 0] = HH

    x = F.conv_transpose2d(y, weight_t, bias=None, stride=2, padding=0, groups=C)
    x = _unpad_even(x, pad_hw)
    return x

class UNet(nn.Module):
    def __init__(self, n_channels: int, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc   = DSConv(n_channels, 64)
        self.down1 = DownSimple(64, 128)
        self.down2 = DownSimple(128, 256)
        self.down3 = DownSimple(256, 512)
        self.down4 = DownSimple(512, 1024 // factor)

        bottleneck_ch = 1024 // factor
        self.up1 = UpSimple(in_ch=bottleneck_ch, skip_ch=512, out_ch=512 // factor, use_deconv=True)
        self.up2 = UpSimple(in_ch=512 // factor,   skip_ch=256, out_ch=256 // factor, use_deconv=True)
        self.up3 = UpSimple(in_ch=256 // factor,   skip_ch=128, out_ch=128 // factor, use_deconv=True)
        self.up4 = UpSimple(in_ch=128 // factor,   skip_ch=64,  out_ch=64,            use_deconv=True)

        self.outc = OutConv(64, n_channels)

    def forward(self, x_wav: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x_wav)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)

        logits_wav = self.outc(x)
        return logits_wav

def split_quads(y: torch.Tensor, base_ch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, C4, H, W = y.shape
    assert C4 == 4 * base_ch
    idx = [[], [], [], []]
    for c in range(base_ch):
        idx[0].append(4 * c + 0)
        idx[1].append(4 * c + 1)
        idx[2].append(4 * c + 2)
        idx[3].append(4 * c + 3)
    LL = y[:, idx[0], :, :]
    LH = y[:, idx[1], :, :]
    HL = y[:, idx[2], :, :]
    HH = y[:, idx[3], :, :]
    return LL, LH, HL, HH


def merge_quads(LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
    B, C, H, W = LL.shape
    out = torch.zeros(B, 4 * C, H, W, device=LL.device, dtype=LL.dtype)
    for c in range(C):
        out[:, 4 * c + 0, :, :] = LL[:, c, :, :]
        out[:, 4 * c + 1, :, :] = LH[:, c, :, :]
        out[:, 4 * c + 2, :, :] = HL[:, c, :, :]
        out[:, 4 * c + 3, :, :] = HH[:, c, :, :]
    return out

class SinCos2DPosEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "pos-enc dim must be divisible by 4"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        y_pos = torch.linspace(-1, 1, steps=H, device=device).unsqueeze(1).repeat(1, W)
        x_pos = torch.linspace(-1, 1, steps=W, device=device).unsqueeze(0).repeat(H, 1)
        pos = torch.stack([x_pos, y_pos], dim=0)  # (2, H, W)

        C4 = C // 4
        freqs = torch.arange(C4, device=device).float() / C4
        freqs = freqs.view(1, C4, 1, 1)

        posx = pos[0].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        posy = pos[1].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        sinx = torch.sin(2 * torch.pi * freqs * (posx + 1e-6))
        cosx = torch.cos(2 * torch.pi * freqs * (posx + 1e-6))
        siny = torch.sin(2 * torch.pi * freqs * (posy + 1e-6))
        cosy = torch.cos(2 * torch.pi * freqs * (posy + 1e-6))
        pe = torch.cat([sinx, cosx, siny, cosy], dim=1)
        return pe.expand(B, -1, H, W)


class LLTransformer(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int = 128, nhead: int = 4, num_layers: int = 2, mlp_ratio: float = 4.0):
        super().__init__()
        self.proj_in  = nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=False)
        self.pos_enc  = SinCos2DPosEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Conv2d(embed_dim, in_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        z = self.proj_in(x)                     # (B, E, H, W)
        z = z + self.pos_enc(z)                 # add position info
        z = z.flatten(2).transpose(1, 2)        # (B, HW, E)
        z = self.encoder(z)                     # Transformer
        z = z.transpose(1, 2).view(B, -1, H, W)
        z = self.proj_out(z)
        return x + z                            # residual


class TinyMamba(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 128, dw_kernel: int = 7):
        super().__init__()
        self.pw1 = nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False)
        self.dw  = nn.Conv2d(hidden, hidden, kernel_size=dw_kernel, padding=dw_kernel//2, groups=hidden, bias=False)
        self.gate= nn.Conv2d(in_ch, hidden, kernel_size=1, bias=True)
        self.pw2 = nn.Conv2d(hidden, in_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pw1(x)
        z = self.bn1(z)
        g = torch.sigmoid(self.gate(x))
        z = self.dw(z) * g
        z = self.act(z)
        z = self.pw2(z)
        z = self.bn2(z)
        return x + z


class ResidualSubband(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)



class WaveNet(nn.Module):
    def __init__(self,
                 n_channels,
                 clip_model,
                 ll_mode: str = "mamba",   # "transformer" | "mamba" | "none"
                 bilinear=True
                 ):
        super().__init__()
        assert n_channels == 3, "RGB only"
        assert ll_mode in ["transformer", "mamba", "none"]

        self.n_channels = n_channels
        self.modify_ll  = True
        self.ll_mode    = ll_mode
        
        self.factor = 2 if bilinear else 1
        
        self.clip_model = clip_model            
        
        self.text_feature_dim = self.clip_model.text_projection.shape[1]  

        self.image_proj = nn.Sequential(
            nn.Conv2d(self.n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, self.text_feature_dim)  
            )           
        encoder_layer = TransformerEncoderLayer(self.text_feature_dim, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6) 

        # Processor for top-level LL
        if ll_mode == "transformer":
            self.ll_proc = LLTransformer(in_ch=n_channels, embed_dim=128, nhead=4, num_layers=2)
        elif ll_mode == "mamba":
            self.ll_proc = TinyMamba(in_ch=n_channels, hidden=128)
        else:
            self.ll_proc = nn.Identity()

#        self.unet1 = UNet(self.n_channels)
#        self.unet2 = UNet(self.n_channels)
#        self.unet3 = UNet(self.n_channels)
        self.unet = UNet(self.n_channels)

        
#        self.unet_hl21 = UNet(self.n_channels)
#        self.unet_hl22 = UNet(self.n_channels)
#        self.unet_hl23 = UNet(self.n_channels)
        self.unet_hl2 = UNet(self.n_channels)

        
#        self.unet_lh21 = UNet(self.n_channels)
#        self.unet_lh22 = UNet(self.n_channels)
#        self.unet_lh23 = UNet(self.n_channels)
        self.unet_lh2 = UNet(self.n_channels)
        
#        self.unet_hh21 = UNet(self.n_channels)
#        self.unet_hh22 = UNet(self.n_channels)
#        self.unet_hh23 = UNet(self.n_channels)
        self.unet_hh2 = UNet(self.n_channels)
        
        self.inc   = DoubleConv(n_channels, 64)
#        self.down = DownSimple(64, 128)
        
#        self.up = UpSimple(in_ch=128, skip_ch=64, out_ch=64, use_deconv=True)
        self.outc = OutConv(64, n_channels)
        
        
        self.down1 = DownSimple(64, 128)
        self.down2 = DownSimple(128, 256)
        self.down3 = DownSimple(256, 512)
        self.down4 = DownSimple(512, 1024 // self.factor)
        bottleneck_ch = 1024 // self.factor
        self.CFM = CFM(img_dim=bottleneck_ch, text_dim=512, num_heads=8)
        # Decoder
        self.up1 = UpSimple(in_ch=bottleneck_ch, skip_ch=512, out_ch=512 // self.factor, use_deconv=True)
        self.up2 = UpSimple(in_ch=512 // self.factor,   skip_ch=256, out_ch=256 // self.factor, use_deconv=True)
        self.up3 = UpSimple(in_ch=256 // self.factor,   skip_ch=128, out_ch=128 // self.factor, use_deconv=True)
        self.up4 = UpSimple(in_ch=128 // self.factor,   skip_ch=64,  out_ch=64,            use_deconv=True)
        
        

    def forward(self, x, text_tokens):
        I = x  
                
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens).type(torch.float32)  # (batch_size, 512)
        text_embedding = text_features  # (batch_size, 512)
        
        image_embedding = self.image_proj(x) # (batch_size, 512)
        
        joint_embedding = torch.cat([image_embedding.unsqueeze(1), text_embedding.unsqueeze(1)], dim=1)  # (batch_size, 2, 512)
        joint_embedding = joint_embedding.permute(1, 0, 2)  # (sequence_length=2, batch_size, 512)
        
        transformer_output = self.transformer_encoder(joint_embedding) # (sequence_length=2, batch_size, 512)
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, 2, 512)

        image_features = transformer_output[:, 0, :]  # (batch_size, 512)
        text_features = transformer_output[:, 1, :]   # (batch_size, 512)
        
        text_features = text_features.unsqueeze(1)  # (batch_size, 1, 512)
        
        # Top-level DWT
        y1, pad1 = dwt_haar(x)                                   # (B, 12, H/2, W/2)
        LL1, LH1, HL1, HH1 = split_quads(y1, self.n_channels)    # each (B,3,H/2,W/2)

        if self.modify_ll:
            LL1_ = self.ll_proc(LL1) if self.ll_mode != "none" else LL1
        else:
            LL1_ = LL1  

        # Process HF with internal pyramid
        LH1_ = self.unet(LH1)
        HL1_ = self.unet(HL1)
        HH1_ = self.unet(HH1)
        
        y_lh2, pad_lh2 = dwt_haar(LH1_)
        LL_lh2, LH_lh2, HL_lh2, HH_lh2 = split_quads(y_lh2, self.n_channels)
        LH_lh2_ = self.unet_lh2(LH_lh2)
        HL_lh2_ = self.unet_lh2(HL_lh2)
        HH_lh2_ = self.unet_lh2(HH_lh2)
        y_lh2_ = merge_quads(LL_lh2, LH_lh2_, HL_lh2_, HH_lh2_)
        LH1_ = idwt_haar(y_lh2_, pad_lh2)
        
        y_hl2, pad_hl2 = dwt_haar(HL1_)
        LL_hl2, LH_hl2, HL_hl2, HH_hl2 = split_quads(y_hl2, self.n_channels)
        LH_hl2_ = self.unet_hl2(LH_hl2)
        HL_hl2_ = self.unet_hl2(HL_hl2)
        HH_hl2_ = self.unet_hl2(HH_hl2)
        y_hl2_ = merge_quads(LL_hl2, LH_hl2_, HL_hl2_, HH_hl2_)
        HL1_ = idwt_haar(y_hl2_, pad_hl2)
        
        y_hh2, pad_hh2 = dwt_haar(HH1_)        
        LL_hh2, LH_hh2, HL_hh2, HH_hh2 = split_quads(y_hh2, self.n_channels)
        LH_hh2_ = self.unet_hh2(LH_hh2)
        HL_hh2_ = self.unet_hh2(HL_hh2)
        HH_hh2_ = self.unet_hh2(HH_hh2)
        y_hh2_ = merge_quads(LL_hh2, LH_hh2_, HL_hh2_, HH_hh2_)
        HH1_ = idwt_haar(y_hh2_, pad_hh2)

        y1_ = merge_quads(LL1_, LH1_, HL1_, HH1_)                # (B, 12, H/2, W/2)
        recon = idwt_haar(y1_, pad1)                             # (B, 3, H, W)
        
        x1 = self.inc(recon)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5 = self.CFM(x5, text_features)
        
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)

        logits = self.outc(x)


        logits = logits + I
        out = normalize_img(logits)
        return out
