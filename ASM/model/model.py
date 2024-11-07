from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F

from AToM.model.rotary_embedding_torch import RotaryEmbedding
from AToM.model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like


class AdaINLayer(nn.Module):
    def __init__(self, input_nc, modulation_nc):
        super().__init__()

        self.InstanceNorm2d = nn.InstanceNorm2d(input_nc, affine=False)

        nhidden = 128
        use_bias = True

        self.mlp_shared = nn.Sequential(
            nn.Linear(modulation_nc, nhidden, bias=use_bias),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, input_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, input_nc, bias=use_bias)

    def forward(self, input, modulation_input):
        # print("input:", input.shape)

        is_3d = input.dim() == 3
        if is_3d:
            input = input.permute(0, 2, 1).unsqueeze(2)
        # Part 1. generate parameter-free normalized activations
        normalized = self.InstanceNorm2d(input)

        # Part 2. produce scaling and bias conditioned on feature
        modulation_input = modulation_input.view(modulation_input.size(0), -1)
        actv = self.mlp_shared(modulation_input)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        # gamma = gamma.view(*gamma.size()[:2], 1,1)
        # beta = beta.view(*beta.size()[:2], 1,1)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        # gamma = gamma.view(gamma.size(0), 1, gamma.size(1))
        # beta = beta.view(beta.size(0), 1, beta.size(1))
        # print(f"normalized:{normalized.shape},gamma:{gamma.shape},beta:{beta.shape}")
        out = normalized * (1 + gamma) + beta
        return out


class AdaIN(torch.nn.Module):

    def __init__(self, input_channel, modulation_channel, kernel_size=3, stride=1, padding=1):
        super(AdaIN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride,
                                      padding=padding)
        # self.conv_1 = nn.Conv1d(16, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride,
                                      padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.adain_layer_1 = AdaINLayer(input_channel, modulation_channel)
        self.adain_layer_2 = AdaINLayer(input_channel, modulation_channel)

    def forward(self, x, modulation):
        x = self.adain_layer_1(x, modulation)
        x = self.leaky_relu(x)
        # x = x.permute(0, 2, 1).unsqueeze(2)

        x = self.conv_1(x)
        x = self.adain_layer_2(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_2(x)
        # print("x111:", x.shape)
        is_3d = x.dim() == 4
        if is_3d:
            x = x.squeeze(2).permute(0, 2, 1)
            # x = x.permute(0, 2, 1).unsqueeze(3)
            # print("xxxxxx", x.shape)

        return x


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(nn.Mish(), nn.Linear(embed_channels, embed_channels * 2))

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,  # 特征向量的维度
            nhead: int,  # 多头注意力机制中的头数
            dim_feedforward: int = 2048,  # 前馈网络中间层的维度，默认为2048
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            layer_norm_eps: float = 1e-5,  # 层归一化的epsilon值，用于防止除零错误，默认为1e-5
            batch_first: bool = False,
            norm_first: bool = True,
            device=None,
            dtype=None,
            rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
            self,
            src: Tensor,  # 输入的源数据张量
            src_mask: Optional[Tensor] = None,  # 用于多头注意力机制的掩码，可选，用于遮蔽（masking）不应被考虑的数据部分
            src_key_padding_mask: Optional[Tensor] = None,  # 针对源数据的键填充掩码，可选，用于在注意力机制中忽略特定的键值
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.relu,
            layer_norm_eps=1e-5,
            batch_first=False,
            norm_first=True,
            device=None,
            dtype=None,
            rotary=None,
    ):
        super().__init__()
        # print(f"d_model:{d_model}")
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, d_model * 2)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.adain1 = AdaIN(d_model, d_model)
        self.adain2 = AdaIN(d_model, d_model)
        self.adain3 = AdaIN(d_model, d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

        self.depthwise_conv = nn.Conv2d(in_channels=470, out_channels=470, kernel_size=(3, 1), padding=(1, 0),
                                        groups=470)
        self.pointwise_conv = nn.Conv2d(in_channels=470, out_channels=d_model, kernel_size=1)
        self.interaction = nn.Linear(d_model * 2, d_model)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dim_reduction_conv = torch.nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1)

    # x, cond, t
    def forward(
            self,
            tgt,
            memory,
            lip_t,
            nonlip_t,
            face_memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
    ):
        # print(
        #     f"filmpre:lip_t:{lip_t.shape},,nonlip_t:{nonlip_t.shape},,tgt:{tgt.shape},,memory:{memory.shape},,face_memory:{face_memory.shape}")
        partition = tgt.shape[2]
        temp = tgt[:, :, :]
        lip, face = temp[:, :, : partition // 2], temp[:, :, partition // 2:]
        lip = lip.view(tgt.shape[0], tgt.shape[1], -1)
        face = face.view(tgt.shape[0], tgt.shape[1], -1)
        print("model_change")
        if self.norm_first:
            # print(face.shape)
            face1 = self._sa_block(self.norm1(face), tgt_mask, tgt_key_padding_mask)
            lip1 = self._sa_block(self.norm1(lip), tgt_mask, tgt_key_padding_mask)
            lip = torch.cat([lip, lip1], dim=-1)  # 在最后一个维度进行连接
            face = torch.cat([face, face1], dim=-1)
            lip = lip.permute(0, 2, 1)  # 将维度从 [16, 156, 512] 变为 [16, 512, 156] 以适应 Conv1d
            # print(lip.shape)
            lip = self.dim_reduction_conv(lip)
            lip = lip.permute(0, 2, 1)  # 将维度恢复为 [16, 156, 256]

            face = face.permute(0, 2, 1)
            face = self.dim_reduction_conv(face)
            face = face.permute(0, 2, 1)
            # print(f"face:{face.shape},lip:{lip.shape}")

            face2 = self._sa_block(self.norm2(face), tgt_mask, tgt_key_padding_mask)
            lip2 = self._mha_block(self.norm2(lip), memory, memory_mask, memory_key_padding_mask)

            lip = torch.cat([lip, lip2], dim=-1)
            face = torch.cat([face, face2], dim=-1)
            lip = lip.permute(0, 2, 1)  # 将维度从 [16, 156, 512] 变为 [16, 512, 156] 以适应 Conv1d
            lip = self.dim_reduction_conv(lip)
            lip = lip.permute(0, 2, 1)  # 将维度恢复为 [16, 156, 256]

            face = face.permute(0, 2, 1)
            face = self.dim_reduction_conv(face)
            face = face.permute(0, 2, 1)

            x_tmp = face + lip  # 这里可以根据需要考虑是否用 concat 替代加法

            # 融合交叉注意力
            x_tmp = self._mha_block(self.norm3(x_tmp), face_memory, memory_mask, memory_key_padding_mask)
            t = (lip_t + nonlip_t) / 2
            x_tmp = x_tmp + featurewise_affine(x_tmp, self.film3(t))
            x = self.linear3(x_tmp)
            # lip = lip + self.adain1(lip1, lip_t)
            # face = face + self.adain1(face1, nonlip_t)
            # face2 = self._sa_block(self.norm2(face), tgt_mask, tgt_key_padding_mask)
            # lip2 = self._mha_block(self.norm2(lip), memory, memory_mask, memory_key_padding_mask)
            # lip = lip + self.adain2(lip2, lip_t)
            # face = face + self.adain2(face2, nonlip_t)
            # x_tmp = face + lip
            # # fusion  cross attention
            # x_tmp = self._mha_block(self.norm3(x_tmp), face_memory, memory_mask, memory_key_padding_mask)
            # t = (lip_t + nonlip_t) / 2
            # x_tmp = x_tmp + featurewise_affine(x_tmp, self.film3(t))
            # x = self.linear3(x_tmp)
        else:
            x = self.norm1(x + featurewise_affine(self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)))
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))

        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def feature_processing(self, feature, affine_params):

        feature = featurewise_affine(feature, affine_params)
        feature = self.relu(feature)
        feature = self.norm(feature)
        feature = self.dropout(feature)
        return feature


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, lip_t, nonlip_t, face_cond):
        for layer in self.stack:
            x = layer(x, cond, lip_t, nonlip_t, face_cond)
        return x


class MotionDecoder(nn.Module):
    def __init__(
            self,
            nfeats: int,
            seq_len: int = 150,  # 5 seconds, 30 fps
            latent_dim: int = 256,
            ff_size: int = 1024,
            num_layers: int = 4,
            num_heads: int = 4,
            dropout: float = 0.1,
            cond_feature_dim: int = 1024,
            activation: Callable[[Tensor], Tensor] = F.gelu,
            use_rotary=True,
            **kwargs
    ) -> None:
        super().__init__()

        output_feats = nfeats
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(latent_dim, dropout, batch_first=True)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim),
        )

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        self.face_mlp = nn.Sequential(
            nn.Linear(96, latent_dim * 4),
            nn.Mish(),
        )
        self.to_face_cond = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim),
        )
        self.to_face_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len * 2, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.null_emo_cond_embed = nn.Parameter(torch.randn(1, seq_len * 2, latent_dim))
        self.null_emo_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.face_null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)
        self.norm_emo_cond = nn.LayerNorm(latent_dim)
        self.input_projection = nn.Linear(nfeats, latent_dim)  # 수정완료

        self.input_projection_lip = nn.Linear(37 * 3, latent_dim)
        self.relu = nn.ReLU()
        self.input_projection_wo_lip = nn.Linear(31 * 3, latent_dim)
        self.cond_encoder = []
        self.face_encoder = []
        self.pos_encoder = []
        self.emo_cond_encoder = []
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
            self.emo_cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
            self.face_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
            self.pos_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        self.cond_encoder = nn.Sequential(*self.cond_encoder)
        self.emo_cond_encoder = nn.Sequential(*self.emo_cond_encoder)
        self.face_encoder = nn.Sequential(*self.face_encoder)
        self.pos_encoder = nn.Sequential(*self.pos_encoder)
        # conditional projection
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.emo_cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.face_projection = nn.Linear(204, latent_dim)
        self.pos_projection = nn.Linear(198, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.non_attn_emo_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.non_attn_face_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.non_attn_pos_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)

        self.final_layer = nn.Linear(latent_dim * 2, output_feats)

    def guided_forward(self, x_pos, x, face, cond_embed, times, guidance_weight):
        unc = self.forward(x_pos, x, face, cond_embed, times, cond_drop_prob=1)
        conditioned = self.forward(x_pos, x, face, cond_embed, times, cond_drop_prob=0)

        return unc + (conditioned - unc) * guidance_weight

    def forward(
            self,
            x_pos: Tensor,
            x: Tensor,
            face: Tensor,
            cond_embed: Tensor,
            times: Tensor,
            cond_drop_prob: float = 0.0,
    ):
        batch_size, frames, device = x.shape[0], x.shape[1], x.device
        temp = x[:, :, :]
        temp = temp.view(batch_size, frames, -1, 3)
        upper_face = temp[:, :, 17:48, :]
        lower_face = temp[:, :, :17, :]
        lip = temp[:, :, 48:, :]

        upper_face = upper_face.view(batch_size, frames, -1)
        lower_face = lower_face.view(batch_size, frames, -1)
        lip = lip.view(batch_size, frames, -1)
        lower_face_w_lip = torch.cat((lower_face, lip), axis=-1)

        lip = self.input_projection_lip(lower_face_w_lip)
        lip = self.relu(lip)
        lip = self.abs_pos_encoding(lip)
        upper_face = self.input_projection_wo_lip(upper_face)
        upper_face = self.relu(upper_face)
        upper_face = self.abs_pos_encoding(upper_face)

        x = torch.cat((lip, upper_face), axis=-1)  # [B, 1024, 204]

        # create music conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        # hubert
        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.relu(cond_tokens)
        emo_cond_tokens = self.emo_cond_projection(cond_embed)
        emo_cond_tokens = self.relu(emo_cond_tokens)
        # encode tokens
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        emo_cond_tokens = self.abs_pos_encoding(emo_cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)
        emo_cond_tokens = self.emo_cond_encoder(emo_cond_tokens)
        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        null_emo_cond_embed = self.null_cond_embed.to(emo_cond_tokens.dtype)
        # print("keep_mask_embed size:", keep_mask_embed.size())
        # print("cond_tokens size:", cond_tokens.size())
        # print("null_cond_embed size:", null_cond_embed.size())
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)
        emo_cond_tokens = torch.where(keep_mask_embed, emo_cond_tokens, null_emo_cond_embed)

        mean_pooled_emo_cond_tokens = emo_cond_tokens.mean(dim=-2)
        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)
        emo_cond_hidden = self.non_attn_emo_cond_projection(mean_pooled_emo_cond_tokens)

        # create the diffusion timestep embedding, add the extra music projection
        t_hidden = self.time_mlp(times)

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)
        lip_t = t
        nonlip_t = t

        face_tokens = self.face_projection(face)  # [B, 150, 2048] #init_ldmk
        face_tokens = self.abs_pos_encoding(face_tokens)
        face_tokens = self.face_encoder(face_tokens)

        face_null_cond_embed = self.face_null_cond_embed.to(cond_tokens.dtype)
        face_tokens = torch.where(keep_mask_embed, face_tokens, face_null_cond_embed)

        mean_pooled_face_tokens = face_tokens.mean(dim=-2)
        face_hidden = self.non_attn_face_projection(mean_pooled_face_tokens)

        lip_t += face_hidden
        nonlip_t += face_hidden

        # ---------------------------------------------------- #
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        lip_t += cond_hidden
        c = torch.cat((cond_tokens, t_tokens, face_tokens), dim=-2)
        memory = self.norm_cond(c)  # 音频条件

        null_emo_cond_hidden = self.null_emo_cond_hidden.to(t.dtype)
        emo_cond_hidden = torch.where(keep_mask_hidden, emo_cond_hidden, null_emo_cond_hidden)
        lip_t += emo_cond_hidden
        c_emo = torch.cat((emo_cond_tokens, t_tokens, face_tokens), dim=-2)
        memory_emo = self.norm_emo_cond(c_emo)  # 情绪条件

        merged_memory = (memory + memory_emo) / 2
        # -------------------------------------------------------- #
        face_wo_lip_c = torch.cat((t_tokens, face_tokens), dim=-2)  # 오디오 cond제외
        face_memory = self.norm_cond(face_wo_lip_c)

        output = self.seqTransDecoder(x, merged_memory, lip_t, nonlip_t, face_memory)  # , pos_memory=None)
        output = self.final_layer(output)
        return output


if __name__ == "__main__":
    # Sample inputs
    d_model = 256
    batch_size = 16
    seq_length_tgt = 156
    seq_length_mem = 470
    seq_length_face_mem = 158

    lip_t = torch.randn(batch_size, 256)
    nonlip_t = torch.randn(batch_size, 256)
    tgt = torch.randn(batch_size, seq_length_tgt, 512)
    memory = torch.randn(batch_size, seq_length_mem, 256)
    face_memory = torch.randn(batch_size, seq_length_face_mem, 256)

    # Initialize and run the layer
    layer = FiLMTransformerDecoderLayer(
        d_model=256,
        nhead=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation=F.gelu,
        batch_first=True,
        rotary=None,
    )

    output = layer(tgt, memory, lip_t, nonlip_t, face_memory)
    print(output.shape)