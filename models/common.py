from stringprep import b1_set
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import math
import numpy as np
from collections import OrderedDict
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class CustomTextEncoder(torch.nn.Module):
    def __init__(self, clip_model, tokenizer, dtype=torch.float16):
        super().__init__()
        self.dtype = dtype

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding

        self.tokenizer = tokenizer

    def tokenize(self, text):
        return torch.cat([self.tokenizer(tok) for tok in text])

    def encode_text(self, text, enable_pos_emb=True):
        token_ids = self.tokenize(text)
        text_features = self.forward(token_ids, None, enable_pos_emb)
        return text_features

    def forward(self, token_ids, token_tensors, enable_pos_emb):
        """The forward function to compute representations for the prompts.

        Args:
            token_ids (torch.tensor): the token ids, which
                contains the <eos> token.
            token_tensors (torch.Tensor, optional): the tensor
                embeddings for the token ids. Defaults to None.
            enable_pos_emb (bool, optional): adds the learned
                positional embeddigngs if true. Defaults to False.

        Returns:
            torch.Tensor: the vector representation of the prompt.
        """
        if token_tensors is not None:
            text_features = token_tensors
        else:
            text_features = self.token_embedding(token_ids)

        if len(text_features.shape) == 3:
            text_features = text_features.type(self.dtype)
            x = (
                text_features + self.positional_embedding.type(self.dtype)
                if enable_pos_emb
                else text_features
            )
            x = x.permute(1, 0, 2)
            text_feature = self.transformer(x)

            x = text_feature.permute(1, 0, 2)
            x = self.ln_final(x)
            tf = (
                x[
                    torch.arange(x.shape[0]), token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_projection
            )
        else:
            tfs, text_features_ = list(), list()
            text_features = text_features.type(self.dtype)
            x = (
                text_features + self.positional_embedding.type(self.dtype)
                if enable_pos_emb
                else text_features
            )
            for idx in range(x.shape[0]):
                t = x[idx].permute(1, 0, 2)
                text_feature = self.transformer(t)

                t = text_feature.permute(1, 0, 2)
                t = self.ln_final(t)
                tf = (
                        t[
                            torch.arange(t.shape[0]), token_ids.argmax(dim=-1)
                        ]  # POS of <EOS>
                        @ self.text_projection
                )
                tfs.append(tf)
                text_features_.append(text_feature)
            tf = torch.stack(tfs)
            text_feature = torch.stack(text_features_)

        return tf, text_feature


class MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''
    def __init__(self, inp_dim, out_dim, num_layers = 1, relu = True, bias = True, dropout = False, norm = False, layers = []):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers.pop(0)
            mod.append(nn.Linear(incoming, outgoing, bias = bias))
            
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace = True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p = 0.3))

        mod.append(nn.Linear(incoming, out_dim, bias = bias))

        if relu:
            mod.append(nn.ReLU(inplace = True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mod = nn.Sequential(*mod)
    
    def forward(self, x):
        return self.mod(x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("drop", nn.Dropout(0.3)),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_x = LayerNorm(d_model)
        self.ln_y = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, y, y, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x + self.attention(self.ln_x(x), self.ln_y(y))
        x = x + self.mlp(self.ln_2(x))
        return x



class FusionTextImageBlock(nn.Module):
    def __init__(self, 
                 width_img: int, 
                 width_txt: int, 
                 attributes: int, 
                 classes: int, 
                 layers: int, 
                 attn_mask: torch.Tensor = None,
                 context_length: int = 8, 
                 fusion: str = "BiFusion"):
        super().__init__()
        self.fusion = fusion
        self.width_img = width_img
        self.width_txt = width_txt
        self.layers = layers
        self.context_length = context_length
        self.attributes = attributes
        self.classes = classes
        self.img2txt_transform_layer1 = nn.Linear(width_img, width_txt)
        self.img2txt_transform_layer2 = nn.Linear(257, context_length * (attributes + classes))
        self.txt2img_transform_layer1 = nn.Linear(width_txt, width_img)
        self.txt2img_transform_layer2 = nn.Linear(context_length * (attributes + classes), 257)
        self.dropout = nn.Dropout(0.3)
        self.crossblock_img = CrossResidualAttentionBlock(width_img, width_img//64, attn_mask)
        self.crossblock_txt = CrossResidualAttentionBlock(width_txt, width_txt//64, attn_mask)
        self.resblocks_img = nn.Sequential(*[ResidualAttentionBlock(width_img, width_img//64, attn_mask) for _ in range(layers)])
        self.resblocks_txt = nn.Sequential(*[ResidualAttentionBlock(width_txt, width_txt//64, attn_mask) for _ in range(layers)])
        self.txt_fine_tune = nn.Linear(self.width_txt, self.width_txt)


    def decompose(self, text_feature, idx):
        t, l, c = text_feature.shape
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        text_att = torch.zeros(t, self.attributes, c).cuda()
        text_obj = torch.zeros(t, self.classes, c).cuda()
        for i in range(self.attributes):
            text_att[:, i, :] = text_feature[:, np.where(att_idx==i)[0], :].mean(-2)
        for i in range(self.classes):
            text_obj[:, i, :] = text_feature[:, np.where(obj_idx==i)[0], :].mean(-2)    
        text_decom_feature = torch.cat([text_att, text_obj], dim=1)
        return text_decom_feature


    def compose(self, text_feature, idx):
        t, l, c = text_feature.shape
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        text_com_feature = torch.zeros(t, len(idx), c).cuda()
        text_com_feature = text_feature[:, att_idx, :] * text_feature[:, obj_idx + self.attributes, :]
        text_com_feature = self.txt_fine_tune(text_com_feature)
        return text_com_feature



    def img2txt(self, x: torch.Tensor):
        x = self.img2txt_transform_layer1(x)
        x = x.permute(2,1,0)
        x = self.img2txt_transform_layer2(x)
        x = x.permute(2,1,0).reshape(-1, (self.attributes + self.classes), self.width_txt)
        x = self.dropout(x)
        return x

    def txt2img(self, x:torch.Tensor, idx, b: int):    
        x = self.decompose(x, idx)
        x = self.txt2img_transform_layer1(x)
        x = rearrange(x, 't l c -> c (t l)')
        x = self.txt2img_transform_layer2(x)
        x = self.dropout(x)
        x = x.permute(1,0).unsqueeze(1).repeat(1,b,1)
        return x
        

    def forward(self, x_image: torch.Tensor, x_text: torch.Tensor, idx, b: int):
        if self.fusion == "BiFusion":
            x_img = self.crossblock_img(x_image, self.txt2img(x_text, idx, b))
            x_txt = self.img2txt(x_image)
            x_text = self.decompose(x_text, idx)
            x_txt = self.crossblock_txt(x_text.repeat(b, 1, 1), x_txt)
            x_txt = self.resblocks_txt(x_txt)
            x_txt = self.compose(x_txt, idx)
            x_txt = x_txt.reshape(b, self.context_length, -1, self.width_txt)
            x_img = self.resblocks_img(x_img)
            return x_img, x_txt
        elif self.fusion == "img2txt":
            x_txt = self.img2txt(x_image)
            x_text = self.decompose(x_text, idx)
            x_txt = self.crossblock_txt(x_text.repeat(b, 1, 1), x_txt)
            x_txt = self.resblocks_txt(x_txt)
            x_txt = self.compose(x_txt, idx)
            x_txt = x_txt.reshape(b, self.context_length, -1, self.width_txt)
            x_img = self.resblocks_img(x_image)
            return x_img, x_txt
        elif self.fusion == "txt2img":
            x_img = self.crossblock_img(x_image, self.txt2img(x_text, idx, b))
            x_img = self.resblocks_img(x_img)
            x_txt = self.resblocks_txt(x_text)
            return x_img, x_txt
        elif self.fusion == "OnlySPM":
            return x_image, x_text


class Adapter(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option

        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x
