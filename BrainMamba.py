import math
from typing import Sequence

import mmengine
import torch
import torch.nn as nn

from mmengine.model import ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.models import build_2d_sincos_position_embedding

from transformers.models.mamba.modeling_mamba import MambaMixer

from mmpretrain.registry import MODELS
from mmpretrain.models.utils import build_norm_layer
from mmpretrain.models.backbones.base_backbone import BaseBackbone



class PatchEmbed_(nn.Module):
    def __init__(self,channel,kernel_size,embed_dim,padding_size):
        super(PatchEmbed_, self).__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels = channel,out_channels= embed_dim, kernel_size= kernel_size,stride = kernel_size,padding = padding_size)
    def forward(self, x):
        x = self.conv(x) # N, embedsize, slice_num
        x = x.transpose(1,2)
        return x

#

@MODELS.register_module()
class BrainMamba(BaseBackbone):


    def __init__(self,
                 patch_channel_timestep=[4,4,24],
                 padding = 0,
                 arch=[128, 2],
                 pe_type='learnable',
                 path_type='forward_reverse_shuffle_gate', # 'forward', 'forward_reverse_mean', 'forward_reverse_gate', 'forward_reverse_shuffle_gate'
                 cls_position='none',  # 'head', 'tail', 'head_tail', 'middle', 'none'
                 out_indices=-1,
                 drop_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 interpolate_mode='bicubic',
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(embednet, self).__init__(init_cfg)

        self.embed_dims = arch[0]
        self.num_layers = arch[1]
        self.channel =patch_channel_timestep[1]
        self.slicedtimestep = patch_channel_timestep[2] # kernel size
        self.cls_position = cls_position # 
        self.path_type = path_type
        self.padding_size = padding
        self.patch_embed = PatchEmbed_(self.channel, self.slicedtimestep,self.embed_dims,self.padding_size)
        self.num_patches =patch_channel_timestep[0]
        self.num_extra_tokens = 0

        if cls_position != 'none':
            if cls_position == 'head_tail':
                self.cls_token = nn.Parameter(torch.zeros(1, 2, self.embed_dims))
                self.num_extra_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
                self.num_extra_tokens = 1
        else:
            self.cls_token = None # None

        self.interpolate_mode = interpolate_mode
        self.pe_type = pe_type
        if pe_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_extra_tokens, self.embed_dims))
        elif pe_type == 'sine':
            self.pos_embed = build_2d_sincos_position_embedding(
                patches_resolution=self.patch_resolution,
                embed_dims=self.embed_dims,
                temperature=10000,
                cls_token=False)
        else:
            self.pos_embed = None

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices
        self.layers = ModuleList()
        self.gate_layers = ModuleList()

        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                hidden_size=self.embed_dims,
                state_size=16,
                intermediate_size=self.embed_dims * 2,
                conv_kernel=4,
                time_step_rank=math.ceil(self.embed_dims / 16),
                use_conv_bias=True,
                hidden_act="silu",
                use_bias=False,
            )

            _layer_cfg.update(layer_cfgs[i])
            _layer_cfg = mmengine.Config(_layer_cfg)
            self.layers.append(MambaMixer(_layer_cfg, i))
            
            if 'gate' in self.path_type:
                gate_out_dim = 2 
                if 'shuffle' in self.path_type:
                    gate_out_dim = 3
                self.gate_layers.append(
                    nn.Sequential(
                        nn.Linear(gate_out_dim*self.embed_dims, gate_out_dim, bias=False),
                        nn.Softmax(dim=-1)
                    )
                )


        self.pre_norm = build_norm_layer(norm_cfg, self.embed_dims)
        self.final_norm = final_norm
        if final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(embednet, self).init_weights()

        if not (isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)


    def forward(self, x):
        B = x.shape[0]
        patch_resolution = None
        x = self.patch_embed(x)

        if self.pos_embed is not None:

            x = x + self.pos_embed.to(device=x.device)
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            residual = x
            if 'forward' == self.path_type:
                x = self.pre_norm(x.to(dtype=self.pre_norm.weight.dtype))
                x = layer(x)

            if 'forward_reverse_mean' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x = torch.split(x_inputs, B, dim=0)
                x = (forward_x + torch.flip(reverse_x, [1])) / 2

            if 'forward_reverse_gate' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x = torch.split(x_inputs, B, dim=0)
                reverse_x = torch.flip(reverse_x, [1])
                mean_forward_x = torch.mean(forward_x, dim=1)
                mean_reverse_x = torch.mean(reverse_x, dim=1)
                gate = torch.cat([mean_forward_x, mean_reverse_x], dim=-1)
                gate = self.gate_layers[i](gate)
                gate = gate.unsqueeze(-1)
                x = gate[:, 0:1] * forward_x + gate[:, 1:2] * reverse_x
            if 'forward_reverse_shuffle_gate' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                rand_index = torch.randperm(x.size(1))
                x_inputs.append(x[:, rand_index])
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x, shuffle_x = torch.split(x_inputs, B, dim=0)
                reverse_x = torch.flip(reverse_x, [1])
                # reverse the random index
                rand_index = torch.argsort(rand_index)
                shuffle_x = shuffle_x[:, rand_index]
                mean_forward_x = torch.mean(forward_x, dim=1)
                mean_reverse_x = torch.mean(reverse_x, dim=1)
                mean_shuffle_x = torch.mean(shuffle_x, dim=1)
                gate = torch.cat([mean_forward_x, mean_reverse_x, mean_shuffle_x], dim=-1)
                gate = self.gate_layers[i](gate)
                gate = gate.unsqueeze(-1)
                x = gate[:, 0:1] * forward_x + gate[:, 1:2] * reverse_x + gate[:, 2:3] * shuffle_x

            if 'forward_reverse_shuffle_mean' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                rand_index = torch.randperm(x.size(1))
                x_inputs.append(x[:, rand_index])
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x, shuffle_x = torch.split(x_inputs, B, dim=0)
                reverse_x = torch.flip(reverse_x, [1])
                # reverse the random index
                rand_index = torch.argsort(rand_index)
                shuffle_x = shuffle_x[:, rand_index]
                x = (forward_x + reverse_x + shuffle_x) / 3
            x = residual + x
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)


