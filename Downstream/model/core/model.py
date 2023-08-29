from __future__ import annotations

import itertools
from collections.abc import Sequence
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from einops import rearrange

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "SwinUNETR",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]

class Model(nn.Module):
    def __init__(self, model_convmae ):
      super().__init__()
      self.convmae_model = model_convmae
      self.conv_transpose = get_conv_layer(
          spatial_dims=3,
          in_channels=256,
          out_channels=48,
          kernel_size=2,
          stride=2,
          is_transposed=True
      )
      self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=48,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )
      
      self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )
      
      self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )
      
      self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=768,
            out_channels=768,
            kernel_size=3,
            stride=1,
           norm_name='instance',
            res_block=True,
      )

      self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=768,
            out_channels=768,
            kernel_size=3,
            upsample_kernel_size=2,
           norm_name='instance',
            res_block=True,
        )
      
      self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=768,
            out_channels=384,
            kernel_size=3,
            upsample_kernel_size=2,
          norm_name='instance',
            res_block=True,
        )
      
      self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            upsample_kernel_size=2,
          norm_name='instance',
            res_block=True,
        )
      
      self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=48,
            out_channels=48,
            kernel_size=3,
            upsample_kernel_size=2,
          norm_name='instance',
            res_block=True,
        )

      self.seg_out = UnetOutBlock(spatial_dims=3, in_channels=48, out_channels=14)

    
    def forward(self, x_in):
        features = self.convmae_model(x_in)

        # UNET Encoder
        enc0 = self.encoder0(x_in)
        # print(f"enc0.shape: {enc0.shape}")
        enc1 = self.encoder1(features[0])
        # print(f"enc1.shape: {enc1.shape}")
        enc2 = self.encoder2(features[1])
        # print(f"enc2.shape: {enc2.shape}")
        enc3 = self.encoder3(features[3])
        # print(f"enc3.shape: {enc3.shape}")
        enc3 = F.interpolate(enc3, size=[3, 3, 3], mode='trilinear', align_corners=False)
        # print(f"enc3.shape: {enc3.shape}")
        

        # UNET Decoder
        dec3 = self.decoder5(enc3, features[2])
        # print(f"dec3.shape: {dec3.shape}")
        dec2 = self.decoder4(dec3, enc2)
        # print(f"dec2.shape: {dec2.shape}")
        dec1 = self.decoder3(dec2, enc1)
        # print(f"dec1.shape: {dec1.shape}")
        dec1 = self.conv_transpose(dec1)
        # print(f"dec1.shape: {dec1.shape}")
        dec0 = self.decoder1(dec1, enc0)
        # print(f"dec0.shape: {dec0.shape}")
        logits = self.seg_out(dec0)
        # print(f"logits .shape: {logits.shape}")
        return logits
