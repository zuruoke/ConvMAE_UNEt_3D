from convmae.components import Block3D, CBlock3D, PatchEmbed3D
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class ConvMAE(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=[96, 24, 12], patch_size=[4,2,2], in_chans=1, num_classes=80, embed_dim=[256,384,768], depth=[2,2,11],
                 num_heads=12, mlp_ratio=[4,4,4], qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 init_values=1, use_checkpoint=False,
                 use_abs_pos_emb=True, use_rel_pos_bias=True, use_shared_rel_pos_bias=False,
                 out_indices=[3, 5, 7, 11], fpn1_norm='SyncBN'):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed1 = PatchEmbed3D(
        img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])

        self.patch_embed2 = PatchEmbed3D(
                img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed3D(
                img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])

        num_patches = self.patch_embed3.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[2]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.use_checkpoint = use_checkpoint
        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks1 = nn.ModuleList([
            Block3D(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0],  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock3D(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1],  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block3D(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2],  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[2])])


        #self.fpn = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fpn = nn.Sequential(
          nn.Conv3d(embed_dim[-1], embed_dim[-1], kernel_size=3, stride=2),
          nn.SyncBatchNorm(embed_dim[-1]),
          nn.GELU(),
        )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks3):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)

            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        features = []
        B = x.shape[0]
        x, _ = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        features.append(x)
        x, _ = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        features.append(x)

        x, (Hp, Wp, Dp) = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None


        for blk in self.blocks3:
            # if self.use_checkpoint:
            #     x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            # else:
            x = blk(x)
        # x = self.norm(x)

        x = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp, Dp)
        features.append(x.contiguous())
        x = self.fpn(x)
        features.append(x)
        return features

    def forward(self, x):
        x = self.forward_features(x)
        return x
