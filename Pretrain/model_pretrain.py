import torch
import torch.nn as nn

from model_utils import PatchEmbed3D, Block3D, CBlock3D, get_3d_sincos_pos_embed_, get_3d_sincos_pos_embed


class MaskedAutoencoderConvViT3D(nn.Module):
    
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # ConvMAE encoder specifics
        self.patch_embed1 = PatchEmbed3D(
        img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])

        self.patch_embed2 = PatchEmbed3D(
                img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed3D(
                img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        # Note: add decoder stages for 3D convolution
        self.stage1_output_decode = nn.Conv3d(embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv3d(embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed3.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[2]), requires_grad=False)
        self.blocks1 = nn.ModuleList([
            CBlock3D(
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
        self.norm = norm_layer(embed_dim[-1])

        # ConvMAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block3D(decoder_embed_dim, decoder_num_heads, mlp_ratio[0], qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 4096 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], 6, cls_token=False)
        # print(pos_embed.shape)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed_(self.decoder_pos_embed.shape[-1], 6,)
        # print(decoder_pos_embed.shape)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def patchify(self, imgs):
        """
        imgs: (N, 3, D, H, W)
        x: (N, L, patch_size**3 * 3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0

        d = h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 1, d, p, h, p, w, p)
        x = torch.einsum('ncdhpHWP->ncdhpHWP', x)
        x = x.reshape(shape=(imgs.shape[0], d * h * w, p**3 * 1))
        return x



    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 *3)
        imgs: (N, 3, D, H, W)
        """
        p = self.patch_embed.patch_size[0]
        d = h = w = int(x.shape[1]**(1./3))
        assert d * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], d, h, w, p, p, p, 3))
        x = torch.einsum('nhdpwqc->nchdpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, d * p, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.patch_embed3.num_patches
        # print(f"Transfomer Patches: {L}")
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore




    def forward_encoder(self, x, mask_ratio):
        # embed patches
        ids_keep, mask, ids_restore = self.random_masking(x, mask_ratio)
        # print(f"ids_keep.shape: {ids_keep.shape}")
        # print(f"mask.shape: {mask.shape}")
        # print(f"ids_restore.shape: {ids_restore.shape}")
        mask_for_patch1 = mask.reshape(-1, 6, 6, 6).unsqueeze(-1).repeat(1, 1, 1, 1, 64).reshape(-1, 6, 6, 6, 4, 4, 4).permute(0, 1, 4, 2, 5, 3, 6).reshape(8, 24, 24, 24).unsqueeze(1)
        # print(f"mask_for_patch1.shape: {mask_for_patch1.shape}")
        mask_for_patch2 = mask.reshape(-1, 6, 6, 6).unsqueeze(-1).repeat(1, 1, 1, 1, 8).reshape(-1, 6, 6, 6, 2, 2, 2).permute(0, 1, 4, 2, 5, 3, 6).reshape(8, 12, 12, 12).unsqueeze(1)
        # print(f"mask_for_patch2.shape: {mask_for_patch2.shape}")
        x = self.patch_embed1(x)
        # print(f"patch_embed1.shape: {x.shape}")
        # print("Doing Cblock ops...")
        for blk in self.blocks1:
            x = blk(x, 1 - mask_for_patch1)
        # print("Finish Cblock ops...")
        # print(f"After Conv1 Block: {x.shape}")
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)
        # print("Done with stage1_embd ops...")
        x = self.patch_embed2(x)
        # print(f"patch_embed2.shape: {x.shape}")
        for blk in self.blocks2:
            x = blk(x, 1 - mask_for_patch2)
        # print(f"After Conv2 Block: {x.shape}")
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)
        # print("Done with stage2_embd ops...")
        # print("Starting Embed3 ops...")
        x = self.patch_embed3(x)
        # print(f"patch_embed3.shape: {x.shape}")
        # print("Done Embed3 ops...")
        x = x.flatten(2).permute(0, 2, 1)
        # print("Finish Flatten ops...")
        # print(f"After Flattening: {x.shape}")
        x = self.patch_embed4(x)
        # print("Finish Embed4 ops...")
        # print(f"patch_embed4: {x.shape}")
        # add pos embed w/o cls token
        x = x + self.pos_embed
        # print("Finish Concatenate ops...")
        # print(f"After Concatenate ops: {x.shape}")
        # print("Starting Gather ops...")
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        # print("Finish Gather ops...")
        # print(f"After Gathering ops: {x.shape}")
        # print("Starting stage_embed ops...")
        stage1_embed = torch.gather(stage1_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
        stage2_embed = torch.gather(stage2_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))
        # print("Starting stage_embed ops...")

        # print("Starting tf blocks ops...")
        # apply Transformer blocks
        # print(x.shape)
        for blk in self.blocks3:
            x = blk(x)
        # print("Finish tf blocks ops...")
        x = x + stage1_embed + stage2_embed
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        # print(f"decoder_stage_1: {x.shape}" )
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]  - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        # print(f"decoder_stage_2: {x.shape}" )
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # print(f"decoder_stage_3: {x.shape}" )

        # add pos embed
        x = x + self.decoder_pos_embed
        # print(f"decoder_concatenate: {x.shape}" )

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        # print(f"After appplying TF Block : {x.shape}" )
        x = self.decoder_norm(x)
        # print(f"After appplying decoder norm : {x.shape}" )
        # predictor projection
        x = self.decoder_pred(x)
        # print(f"After appplying decoder pred : {x.shape}" )

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        # temp_i = pred
        # temp_i = mask # [N, L, p*p*3]
        # print(f"imgs_shape: {imgs.shape}")
        # print(f"pred _shape: {pred.shape}")
        # print(f"mask_shape: {mask.shape}")
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
