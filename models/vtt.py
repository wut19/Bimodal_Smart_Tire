""" 
    Modified from VISUO-TACTILE TRANSFORMERS FOR MANIPULATION
    https://github.com/yich7045/Visuo-Tactile-Transformers-for-Manipulation
"""

import torch
import torch.nn as nn
import warnings
import math
from torch.nn import functional as F
from torch.distributions import Normal

class PatchEmbedding(nn.Module):
    def __init__(self, visual_size=128, visual_patch_size=16, tactile_size=128, tactile_patch_size=16, in_chan=3, embeded_dim=384):
        super().__init__()
        # visual
        self.visual_patches = int((visual_size/visual_patch_size)*(visual_size/visual_patch_size))
        self.visual_size = visual_size
        # tactile
        self.tactile_patches = int((tactile_size/tactile_patch_size)*(tactile_size/tactile_patch_size))
        self.tactile_size = tactile_patch_size

        self.embeded_dim = embeded_dim
        """ use different feature extractor for visual and tactile """
        # TODO: use different feature extractor for even smoky and dark environments
        self.visual_proj = nn.Conv2d(in_chan, embeded_dim, kernel_size=visual_patch_size, stride=visual_patch_size)
        self.tactile_proj = nn.Conv2d(in_chan, embeded_dim, kernel_size=tactile_patch_size, stride=tactile_patch_size)
        self.norm_v = nn.LayerNorm(embeded_dim)
        self.norm_t = nn.LayerNorm(embeded_dim)

    def forward(self, visual, tactile):
        """
        Visual input shape: (batch, types, in_Channels, H, W)
        Tactile input shape: (batch,  in_channels, H, W)
        Output shape: (batch, N , embedd)
        """
        # visual
        if visual.ndim == 5:
            B, T, C, H, W = visual.shape
            visual = visual.view(B * T, C, H, W)
            patched_visual_feats = self.visual_proj(visual).flatten(2).transpose(1, 2).reshape(B, -1, self.embeded_dim)
            patched_visual_feats = self.norm_v(patched_visual_feats.reshape(-1, self.embeded_dim)).reshape(B, -1, self.embeded_dim)
        else:
            patched_visual_feats = None

        # tactile
        if tactile.ndim == 4:
            B, C, H, W = tactile.shape
            patched_tactile_feats = self.tactile_proj(tactile).flatten(2).transpose(1, 2)
            patched_tactile_feats = self.norm_t(patched_tactile_feats.reshape(-1, self.embeded_dim)).reshape(B, -1, self.embeded_dim)
        else:
            patched_tactile_feats = None
        return patched_visual_feats, patched_tactile_feats

class MMPatchEmbedding(nn.Module):
    def __init__(self, visual_size=128, visual_patch_size=16, visual_embed_mapping={}, tactile_size=128, tactile_patch_size=16, tactile_embed_mapping={}, in_chan=3, embeded_dim=384):
        super().__init__()
        # visual
        self.visual_patches = int((visual_size/visual_patch_size)*(visual_size/visual_patch_size))
        self.visual_size = visual_size
        self.visual_embed_mapping = visual_embed_mapping
        self.visual_embed_type = len(list(visual_embed_mapping.keys()))
        # tactile
        self.tactile_patches = int((tactile_size/tactile_patch_size)*(tactile_size/tactile_patch_size))
        self.tactile_size = tactile_patch_size
        self.tactile_embed_mapping = tactile_embed_mapping
        self.tactile_embed_type = len(list(tactile_embed_mapping.keys()))

        self.embeded_dim = embeded_dim

        """ use different feature extractor for visual and tactile """
        self.visual_projs = nn.ModuleList([nn.Conv2d(in_chan, embeded_dim, kernel_size=visual_patch_size, stride=visual_patch_size) for i in range(self.visual_embed_type)])
        self.visual_norms = nn.ModuleList([nn.LayerNorm(embeded_dim) for i in range(self.visual_embed_type)])
        self.tactile_projs = nn.ModuleList([nn.Conv2d(in_chan, embeded_dim, kernel_size=tactile_patch_size, stride=tactile_patch_size) for i in range(self.tactile_embed_type)])
        self.tactile_norms = nn.ModuleList([nn.LayerNorm(embeded_dim) for i in range(self.tactile_embed_type)])

    def forward(self, visual, tactile):
        """
        Visual input shape: (batch, types, in_Channels, H, W)
        Tactile input shape: (batch, types, in_channels, H, W)
        Output shape: (batch, N , embedd)
        """
        # visual
        if visual.ndim == 5:
            patched_visual_feats = []
            for key, index in self.visual_embed_mapping.items():
                visual_seg = visual[:, index]
                B, T, C, H, W = visual_seg.shape
                visual_seg = visual_seg.view(B * T, C, H, W)
                patched_visual_feat = self.visual_projs[key](visual_seg).flatten(2).transpose(1, 2).reshape(B, -1, self.embeded_dim)
                patched_visual_feat = self.visual_norms[key](patched_visual_feat.reshape(-1, self.embeded_dim)).reshape(B, -1, self.embeded_dim)
                patched_visual_feats.append(patched_visual_feat)
            patched_visual_feats = torch.concat(patched_visual_feats, dim=1)
        else:
            patched_visual_feats = None

        # tactile
        if tactile.ndim == 5:
            patched_tactile_feats = []
            for key, index in self.tactile_embed_mapping.items():
                tactile_seg = tactile[:, index]
                B, T, C, H, W = tactile_seg.shape
                tactile_seg = tactile_seg.view(B * T, C, H, W)
                patched_tactile_feat = self.tactile_projs[key](tactile_seg).flatten(2).transpose(1, 2).reshape(B, -1, self.embeded_dim)
                patched_tactile_feat = self.tactile_norms[key](patched_tactile_feat.reshape(-1, self.embeded_dim)).reshape(B, -1, self.embeded_dim)
                patched_tactile_feats.append(patched_tactile_feat)
            patched_tactile_feats = torch.concat(patched_tactile_feats, dim=1)
        else:
            patched_tactile_feats = None
        
        return patched_visual_feats, patched_tactile_feats

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3 x B x num_heads x N x att_dim
        q, k, v = qkv[0], qkv[1], qkv[2] # B x num_heads x N x att_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # B x num_heads x N x att_dim1 -> B x N x num_heads x att_dim1 -> B x N x C
        attn = attn.view(B, -1, N, N)   # can be used for visualization
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.MLP = nn.Sequential(nn.Linear(in_features, hidden_features),
                            act_layer(),
                            nn.Linear(hidden_features, out_features))
    def forward(self, x):
        x = self.MLP(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, return_attention: bool = False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VTT(nn.Module):
    """
        NOTICE: this is a prior version of VTT, which is depreciated. Used MMVTT instead.
    """
    def __init__(self, visual_size=84, visual_patch_size=14, visual_type=3, tactile_size=84, tactile_patch_size=2, use_tactile=True, in_chans=3, num_classes=12, embed_dim=384, depth=6,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            visual_size=visual_size, visual_patch_size=visual_patch_size, tactile_size=tactile_size, tactile_patch_size=tactile_patch_size,
            in_chan=in_chans, embeded_dim=embed_dim
        )
        visual_patches = self.patch_embed.visual_patches
        tactile_pathes = self.patch_embed.tactile_patches

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, visual_patches*visual_type + tactile_pathes*use_tactile, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.compress_patches = nn.Sequential(nn.Linear(embed_dim, embed_dim//4),
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(embed_dim//4, embed_dim//12))

        self.compress_layer = nn.Sequential(nn.Linear((visual_patches*visual_type + tactile_pathes*use_tactile)*embed_dim//12, 640),
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(640, 288))

        self.classifier = nn.Sequential(nn.Linear(288, num_classes),
                                                 nn.Softmax(dim=-1))

        trunc_normal_(self.pos_embed, std=.02)

    def interpolate_pos_encoding(self, x, w: int, h: int):
        """
            x: B x N x embed_dim
        """
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        else:
            raise ValueError('Position Encoder does not match dimension')

    def prepare_tokens(self, visual, tactile):
        """
            visual: B x T x C x H x W
            tactile: B x C x H_ x W_
        """
        if visual.ndim == 5:
            _, _, _, w, h = visual.shape
        else:
            _,_, w, h = tactile.shape
        patched_visual, patched_tactile = self.patch_embed(visual, tactile) # B x patch_num x embed_dim
        if patched_visual is not None and patched_tactile is not None:
            x = torch.cat((patched_visual, patched_tactile),dim=1) # B x N x embed_dim
        elif patched_visual is not None and patched_tactile is None:
            x = patched_visual
        elif patched_visual is None and patched_tactile is not None:
            x = patched_tactile
        # introduce contact embedding & alignment embedding
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def forward(self, visual, tactile):
        x = self.prepare_tokens(visual, tactile)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        img_tactile = self.compress_patches(x)
        B, patches, dim = img_tactile.size()
        img_tactile = img_tactile.view(B, -1)
        img_tactile = self.compress_layer(img_tactile)
        pred = self.classifier(img_tactile)
        return img_tactile, pred
    
class MMVTT(nn.Module):
    def __init__(self, 
                 visual_size=128, 
                 visual_patch_size=16, 
                 visual_type=3, 
                 visual_embed_mapping={0: [0, 1, 2]}, 
                 tactile_size=128, 
                 tactile_patch_size=16, 
                 tactile_type=3, 
                 tactile_embed_mapping={0: [0], 1: [1], 2:[2]}, 
                 in_chans=3, 
                 num_classes=12, 
                 embed_dim=384, 
                 depth=6,
                 num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 **kwargs
    ):
        super().__init__()
        self.patch_embed = MMPatchEmbedding(
            visual_size=visual_size, visual_patch_size=visual_patch_size, tactile_size=tactile_size, visual_embed_mapping=visual_embed_mapping, 
            tactile_patch_size=tactile_patch_size, tactile_embed_mapping=tactile_embed_mapping,
            in_chan=in_chans, embeded_dim=embed_dim
        )
        visual_patches = self.patch_embed.visual_patches
        tactile_pathes = self.patch_embed.tactile_patches

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, visual_patches*visual_type + tactile_pathes*tactile_type, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.compress_patches = nn.Sequential(nn.Linear(embed_dim, embed_dim//4),
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(embed_dim//4, embed_dim//12))

        self.compress_layer = nn.Sequential(nn.Linear((visual_patches*visual_type + tactile_pathes*tactile_type)*embed_dim//12, 640),
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(640, 288))

        self.classifier = nn.Sequential(nn.Linear(288, num_classes),
                                                 nn.Softmax(dim=-1))

        trunc_normal_(self.pos_embed, std=.02)

    def interpolate_pos_encoding(self, x, w: int, h: int):
        """
            x: B x N x embed_dim
        """
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        else:
            raise ValueError('Position Encoder does not match dimension')

    def prepare_tokens(self, visual, tactile):
        """
            visual: B x T x C x H x W
            tactile: B x T x C x H_ x W_
        """
        if visual.ndim == 5:
            _, _, _, w, h = visual.shape
        else:
            _, _, _, w, h = tactile.shape
        
        patched_visual, patched_tactile = self.patch_embed(visual, tactile) # B x patch_num x embed_dim
        if patched_visual is not None and patched_tactile is not None:
            x = torch.cat((patched_visual, patched_tactile),dim=1) # B x N x embed_dim
        elif patched_visual is not None and patched_tactile is None:
            x = patched_visual
        elif patched_visual is None and patched_tactile is not None:
            x = patched_tactile
        # introduce contact embedding & alignment embedding
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def forward(self, visual, tactile):
        x = self.prepare_tokens(visual, tactile)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        img_tactile = self.compress_patches(x)
        B, patches, dim = img_tactile.size()
        img_tactile = img_tactile.view(B, -1)
        img_tactile = self.compress_layer(img_tactile)
        pred = self.classifier(img_tactile)
        return img_tactile, pred

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,)*(x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
    
if __name__ == "__main__":
    """ test """
    vtt = MMVTT()
    B = 16
    T = 3
    C = 3
    H=W=128
    a = torch.zeros((B,T,C,H,W))
    b = torch.ones((B,T,C,H,W))
    z, c = vtt(a,b)
    print(c.shape)
