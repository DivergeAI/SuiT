from functools import partial
from typing import Optional

import torch
import os
import gdown

import torch.nn as nn
from torch.nn.functional import interpolate, scaled_dot_product_attention
from torch.jit import Final

from timm.models.vision_transformer import VisionTransformer, _cfg, LayerScale
from timm.models import register_model
from timm.layers import Mlp, DropPath, get_act_layer, get_norm_layer, use_fused_attn

from einops import repeat, rearrange
from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_softmax, scatter_min
from torch_scatter.composite import scatter_std

__all__ = [
    'suit_tiny_224', 'suit_small_224', 'suit_base_224', 'suit_base_dino'
]

# Poisitional Encoding proposed in Vaswani et al., https://arxiv.org/abs/1706.03762
class PositionalEncoding(nn.Module):
    """Sine-cosine positional encoding, as proposed in "Attention Is All You Need". """
    def __init__(self, pos_dim: int, ch: int, denominator: float = 10000.0):
        super(PositionalEncoding, self).__init__()
        assert ch % (pos_dim * 2) == 0, 'dimension of positional encoding must be equal to dim * 2.'
        enc_dim = int(ch / 2)
        div_term = torch.exp(torch.arange(0., enc_dim, 2) * -(torch.log(denominator) / enc_dim))
        freqs = torch.zeros([pos_dim, enc_dim])
        for i in range(pos_dim):
            freqs[i, : enc_dim // 2] = div_term
            freqs[i, enc_dim // 2:] = div_term
        self.freqs = freqs

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos (torch.Tensor): Input tensor with coordinates to encode.
                                Shape can be (B, L, C), (B, H, W, C), etc., where C is pos_dim.
        Returns:
            torch.Tensor: Positional encodings of the same shape as input, but last dim is `ch`.
        """
        # pos: (B L C), (B H W C), (B H W T C)
        pos_enc = torch.matmul(pos.float(), self.freqs.to(pos.device))
        pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)
        return pos_enc


# Fourier Features as Positional Encoding proposed in Tancik et al., https://arxiv.org/abs/2006.10739
class FourierFeatures(nn.Module):
    """Fourier Features for positional encoding, as proposed in "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains". """
    def __init__(self, pos_dim: int, ch: int, sigma: float = 10.0, train: bool = False):
        super(FourierFeatures, self).__init__()
        assert ch % 2 == 0, 'number of channels must be divisible by 2.'
        enc_dim = int(ch / 2)
        B = torch.randn([pos_dim, enc_dim]) * sigma
        if train:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos (torch.Tensor): Input tensor with coordinates to encode.
                                Shape can be (B, L, C), (B, H, W, C), etc., where C is pos_dim.
        Returns:
            torch.Tensor: Fourier feature encodings of the same shape as input, but last dim is `ch`.
        """
        # pos: (B L C), (B H W C), (B H W T C)
        pos_enc = torch.matmul(pos.float(), self.B)
        pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)
        return pos_enc


# modified from Self-attention block of ViT: timm.models.vision_transformer.Block
class EmptyMaskingBlock(nn.Module):
    """
    Transformer Block with support for an empty token mask in self-attention.
    This is a standard Vision Transformer block where the self-attention layer
    is replaced by `EmptyMaskingAttention`.
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EmptyMaskingAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_attention: bool = False) -> torch.Tensor:
        if return_attention:
            x_, attn = self.attn(self.norm1(x), mask, return_attention=return_attention)
        else:
            x_ = self.attn(self.norm1(x), mask)
        x = x + self.drop_path1(self.ls1(x_))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if return_attention:
            return x, attn
        else:
            return x
    

# modified from original attention layer of transformers: timm.models.vision_transformer.Attention
class EmptyMaskingAttention(nn.Module):
    """
    Self-Attention mechanism that supports masking of "empty" tokens.
    Empty tokens (e.g., from superpixels with no pixels assigned after downsampling)
    can be masked out so they don't participate in attention.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_attention: bool = False) -> torch.Tensor:
        if return_attention:
            self.fused_attn = False

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:  # no masking
            x = scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # mask padded tokens (empty superpixel clusters)
            if mask is not None:
                # attn shape: [B, heads, N, N]
                # mask shape: [B, N, 1]
                mask = torch.where(mask, torch.tensor(0.0), torch.tensor(-float('inf')))
                attn = attn + mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


class SuperpixelVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        self.pe_type = kwargs.get('pe_type', 'ff')
        self.pe_injection = kwargs.get('pe_injection', 'concat')
        self.downsample = kwargs.get('downsample', 2)
        self.aggregate = kwargs.get('aggregate', ['max', 'avg'])
        self.base_dim = kwargs.get('base_dim', 96)
        # embed_dim is part of kwargs for VisionTransformer, so it's already set on self by super().__init__
        # self.embed_dim = kwargs.get('embed_dim', 384) # This line is effectively handled by super()
        self.use_proj = kwargs.get('use_proj', True)
        
        # New parameters for hierarchical superpixels
        self.hierarchical_superpixels = kwargs.get('hierarchical_superpixels', False)
        self.num_scales = kwargs.get('num_scales', 0)

        # filter keywords before passing to super().__init__
        suit_only_keywords = ['pe_type', 'pe_injection', 'downsample', 'aggregate', 'base_dim', 'use_proj', 'hierarchical_superpixels', 'num_scales']
        old_timm_keywords = ['pretrained_cfg', 'pretrained_cfg_overlay', 'cache_dir']
        keywords_to_filter = suit_only_keywords + old_timm_keywords
        
        # We need embed_dim for scale_embeddings, so pop it specifically if it's in kwargs for super
        # but keep its value.
        current_embed_dim = kwargs.get('embed_dim', 384) # Default if not passed

        for k in keywords_to_filter:
            if k in kwargs:
                kwargs.pop(k)
        super().__init__(*args, **kwargs) # embed_dim is passed to super if it was in original kwargs
        
        # If embed_dim was not in kwargs for super, VisionTransformer sets a default. Ensure our self.embed_dim is correct.
        # self.embed_dim is already an attribute from VisionTransformer
        # If it was explicitly passed in kwargs, super() used it. If not, super() used its own default.
        # We need to ensure self.embed_dim matches what was intended.
        # However, self.embed_dim IS correctly set by super() if 'embed_dim' is in kwargs.
        # If 'embed_dim' is NOT in kwargs, super() uses its default (e.g. 768 for ViT-B).
        # Let's ensure consistency:
        self.embed_dim = current_embed_dim # Ensure self.embed_dim is what we expect for scale embeddings

        self.img_size = kwargs.get('img_size', 224) # This img_size is from the original kwargs, not the filtered one
        self.make_coords(self.img_size, self.downsample)

        token_dim = self.base_dim * len(self.aggregate)
        if self.pe_injection == 'concat':
            token_dim = token_dim * 2

        if self.use_proj:
            assert self.embed_dim % len(self.aggregate) == 0, 'embed dim must be divisible by number of aggregation methods.'
            if self.pe_injection == 'concat':
                self.projection = nn.Conv2d(self.base_dim * 2, int(self.embed_dim / len(self.aggregate)), 1)
            else:
                self.projection = nn.Conv2d(self.base_dim, int(self.embed_dim / len(self.aggregate)), 1)

        self.get_feats = nn.Sequential(
            nn.Conv2d(3, self.base_dim, 7, self.downsample, padding=3, padding_mode='replicate'),
            nn.BatchNorm2d(self.base_dim),
            nn.GELU(),
        )
        if self.pe_type == 'ff':
            self.pe = FourierFeatures(2, self.base_dim, train=True)
        else:
            self.pe = PositionalEncoding(2, self.base_dim)

        # Initialize scale embeddings if hierarchical_superpixels is True
        if self.hierarchical_superpixels and self.num_scales > 0:
            self.scale_embeddings = nn.Parameter(torch.randn(self.num_scales, self.embed_dim))
        else:
            self.scale_embeddings = None

        # re-init transformer blocks capable of masking padded tokens (empty superpixel clusters)
        # Use self.embed_dim which is now consistently set
        norm_layer = get_norm_layer(kwargs.get('norm_layer', None)) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(kwargs.get('act_layer', None)) or nn.GELU
        mlp_layer = kwargs.get('mlp_layer', Mlp) # Mlp is already imported
        
        # Retrieve depth, num_heads, etc., from self (set by super().__init__) or use defaults matching ViT
        depth = self.depth if hasattr(self, 'depth') else kwargs.get('depth',12) # self.depth should exist from ViT
        num_heads = self.num_heads if hasattr(self, 'num_heads') else kwargs.get('num_heads', 6) # self.num_heads should exist
        mlp_ratio = self.mlp_ratio if hasattr(self, 'mlp_ratio') else kwargs.get('mlp_ratio', 4.0) # self.mlp_ratio should exist
        qkv_bias = self.qkv_bias if hasattr(self, 'qkv_bias') else kwargs.get('qkv_bias', True) # self.qkv_bias should exist
        qk_norm = kwargs.get('qk_norm', False) # Not typically a self attribute in ViT
        init_values = self.init_values if hasattr(self, 'init_values') else kwargs.get('init_values', None) # self.init_values should exist
        proj_drop_rate = self.proj_drop.p if hasattr(self, 'proj_drop') else kwargs.get('proj_drop_rate', 0.) # self.proj_drop should exist
        attn_drop_rate = self.attn_drop.p if hasattr(self, 'attn_drop') else kwargs.get('attn_drop_rate', 0.) # self.attn_drop should exist
        drop_path_rate = self.drop_path_rate if hasattr(self, 'drop_path_rate') else kwargs.get('drop_path_rate', 0.) # self.drop_path_rate should exist

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            EmptyMaskingBlock(
                dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        
        original_weight_init = kwargs.get('weight_init', '') # From original kwargs
        if original_weight_init != 'skip':
            self.init_weights(original_weight_init)
        if kwargs.get('fix_init', False): # From original kwargs
            self.fix_init_weight()

        # remove redundant components from original ViT
        del self.patch_embed, self.pos_embed

    def tokenization(self, x: torch.Tensor, spix_label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates features from `x` based on superpixel labels `spix_label`.

        This method uses various scatter operations (mean, max, etc.) defined in
        `self.aggregate` to pool features within each superpixel segment.
        The results of these aggregations are concatenated to form the final
        superpixel tokens. It also computes a mask for empty superpixels.

        Args:
            x (torch.Tensor): Pixel features, typically of shape
                              [B, C_feat, H_feat, W_feat].
                              C_feat is `self.embed_dim // len(self.aggregate)` if projection is used,
                              or `self.base_dim * (2 if pe_concat else 1)` otherwise.
            spix_label (torch.Tensor): Superpixel label map of shape [B, 1, H_feat, W_feat],
                                       resized to match the feature map dimensions.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - tokens (torch.Tensor): Superpixel tokens of shape [B, N_tokens, embed_dim].
                - mask (torch.Tensor): Boolean mask for non-empty tokens, shape [B, 1, 1, N_tokens].
        """
        b, c, h, w = x.shape
        spix_label = spix_label.long()
        
        # NOTE: Superpixel label starts from 0.
        n_tokens = int(spix_label.max()) + 1 # spix_label: [b,1,h,w] 
        spix_label = spix_label.view(b, -1)  # Flatten the labels to shape (b, h * w)
        x = x.view(b, c, -1)  # Flatten the features to shape (b, c, h * w)
        # Expand labels to shape (b, c, h * w) to match features' shape
        labels_expanded = spix_label.unsqueeze(1).expand(-1, c, -1) # Shape: (b, c, h * w)
        # Flatten the batch and channel dimensions to shape (b * c, h * w)
        labels_expanded = labels_expanded.reshape(-1, h * w).long()
        x_flat  = x.reshape(-1, h * w)
        x_flat = x_flat.float()

        out_list = []
        # NOTE: MAX
        if 'max' in self.aggregate:
            # Perform scatter_max
            max_out, _ = scatter_max(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            max_out = max_out.view(b, c, n_tokens).permute(0, 2, 1)  # Shape: (b, n_tokens, c)
            out_list.append(max_out)

        # NOTE: MIN
        if 'min' in self.aggregate:
            # Perform scatter_min
            min_out, _ = scatter_min(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            min_out = min_out.view(b, c, n_tokens).permute(0, 2, 1)  # Shape: (b, n_tokens, c)
            out_list.append(min_out)
            
        # NOTE: AVG
        if 'avg' in self.aggregate:
            # Perform scatter_mean
            mean_out = scatter_mean(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            mean_out = mean_out.view(b, c, n_tokens).permute(0, 2, 1)  # Shape: (b, n_tokens, c)
            out_list.append(mean_out)

        # NOTE: STD
        if 'std' in self.aggregate:
            # Perform scatter_max
            std_out = scatter_std(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            std_out = std_out.view(b, c, n_tokens).permute(0, 2, 1)  # Shape: (b, n_tokens, c)
            out_list.append(std_out)

        # NOTE: SOFTMAX
        if 'softmax' in self.aggregate:
            # Perform scatter_softmax
            softmax_weights = scatter_softmax(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            # Weighted features
            weighted_softmax_features = x_flat * softmax_weights
            # Perform scatter_sum to aggregate the weighted features
            softmax_out = scatter_sum(weighted_softmax_features, labels_expanded, dim=1, dim_size=n_tokens)
            softmax_out = softmax_out.view(b, c, n_tokens).permute(0, 2, 1)  # Shape: (b, n_tokens, c)
            out_list.append(softmax_out)
        
        # NOTE: mask padded tokens (empty superpixel clusters)
        ones = torch.ones(b, 1, h, w, device=x.device, requires_grad=False).reshape(-1, h * w).int()
        spix_labels_flat = spix_label.reshape(-1, h * w).long()
        cluster_counter = scatter_sum(ones, spix_labels_flat, dim=1, dim_size=n_tokens).view(b, 1, n_tokens).permute(0, 2, 1)
        mask = (cluster_counter != 0)  # mask empty tokens (B, N, 1)
        mask = rearrange(mask, 'b n 1 -> b 1 1 n')

        # Concatenate all selected pooled features
        tokens = torch.cat(out_list, dim=2)  # Concatenate along the channel dimension

        return tokens, mask

    def prepare_tokens(self, x: torch.Tensor, spix_label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares superpixel tokens from raw input images and superpixel labels.

        This involves:
        1. Extracting initial pixel features using `self.get_feats`.
        2. Adding positional encodings (`self.pe`).
        3. Optionally projecting features (`self.projection`).
        4. Handling single or hierarchical superpixel labels:
            - Resizing labels to match feature map dimensions.
            - Calling `self.tokenization` for each scale (if hierarchical).
            - Adding learnable scale embeddings (if hierarchical).
            - Concatenating tokens from different scales.
        5. Prepending a CLS token and its corresponding mask.
        6. Applying dropout (`self.pos_drop`).

        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H_img, W_img].
            spix_label (torch.Tensor | list[torch.Tensor]): Superpixel label map(s).
                - If single scale: A tensor of shape [B, 1, H_spix, W_spix].
                - If hierarchical: A list of tensors, each of shape [B, 1, H_spix, W_spix].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - final_tokens (torch.Tensor): The sequence of superpixel tokens (plus CLS token)
                                               ready for the transformer blocks,
                                               shape [B, N_total_tokens + 1, embed_dim].
                - final_mask (torch.Tensor): Boolean mask for non-empty tokens (including CLS),
                                             shape [B, 1, 1, N_total_tokens + 1].
        """
        # 1. Initial feature processing (common for all scales)
        processed_x = self.get_feats(x)  # [B, base_dim, H_feat, W_feat]
        b = processed_x.shape[0]

        # Prepare Positional Encoding
        pe = self.pe(self.coords).permute(0, 3, 2, 1)  # [1, base_dim, H_feat, W_feat]
        pe = repeat(pe, '1 c h w -> b c h w', b=b)

        # Apply PE to processed features
        x_with_pe = torch.cat([processed_x, pe], dim=1) if self.pe_injection == 'concat' else processed_x + pe

        # Apply projection if used
        # self.projection output channels: self.embed_dim // len(self.aggregate)
        # self.tokenization will concatenate aggregated features to match self.embed_dim
        pixel_features_for_tokenization = self.projection(x_with_pe) if self.use_proj else x_with_pe
        
        tokens: torch.Tensor
        mask: torch.Tensor

        if self.hierarchical_superpixels:
            if not isinstance(spix_label, list) or len(spix_label) != self.num_scales:
                raise ValueError(f"In hierarchical mode, spix_label must be a list of length num_scales ({self.num_scales}). Got: {type(spix_label)}")
            if self.scale_embeddings is None:
                raise RuntimeError("Hierarchical superpixels are enabled but scale_embeddings are not initialized.")

            all_tokens_list = []
            all_masks_list = []

            for i in range(self.num_scales):
                current_spix_label = spix_label[i]
                # Resize superpixel label to match the spatial dimensions of pixel_features
                current_spix_label_resized = interpolate(
                    current_spix_label.to(torch.float), 
                    size=pixel_features_for_tokenization.shape[-2:], 
                    mode='nearest'
                )
                
                # Tokenize features for the current scale
                tokens_scale_i, mask_scale_i = self.tokenization(pixel_features_for_tokenization, current_spix_label_resized)
                
                # Add scale embedding
                # self.scale_embeddings[i] has shape [embed_dim]
                # tokens_scale_i has shape [B, N_tokens_i, embed_dim]
                # Broadcasting will add self.scale_embeddings[i] to each token in tokens_scale_i
                tokens_scale_i = tokens_scale_i + self.scale_embeddings[i] 
                
                all_tokens_list.append(tokens_scale_i)
                all_masks_list.append(mask_scale_i) # mask_scale_i is [B, 1, 1, N_tokens_i]
            
            tokens = torch.cat(all_tokens_list, dim=1) # Concatenate along token sequence dimension
            mask = torch.cat(all_masks_list, dim=-1)   # Concatenate along the last dimension of masks
        
        else: # Single scale tokenization
            if isinstance(spix_label, list):
                 raise ValueError(f"Not in hierarchical mode, but spix_label is a list. Expected a single tensor.")
            spix_label_resized = interpolate(
                spix_label.to(torch.float), 
                size=pixel_features_for_tokenization.shape[-2:], 
                mode='nearest'
            )
            tokens, mask = self.tokenization(pixel_features_for_tokenization, spix_label_resized)

        # Prepend CLS token
        cls_tokens = repeat(self.cls_token, '1 1 c -> b 1 c', b=b) # cls_tokens: [B, 1, embed_dim] 
        final_tokens = torch.cat((cls_tokens, tokens), dim=1)

        # Prepend CLS mask
        cls_mask = torch.ones(b, 1, 1, 1, dtype=torch.bool, device=final_tokens.device, requires_grad=False)
        final_mask = torch.cat((cls_mask, mask), dim=-1)
        
        final_tokens = self.pos_drop(final_tokens) # Apply positional dropout
        return final_tokens, final_mask

    def forward_features(self, x: torch.Tensor, spix_label: torch.Tensor) -> torch.Tensor:
        """
        Processes input image and superpixel labels into a sequence of token features.
        This method prepares tokens using `prepare_tokens` and then passes them
        through the transformer blocks.
        """
        x, mask = self.prepare_tokens(x, spix_label) # Generate superpixel tokens
        x = self.patch_drop(x) # This is a no-op if patch_dropout is 0, but kept for ViT compatibility
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x) # Final normalization after transformer blocks
        return x

    def forward(self, x: torch.Tensor, spix_label: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass of the model, from input image to classification head.
        """
        x = self.forward_features(x, spix_label)
        x = self.forward_head(x)
        return x
    
    def make_coords(self, img_size: Optional[tuple[int, int] | int] = None, downsample: Optional[int] = None):
        """
        Generates normalized 2D coordinates for positional encoding.
        The coordinates are scaled to [0, 1].
        Updates `self.coords`.
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if img_size is None:
            img_size = self.img_size
        if downsample is None:
            downsample = self.downsample

        if isinstance(img_size, tuple):
            # Coordinates for Positional Encoding
            x_coord, y_coord = torch.arange(0, int(img_size[1]), device=device), torch.arange(0, int(img_size[0]), device=device) # x_coord: [w], y_coord: [h]
            x_coord, y_coord = x_coord / (img_size[1]-1), y_coord / (img_size[0]-1)
            self.coords = torch.cartesian_prod(x_coord, y_coord).reshape(1, img_size[1], img_size[0], 2) # [w*h,2] -> [1,w,h,2]
        else:
            img_size = int(img_size / downsample)
            # Coordinates for Positional Encoding
            x_coord, y_coord = torch.arange(0, img_size, device=device), torch.arange(0, img_size, device=device) # x_coord: [w], y_coord: [h]
            x_coord, y_coord = x_coord / (img_size-1), y_coord / (img_size-1)
            self.coords = torch.cartesian_prod(x_coord, y_coord).reshape(1, img_size, img_size, 2) # [w*h,2] -> [1,w,h,2]

    def reset_stride(self, new_stride, reset_coords=True):
        self.get_feats[0].stride = new_stride
        self.downsample = new_stride
        if reset_coords:
            self.make_coords(downsample=new_stride)

    def reset_img_size(self, img_size: tuple[int, int] | int, reset_coords: bool = True):
        """Resets the internal image size and optionally regenerates coordinates."""
        self.img_size = img_size
        if reset_coords:
            self.make_coords(img_size=img_size)
    
    def get_last_selfattention(self, x: torch.Tensor, spix_label: torch.Tensor) -> torch.Tensor:
        """Extracts the self-attention map from the last transformer block."""
        x, mask = self.prepare_tokens(x, spix_label) # Prepare superpixel tokens
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, mask)
            else:
                # Assuming the block is EmptyMaskingBlock and its attn is EmptyMaskingAttention
                _, attn = blk(x, mask, return_attention=True) 
        
        return attn

    def get_selfattentions(self, x: torch.Tensor, spix_label: torch.Tensor) -> torch.Tensor:
        """Extracts all self-attention maps from all transformer blocks."""
        x, mask = self.prepare_tokens(x, spix_label) # Prepare superpixel tokens
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        attns = []
        for blk in self.blocks:
            x, attn = blk(x, mask, return_attention=True) # Assuming block returns attention
            attns.append(attn)

        attns = torch.stack(attns) # Stacks along a new dimension 0
        return attns
    
    def get_intermediate_features(self, x: torch.Tensor, spix_label: torch.Tensor) -> torch.Tensor:
        """Extracts features from all intermediate layers (input + each block output)."""
        features = []
        x, mask = self.prepare_tokens(x, spix_label) # Prepare superpixel tokens
        features.append(x) # Features after token preparation
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        for blk in self.blocks:
            x = blk(x, mask)
            features.append(x) # Features after each block
                
        features = torch.stack(features) # Stacks along a new dimension 0
        return features


@register_model
def suit_tiny_224(pretrained: bool = False, hierarchical_superpixels: bool = False, num_scales: int = 0, **kwargs):
    """
    SuiT-Tiny model variant.
    Image size: 224x224. Embed dim: 192. Depth: 12. Heads: 3. Base feature dim: 48.
    """
    model_kwargs = dict(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, base_dim=48,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        hierarchical_superpixels=hierarchical_superpixels,
        num_scales=num_scales,
        **kwargs
    )
    model = SuperpixelVisionTransformer(**model_kwargs)
    if pretrained:
        url="https://drive.google.com/uc?export=download&id=1Yvje-LLkHdeAo3RXrguzV-twNH2sn4Js"
        output = "suit_tiny_224.pth"
        if not os.path.exists(output):
            print(f"{output} not found. Downloading from {url}...")
            gdown.download(url, output, quiet=False)
        else:
            print(f"{output} already exists. Skipping download.")
        checkpoint = torch.load(output)
        model.load_state_dict(checkpoint["model"])

    return model

@register_model
def suit_small_224(pretrained: bool = False, hierarchical_superpixels: bool = False, num_scales: int = 0, **kwargs):
    """
    SuiT-Small model variant.
    Image size: 224x224. Embed dim: 384. Depth: 12. Heads: 6. Base feature dim: 96.
    """
    model_kwargs = dict(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, base_dim=96,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        hierarchical_superpixels=hierarchical_superpixels,
        num_scales=num_scales,
        **kwargs
    )
    model = SuperpixelVisionTransformer(**model_kwargs)
    if pretrained:
        url="https://drive.google.com/uc?export=download&id=1steBEtsFYnAtTyS29jJUb2DrsN_qQPPv"
        output = "suit_small_224.pth"
        if not os.path.exists(output):
            print(f"{output} not found. Downloading from {url}...")
            gdown.download(url, output, quiet=False)
        else:
            print(f"{output} already exists. Skipping download.")
        checkpoint = torch.load(output)
        model.load_state_dict(checkpoint["model"])

    return model

@register_model
def suit_base_224(pretrained: bool = False, hierarchical_superpixels: bool = False, num_scales: int = 0, **kwargs):
    """
    SuiT-Base model variant.
    Image size: 224x224. Embed dim: 768. Depth: 12. Heads: 12. Base feature dim: 192.
    """
    model_kwargs = dict(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, base_dim=192,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        hierarchical_superpixels=hierarchical_superpixels,
        num_scales=num_scales,
        **kwargs
    )
    model = SuperpixelVisionTransformer(**model_kwargs)
    if pretrained:
        url="https://drive.google.com/uc?export=download&id=1ZwS2Ig8YL3WjOWkqYJjzjiRSh1J_s0KW"
        output = "suit_base_224.pth"
        if not os.path.exists(output):
            print(f"{output} not found. Downloading from {url}...")
            gdown.download(url, output, quiet=False)
        else:
            print(f"{output} already exists. Skipping download.")
        checkpoint = torch.load(output)
        model.load_state_dict(checkpoint["model"])

    return model

@register_model
def suit_base_dino(pretrained: bool = False, hierarchical_superpixels: bool = False, num_scales: int = 0, **kwargs):
    """
    SuiT-Base model variant, intended for use with DINO pretraining.
    Image size: 224x224. Embed dim: 768. Depth: 12. Heads: 12. Base feature dim: 192.
    """
    model_kwargs = dict(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, base_dim=192,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        hierarchical_superpixels=hierarchical_superpixels,
        num_scales=num_scales,
        **kwargs
    )
    model = SuperpixelVisionTransformer(**model_kwargs)
    if pretrained:
        url="https://drive.google.com/uc?export=download&id=1jnr9WLEzyrv4AzKWT0U04PS6CBO9v0IH"
        output = "suit_base_dino.pth"
        if not os.path.exists(output):
            print(f"{output} not found. Downloading from {url}...")
            gdown.download(url, output, quiet=False)
        else:
            print(f"{output} already exists. Skipping download.")
        checkpoint = torch.load(output)
        model.load_state_dict(checkpoint["model"])

    return model