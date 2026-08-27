"""DepthAnything V2 official model architecture and weights handling."""

import os
import math
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from backend.config import settings

logger = logging.getLogger("depthwizard.depth")

# Official DepthAnything V2 model release URLs
MODEL_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
}


class LayerScale(nn.Module):
    """LayerScale module for Vision Transformer blocks."""

    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Attention(nn.Module):
    """Multi-head Self-Attention with projection for ViT."""

    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    """MLP / Feed-Forward Network for ViT."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm and LayerScale."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding with patch size 14x14."""

    def __init__(self, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DinoVisionTransformer(nn.Module):
    """DINOv2 Vision Transformer backbone for DepthAnything V2."""

    def __init__(
        self,
        patch_size: int = 14,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, intermediate_indices: Optional[List[int]] = None) -> List[torch.Tensor]:
        if intermediate_indices is None:
            intermediate_indices = [2, 5, 8, 11] if len(self.blocks) == 12 else [4, 11, 17, 23]

        B, C, H, W = x.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Dynamic 2D positional embedding interpolation for variable image sizes
        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]
        src_size = int(math.isqrt(patch_pos.shape[1]))
        patch_pos = patch_pos.reshape(1, src_size, src_size, self.embed_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(patch_h, patch_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).flatten(1, 2)
        pos_embed = torch.cat([cls_pos, patch_pos], dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed

        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in intermediate_indices:
                feat = x[:, 1:, :].permute(0, 2, 1).reshape(B, self.embed_dim, patch_h, patch_w)
                features.append(feat)
        return features


class ResidualConvUnit(nn.Module):
    """Residual Convolutional Unit for DPT RefineNet blocks."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature Fusion Block with residual conv units and bilinear upsampling."""

    def __init__(self, features: int):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, *xs: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            if size is None:
                size = res.shape[-2:]
            output = F.interpolate(output, size=size, mode="bilinear", align_corners=True)
            output = output + res
        output = self.resConfUnit2(output)
        output = self.out_conv(output)
        return output


class Scratch(nn.Module):
    """Scratch fusion container for DPT head."""

    def __init__(self, in_channels: List[int], features: int = 64):
        super().__init__()
        self.layer1_rn = nn.Conv2d(in_channels[0], features, 3, 1, 1, bias=False)
        self.layer2_rn = nn.Conv2d(in_channels[1], features, 3, 1, 1, bias=False)
        self.layer3_rn = nn.Conv2d(in_channels[2], features, 3, 1, 1, bias=False)
        self.layer4_rn = nn.Conv2d(in_channels[3], features, 3, 1, 1, bias=False)
        self.refinenet1 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet4 = FeatureFusionBlock(features)
        self.output_conv1 = nn.Conv2d(features, 32, kernel_size=3, stride=1, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )


class DPTHead(nn.Module):
    """Dense Prediction Transformer (DPT) Depth Head for DepthAnything V2."""

    def __init__(
        self,
        embed_dim: int = 384,
        features: int = 64,
        out_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = [48, 96, 192, 384]
        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim, out_ch, kernel_size=1, stride=1, padding=0)
            for out_ch in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])
        self.scratch = Scratch(out_channels, features)

    def forward(self, out_features: List[torch.Tensor], patch_h: int, patch_w: int) -> torch.Tensor:
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_3_rn.shape[-2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_2_rn.shape[-2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, size=layer_1_rn.shape[-2:])

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        return out.squeeze(1)


class DepthAnythingV2Model(nn.Module):
    """Production DepthAnything V2 Monocular Depth Estimation Network."""

    def __init__(
        self,
        encoder: str = "vits",
        features: int = 64,
        out_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.encoder_name = encoder
        encoder_configs = {
            "vits": {"embed_dim": 384, "depth": 12, "num_heads": 6, "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"embed_dim": 768, "depth": 12, "num_heads": 12, "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        cfg = encoder_configs.get(encoder, encoder_configs["vits"])

        self.pretrained = DinoVisionTransformer(
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
        )
        self.depth_head = DPTHead(
            embed_dim=cfg["embed_dim"],
            features=cfg["features"],
            out_channels=cfg["out_channels"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict continuous relative depth from RGB input tensor.

        Args:
            x: (B, 3, H, W) normalized image tensor.

        Returns:
            (B, H, W) relative inverse depth map.
        """
        h, w = x.shape[-2:]
        patch_h, patch_w = h // 14, w // 14
        feats = self.pretrained(x)
        depth = self.depth_head(feats, patch_h, patch_w)
        return depth


def ensure_checkpoint_exists(encoder: str = "vits") -> Optional[Path]:
    """Check for local checkpoint file; download from Hugging Face if missing."""
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    target_path = settings.models_dir / f"depth_anything_v2_{encoder}.pth"

    if target_path.is_file() and target_path.stat().st_size > 1_000_000:
        return target_path

    url = MODEL_URLS.get(encoder)
    if url:
        try:
            logger.info(f"Downloading DepthAnything V2 ({encoder}) weights from {url}...")
            urllib.request.urlretrieve(url, str(target_path))
            logger.info(f"Successfully downloaded weights to {target_path} ({target_path.stat().st_size // (1024*1024)} MB).")
            return target_path
        except Exception as e:
            logger.warning(f"Auto-download failed: {str(e)}. Attempting local search.")

    return None


def load_depth_anything_model(
    checkpoint_path: Optional[str] = None,
    encoder: str = "vits",
    device: torch.device = torch.device("cpu"),
) -> DepthAnythingV2Model:
    """Instantiate DepthAnything V2 and load pretrained official weights."""
    model = DepthAnythingV2Model(encoder=encoder)

    ckpt_path = None
    if checkpoint_path and Path(checkpoint_path).is_file():
        ckpt_path = Path(checkpoint_path)
    else:
        ckpt_path = ensure_checkpoint_exists(encoder)

    if ckpt_path and ckpt_path.is_file():
        try:
            logger.info(f"Loading DepthAnything V2 checkpoint from: {ckpt_path}")
            state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"Successfully loaded 100% of official DepthAnything V2 ({encoder}) weights.")
        except Exception as e:
            logger.warning(f"Strict weight loading issue ({str(e)}), trying non-strict loading.")
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception as e2:
                logger.error(f"Could not load weights: {str(e2)}")
    else:
        logger.warning(f"No checkpoint file found for DepthAnything V2 ({encoder}). Running with initialized weights.")

    model.to(device)
    model.eval()
    return model
