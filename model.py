from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)


class MLPBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FaceShapeExtractor(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.rotate_mlp = MLPBlock(6, channels, channels)
        self.fusion_mlp = MLPBlock(channels, channels * 2, channels * 2)

    def forward(self, corners: torch.Tensor) -> torch.Tensor:
        fea = (
            self.rotate_mlp(corners[:, :6])
            + self.rotate_mlp(corners[:, 3:9])
            + self.rotate_mlp(torch.cat([corners[:, 6:], corners[:, :3]], dim=1))
        ) / 3.0
        return self.fusion_mlp(fea)


class SpatialExtractor(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.spatial_mlp = MLPBlock(3, channels, channels)

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        return self.spatial_mlp(centers)


class StructuralExtractor(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.center_mlp = MLPBlock(6, channels, channels)
        self.neighbor_mlp = MLPBlock(9, channels, channels)
        self.fusion_mlp = MLPBlock(channels * 2, channels * 4, channels * 4)

    def forward(self, normals: torch.Tensor, neighbor_normals: torch.Tensor) -> torch.Tensor:
        fea_center = (
            self.center_mlp(torch.cat([normals, neighbor_normals[:, :3]], dim=1))
            + self.center_mlp(torch.cat([normals, neighbor_normals[:, 3:6]], dim=1))
            + self.center_mlp(torch.cat([normals, neighbor_normals[:, 6:]], dim=1))
        ) / 3.0
        fea_neighbor = self.neighbor_mlp(neighbor_normals)
        fea = self.fusion_mlp(torch.cat([fea_center, fea_neighbor], dim=1))
        return fea


class GeometryEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, num_groups: int = 33) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.spatial = SpatialExtractor(channels=128)
        self.face_shape = FaceShapeExtractor(channels=64)
        self.structural = StructuralExtractor(channels=32)
        self.color_mlp = MLPBlock(9, 64, 128)

        fusion_in = 128 * 4
        self.group_fusion = MLPBlock(fusion_in, 256, 256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.group_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bsz, num_locals, num_neighbors, channels = inputs.shape
        if channels != 33:
            raise ValueError(f"Expected geometry channel size 33, got {channels}")

        x = rearrange(inputs, "b l n c -> (b l) c n")
        centers = x[:, 0:3]
        corners = x[:, 3:12]
        colors = x[:, 12:21]
        normals = x[:, 21:24]
        neighbor_normals = x[:, 24:33]

        fea_spatial = self.spatial(centers)
        fea_shape = self.face_shape(corners)
        fea_color = self.color_mlp(colors)
        fea_struct = self.structural(normals, neighbor_normals)

        fea = torch.cat([fea_spatial, fea_shape, fea_color, fea_struct], dim=1)
        fea = self.group_fusion(fea).transpose(1, 2)
        fea = self.group_transformer(fea)
        fea = fea.mean(dim=1)
        fea = self.output_proj(fea)
        return rearrange(fea, "(b l) c -> b l c", b=bsz, l=num_locals)


class SwinTextureEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        # model_name: str = "resnet18.a1_in1k",
        pretrained: bool = True,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.visual = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        visual_dim = self.visual.num_features
        self.input_size = 224
        # self.image_proj = nn.Sequential(
        #     nn.Linear(visual_dim, visual_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(visual_dim),
        # )
        # self.depth_proj = nn.Sequential(
        #     nn.Linear(visual_dim, visual_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(visual_dim),
        # )
        # self.normal_proj = nn.Sequential(
        #     nn.Linear(visual_dim, visual_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(visual_dim),
        # )

        self.fusion = nn.Sequential(
            nn.Linear(visual_dim*3, visual_dim),
            nn.GELU(),
            nn.LayerNorm(visual_dim),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(visual_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def _patches_to_mosaic(self, patches: torch.Tensor) -> torch.Tensor:
        bsz, num_locals, num_views, channels, height, width = patches.shape
        grid_size = int(num_views ** 0.5)
        if grid_size * grid_size != num_views:
            raise ValueError(f"Expected a square number of texture views, got {num_views}")
        return rearrange(
            patches,
            "b l (gh gw) c h w -> b l c (gh h) (gw w)",
            gh=grid_size,
            gw=grid_size,
        )

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
        std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
        return (images - mean) / std

    def forward(self, mosaics: torch.Tensor, depth: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        if mosaics.dim() == 6:
            mosaics = self._patches_to_mosaic(mosaics)
            depth = self._patches_to_mosaic(depth)
            normal = self._patches_to_mosaic(normal)
        elif mosaics.dim() != 5:
            raise ValueError(f"Expected texture input with 5 or 6 dims, got shape {tuple(mosaics.shape)}")

        # if float(mosaics.detach().amin()) >= 0.0 and float(mosaics.detach().amax()) <= 1.0:
        mosaics = self._normalize_images(mosaics)
        depth = self._normalize_images(depth)
        normal = self._normalize_images(normal)


        bsz, num_locals, channels, height, width = mosaics.shape
        images = rearrange(mosaics, "b l c h w -> (b l) c h w")
        depth = rearrange(depth, "b l c h w -> (b l) c h w")
        normal = rearrange(normal, "b l c h w -> (b l) c h w")

        # if height != self.input_size or width != self.input_size:
        #     images = F.interpolate(images, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        #     depth = F.interpolate(depth, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        #     normal = F.interpolate(normal, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

        # image_feat = self.image_proj(self.visual(images)).view(bsz, num_locals, -1)
        # depth_feat = self.depth_proj(self.visual(depth)).view(bsz, num_locals, -1)
        # normal_feat = self.normal_proj(self.visual(normal)).view(bsz, num_locals, -1)
        image_feat = self.visual(images).view(bsz, num_locals, -1)
        depth_feat = self.visual(depth).view(bsz, num_locals, -1)
        normal_feat = self.visual(normal).view(bsz, num_locals, -1)
        fused = self.fusion(torch.cat([image_feat, depth_feat, normal_feat], dim=-1))
        # fused = self.fusion(image_feat)
        return self.output_proj(fused)


class LocalGatedFusion(nn.Module):
    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, geo_feat: torch.Tensor, tex_feat: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([geo_feat, tex_feat], dim=-1))
        fused = gate * geo_feat + (1.0 - gate) * tex_feat
        return self.proj(torch.cat([geo_feat, tex_feat, fused], dim=-1))


class RefGuidedCrossAttention(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, query_tokens: torch.Tensor, ref_tokens: torch.Tensor) -> torch.Tensor:
        query_norm = self.norm_q(query_tokens)
        ref_norm = self.norm_kv(ref_tokens)
        attended, _ = self.attn(query=query_norm, key=ref_norm, value=ref_norm, need_weights=False)
        return self.out_proj(torch.cat([query_tokens, attended], dim=-1))


class GlobalAggregator(nn.Module):
    def __init__(self, embed_dim: int = 256, num_tokens: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.weight_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    # def forward(self, tokens: torch.Tensor, reliability: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    #     x = tokens + self.pos_embed[:, : tokens.size(1)]
    #     x = self.encoder(x)
    #     x = self.norm(x)

    #     if reliability is None:
    #         reliability = torch.ones(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)
    #     else:
    #         reliability = reliability.unsqueeze(-1).to(dtype=x.dtype)

    #     logits = self.weight_proj(torch.cat([x, reliability], dim=-1)).squeeze(-1)
    #     logits = logits.masked_fill(reliability.squeeze(-1) <= 0.0, -1e4)
    #     weights = torch.softmax(logits, dim=1)
    #     pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
    #     return pooled, weights

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x = tokens + self.pos_embed[:, : tokens.size(1)]
        # x = self.encoder(x)
        x = self.encoder(tokens)
        x = self.norm(x)

        logits = self.weight_proj(x).squeeze(-1)
        
        weights = torch.softmax(logits, dim=1)
        # pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        # pooled = torch.mean(x * weights.unsqueeze(-1), dim=1)
        pooled = 0.7 * torch.sum(x * weights.unsqueeze(-1), dim=1) + 0.3 * torch.mean(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class DiffFeatureFusion(nn.Module):
    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 5, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        dist_tokens: torch.Tensor,
        ref_tokens: torch.Tensor,
        refined_dist: torch.Tensor,
    ) -> torch.Tensor:
        signed_diff = refined_dist - ref_tokens
        abs_diff = signed_diff.abs()
        product = refined_dist * ref_tokens
        fusion = torch.cat(
            [
                refined_dist,
                ref_tokens,
                signed_diff,
                abs_diff,
                product,
            ],
            dim=-1,
        )
        return self.proj(fusion)


class MeshBranch(nn.Module):
    def __init__(self, embed_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        self.geometry_encoder = GeometryEncoder(embed_dim=embed_dim)
        self.texture_encoder = SwinTextureEncoder(pretrained=pretrained, embed_dim=embed_dim)
        self.fusion = LocalGatedFusion(embed_dim=embed_dim)
        self.aggregator = GlobalAggregator(embed_dim=embed_dim)

    def forward(
        self,
        mosaics: torch.Tensor,
        depth: torch.Tensor, 
        normal: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        local_tokens = self.encode_local_tokens(mosaics, depth, normal, geometry)
        return self.aggregate(local_tokens)[0]

    def encode_local_tokens(self, mosaics: torch.Tensor, depth: torch.Tensor, normal: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        geo_feat = self.geometry_encoder(geometry)
        tex_feat = self.texture_encoder(mosaics, depth, normal)
        return self.fusion(geo_feat, tex_feat)

    def aggregate(
        self,
        local_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.aggregator(local_tokens)


class FR_MeshQA(nn.Module):
    def __init__(self, embed_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        self.branch = MeshBranch(embed_dim=embed_dim, pretrained=pretrained)
        self.ref_guided = RefGuidedCrossAttention(embed_dim=embed_dim)
        self.ref_guided_scale = nn.Parameter(torch.tensor(0.25))
        self.diff_fusion = DiffFeatureFusion(embed_dim=embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 192),
            nn.GELU(),
            nn.Dropout(p=0.45),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        dist_mosaic: torch.Tensor,
        dist_depth: torch.Tensor,
        dist_normal: torch.Tensor,
        dist_geometry: torch.Tensor,
        ref_mosaic: torch.Tensor,
        ref_depth: torch.Tensor,
        ref_normal: torch.Tensor,
        ref_geometry: torch.Tensor,
    ) -> torch.Tensor:
        dist_tokens = self.branch.encode_local_tokens(dist_mosaic, dist_depth, dist_normal, dist_geometry)
        ref_tokens = self.branch.encode_local_tokens(ref_mosaic, ref_depth, ref_normal, ref_geometry)
        ref_guided_dist = self.ref_guided(dist_tokens, ref_tokens)
        fused_dist = dist_tokens + self.ref_guided_scale * (ref_guided_dist - dist_tokens)
        diff_tokens = self.diff_fusion(dist_tokens, ref_tokens, fused_dist)
        pooled, weights = self.branch.aggregate(diff_tokens)
        score = self.regressor(pooled)
        return score.squeeze(-1)


class NR_MeshQA(nn.Module):
    def __init__(self, embed_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        self.branch = MeshBranch(embed_dim=embed_dim, pretrained=pretrained)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        dist_mosaic: torch.Tensor,
        dist_depth: torch.Tensor,
        dist_normal: torch.Tensor,
        dist_geometry: torch.Tensor,
    ) -> torch.Tensor:
        dist_repr = self.branch(dist_mosaic, dist_depth, dist_normal, dist_geometry)
        score = self.regressor(dist_repr)
        return score.squeeze(-1)


# if __name__=='__main__':
#     # model = FR_MeshQA()
#     model = NR_MeshQA()
#     dist_mosaic = torch.rand(4, 32, 16, 3, 56, 56)
#     dist_depth = torch.rand(4, 32, 16, 3, 56, 56)
#     dist_normal = torch.rand(4, 32, 16, 3, 56, 56)
#     dist_geometry = torch.rand(4, 32, 33, 33)
#     ref_mosaic = torch.rand(4, 32, 16, 3, 56, 56)
#     ref_depth = torch.rand(4, 32, 16, 3, 56, 56)
#     ref_normal = torch.rand(4, 32, 16, 3, 56, 56)
#     ref_geometry = torch.rand(4, 32, 33, 33)
#     score = model(
#     dist_mosaic,
#     dist_depth,
#     dist_normal,
#     dist_geometry,
#     # ref_mosaic,
#     # ref_depth,
#     # ref_normal,
#     # ref_geometry,
#     )
#     print(score.shape)