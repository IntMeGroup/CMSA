import argparse
import logging
import os
from typing import Dict, Tuple, Optional, List, Set

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pytorch3d.ops import knn_points, sample_farthest_points
from pytorch3d.renderer import (
    AmbientLights,
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    look_at_view_transform,
)

from utils import load_objs_as_meshes


logging.basicConfig(
    filename="./process_aligned_fr-yn2023-112-512.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)


def log_info(msg: str, *args) -> None:
    logging.info(msg, *args)
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


# =========================================================
# Basic geometry
# =========================================================
def get_face_centers_and_normals(
    faces: torch.Tensor,
    verts: torch.Tensor,
    face_normals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    face_centers = verts[faces].mean(dim=1)
    face_normals = F.normalize(face_normals, dim=-1)
    return face_centers, face_normals


def get_face_adjacency(faces: torch.Tensor, verts: torch.Tensor) -> torch.Tensor:
    num_faces = int(faces.shape[0])
    face_adjacency_cpu = -1 * torch.ones((num_faces, 3), dtype=torch.long)

    edge_to_face = {}
    faces_cpu = faces.detach().cpu().numpy()
    for f in range(num_faces):
        v0, v1, v2 = [int(x) for x in faces_cpu[f]]
        tri_edges = [(v0, v1), (v1, v2), (v2, v0)]
        for ei, (a, b) in enumerate(tri_edges):
            key = (a, b) if a < b else (b, a)
            if key in edge_to_face:
                of, oei = edge_to_face[key]
                face_adjacency_cpu[f, ei] = of
                face_adjacency_cpu[of, oei] = f
            else:
                edge_to_face[key] = (f, ei)

    return face_adjacency_cpu.to(device=verts.device)


def get_colors_from_mesh(mesh, colored_mesh: bool) -> torch.Tensor:
    if colored_mesh:
        return mesh.textures.verts_features_packed()

    if mesh.textures is None:
        return torch.ones_like(mesh.verts_packed())

    if not hasattr(mesh.textures, "verts_uvs_padded") or not hasattr(mesh.textures, "maps_padded"):
        return torch.ones_like(mesh.verts_packed())

    v = mesh.verts_packed().shape[0]
    verts_uvs = mesh.textures.verts_uvs_padded()
    faces_uv_idx = mesh.textures.faces_uvs_padded()
    faces = mesh.faces_packed()

    v_to_uv = torch.zeros((v, 2), device=mesh.device)
    v_to_uv.index_copy_(0, faces.view(-1), verts_uvs[0, faces_uv_idx.view(-1)])

    pixel_uvs = v_to_uv * 2.0 - 1.0
    pixel_uvs[:, 1] = 1.0 - pixel_uvs[:, 1]
    pixel_uvs = pixel_uvs.view(1, 1, -1, 2)

    tex_maps = mesh.textures.maps_padded()
    colors = F.grid_sample(
        tex_maps.permute(0, 3, 1, 2),
        pixel_uvs,
        mode="bilinear",
        align_corners=False,
    ).squeeze().permute(1, 0)
    return colors


def get_face_attributes(
    face_indices: torch.Tensor,
    faces: torch.Tensor,
    verts: torch.Tensor,
    face_normals: torch.Tensor,
    face_adjacency: torch.Tensor,
    colors: torch.Tensor,
) -> torch.Tensor:
    target_faces = faces[face_indices]
    target_colors = colors[target_faces].reshape(face_indices.shape[0], 9)

    v0 = verts[target_faces[:, 0]]
    v1 = verts[target_faces[:, 1]]
    v2 = verts[target_faces[:, 2]]
    centers = (v0 + v1 + v2) / 3.0
    normals = face_normals[face_indices]

    c0 = v0 - centers
    c1 = v1 - centers
    c2 = v2 - centers
    corners = torch.stack([c0, c1, c2], dim=1).reshape(face_indices.shape[0], 9)

    neighbor_idx = face_adjacency[face_indices].clone()
    valid_neighbor = neighbor_idx >= 0
    neighbor_idx = neighbor_idx.clamp(min=0)
    neighbor_normals = face_normals[neighbor_idx]
    neighbor_normals = neighbor_normals * valid_neighbor.unsqueeze(-1)
    neighbor_normals = neighbor_normals.reshape(face_indices.shape[0], 9)

    return torch.cat([centers, corners, target_colors, normals, neighbor_normals], dim=-1)


def collect_geo_indices(fps_idx: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
    n, k = knn_idx.shape
    all_idx = torch.empty((n * (k + 1),), dtype=torch.long, device=fps_idx.device)
    ptr = 0
    for i in range(n):
        all_idx[ptr] = fps_idx[i]
        ptr += 1
        all_idx[ptr:ptr + k] = knn_idx[i]
        ptr += k
    return all_idx


# =========================================================
# Saliency + diverse anchor sampling
# =========================================================
def compute_face_saliency(
    faces: torch.Tensor,
    verts: torch.Tensor,
    face_normals: torch.Tensor,
    face_adjacency: torch.Tensor,
    colors: torch.Tensor,
) -> torch.Tensor:
    face_colors = colors[faces].mean(dim=1)  # [F, 3]
    face_normals = F.normalize(face_normals, dim=-1)

    neighbor_idx = face_adjacency.clone()
    valid = neighbor_idx >= 0
    neighbor_idx = neighbor_idx.clamp(min=0)

    neigh_normals = face_normals[neighbor_idx]
    neigh_colors = face_colors[neighbor_idx]

    normal_diff = 1.0 - (face_normals.unsqueeze(1) * neigh_normals).sum(dim=-1)
    normal_diff = (normal_diff * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp(min=1.0)

    color_diff = torch.norm(face_colors.unsqueeze(1) - neigh_colors, dim=-1)
    color_diff = (color_diff * valid.float()).sum(dim=1) / valid.float().sum(dim=1).clamp(min=1.0)

    normal_diff = (normal_diff - normal_diff.min()) / (normal_diff.max() - normal_diff.min() + 1e-8)
    color_diff = (color_diff - color_diff.min()) / (color_diff.max() - color_diff.min() + 1e-8)

    saliency = 0.7 * normal_diff + 0.3 * color_diff
    return saliency


def sample_faces_fps_knn_diverse(
    faces: torch.Tensor,
    verts: torch.Tensor,
    face_normals: torch.Tensor,
    colors: torch.Tensor,
    face_adjacency: torch.Tensor,
    n_faces: int,
    k_neighbors: int,
    saliency_keep_ratio: float = 0.5,
    overlap_thresh: float = 0.35,
    oversample_factor: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    face_centers, _ = get_face_centers_and_normals(faces, verts, face_normals)
    num_faces = faces.shape[0]

    saliency = compute_face_saliency(faces, verts, face_normals, face_adjacency, colors)

    keep_num = max(n_faces * 2, int(num_faces * saliency_keep_ratio))
    keep_num = min(keep_num, num_faces)

    top_idx = torch.topk(saliency, k=keep_num, dim=0).indices
    cand_centers = face_centers[top_idx].unsqueeze(0)

    fps_k = min(keep_num, max(n_faces * oversample_factor, n_faces))
    _, fps_local_idx = sample_farthest_points(cand_centers, K=fps_k)
    fps_candidates = top_idx[fps_local_idx.squeeze(0)]

    _, knn_idx, _ = knn_points(
        face_centers[fps_candidates].unsqueeze(0),
        face_centers.unsqueeze(0),
        K=min(k_neighbors, num_faces),
    )
    knn_idx = knn_idx.squeeze(0)

    selected_anchors = []
    selected_groups: List[Set[int]] = []

    for i in range(fps_candidates.shape[0]):
        anchor = fps_candidates[i]
        group = set(knn_idx[i].tolist() + [int(anchor.item())])

        keep = True
        for g in selected_groups:
            inter = len(group & g)
            union = len(group | g)
            iou = inter / max(union, 1)
            if iou > overlap_thresh:
                keep = False
                break

        if keep:
            selected_anchors.append(anchor)
            selected_groups.append(group)

        if len(selected_anchors) >= n_faces:
            break

    if len(selected_anchors) < n_faces:
        selected_set = set([int(x.item()) for x in selected_anchors])
        for i in range(fps_candidates.shape[0]):
            anchor = fps_candidates[i]
            if int(anchor.item()) not in selected_set:
                selected_anchors.append(anchor)
                if len(selected_anchors) >= n_faces:
                    break

    fps_idx = torch.stack(selected_anchors[:n_faces], dim=0)

    _, knn_idx, _ = knn_points(
        face_centers[fps_idx].unsqueeze(0),
        face_centers.unsqueeze(0),
        K=min(k_neighbors, num_faces),
    )
    knn_idx = knn_idx.squeeze(0)

    return fps_idx, knn_idx


# =========================================================
# Matching
# =========================================================
def match_ref_anchors_to_dist_faces(
    ref_anchor_centers: torch.Tensor,
    ref_anchor_normals: torch.Tensor,
    dist_face_centers: torch.Tensor,
    dist_face_normals: torch.Tensor,
    alpha_center: float = 1.0,
    alpha_normal: float = 0.2,
    chunk_size: int = 2048,
) -> torch.Tensor:
    ref_anchor_normals = F.normalize(ref_anchor_normals, dim=-1)
    dist_face_normals = F.normalize(dist_face_normals, dim=-1)

    n = ref_anchor_centers.shape[0]
    out = []

    for st in range(0, n, chunk_size):
        ed = min(st + chunk_size, n)
        rc = ref_anchor_centers[st:ed]
        rn = ref_anchor_normals[st:ed]

        center_dist = torch.cdist(rc, dist_face_centers, p=2)
        cos_sim = torch.matmul(rn, dist_face_normals.t()).clamp(-1.0, 1.0)
        normal_dist = 1.0 - cos_sim

        score = alpha_center * center_dist + alpha_normal * normal_dist
        idx = torch.argmin(score, dim=1)
        out.append(idx)

    return torch.cat(out, dim=0)


def build_knn_groups_from_anchor_faces(
    anchor_face_idx: torch.Tensor,
    face_centers: torch.Tensor,
    k_neighbors: int,
) -> torch.Tensor:
    anchor_centers = face_centers[anchor_face_idx].unsqueeze(0)
    all_centers = face_centers.unsqueeze(0)
    _, knn_idx, _ = knn_points(anchor_centers, all_centers, K=k_neighbors)
    return knn_idx.squeeze(0)


# =========================================================
# Camera utilities (NEW canonical multi-view)
# =========================================================
def build_local_frame(normal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = normal / (normal.norm() + 1e-8)

    if torch.abs(n[1]) < 0.9:
        helper = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)
    else:
        helper = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype)

    t1 = torch.cross(n, helper, dim=0)
    t1 = t1 / (t1.norm() + 1e-8)

    t2 = torch.cross(n, t1, dim=0)
    t2 = t2 / (t2.norm() + 1e-8)

    return n, t1, t2


def generate_canonical_view_dirs(
    face_normal: torch.Tensor,
    num_views: int,
    spread: float = 0.55,
    use_backside: bool = False,
) -> List[torch.Tensor]:
    n, t1, t2 = build_local_frame(face_normal)
    base = -n if use_backside else n

    dirs = [
        # base,
        F.normalize(base + spread * t1, dim=0),
        F.normalize(base - spread * t1, dim=0),
        F.normalize(base + spread * t2, dim=0),
        F.normalize(base - spread * t2, dim=0),
        F.normalize(base + spread * (t1 + t2), dim=0),
        F.normalize(base + spread * (t1 - t2), dim=0),
        F.normalize(base + spread * (-t1 + t2), dim=0),
        F.normalize(base + spread * (-t1 - t2), dim=0),

        F.normalize(base + spread * (t1/2 + t2), dim=0),
        F.normalize(base + spread * (t1/2 - t2), dim=0),
        F.normalize(base + spread * (-t1/2 + t2), dim=0),
        F.normalize(base + spread * (-t1/2 - t2), dim=0),
        F.normalize(base + spread * (t1 + t2/2), dim=0),
        F.normalize(base + spread * (t1 - t2/2), dim=0),
        F.normalize(base + spread * (-t1 + t2/2), dim=0),
        F.normalize(base + spread * (-t1 - t2/2), dim=0),
    ]
    return dirs[:num_views]


def perturb_view_dir(view_dir: torch.Tensor, noise_scale: float = 0.08) -> torch.Tensor:
    noise = torch.randn_like(view_dir)
    noise = noise - torch.dot(noise, view_dir) * view_dir
    noise = noise / (noise.norm() + 1e-8)
    out = view_dir + noise_scale * noise
    out = out / (out.norm() + 1e-8)
    return out


def sample_camera_from_view_dir(
    face_center: torch.Tensor,
    view_dir: torch.Tensor,
    dist_range: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    dist_min, dist_max = dist_range
    dist = torch.rand(1, device=face_center.device) * (dist_max - dist_min) + dist_min
    cam_pos = face_center + view_dir * dist

    r, t = look_at_view_transform(
        eye=cam_pos.unsqueeze(0),
        at=face_center.unsqueeze(0),
        up=((0, 1, 0),),
    )
    return r, t


# =========================================================
# Rendering
# =========================================================
def crop_with_padding(
    image: torch.Tensor,
    center_y: int,
    center_x: int,
    h: int,
    w: int,
    fill_value: float = 1.0,
) -> torch.Tensor:
    out = torch.full((h, w, image.shape[-1]), fill_value=fill_value, device=image.device, dtype=image.dtype)
    half_h, half_w = h // 2, w // 2

    y0, y1 = center_y - half_h, center_y - half_h + h
    x0, x1 = center_x - half_w, center_x - half_w + w

    src_y0, src_y1 = max(0, y0), min(image.shape[0], y1)
    src_x0, src_x1 = max(0, x0), min(image.shape[1], x1)
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return out

    dst_y0, dst_y1 = src_y0 - y0, src_y1 - y0
    dst_x0, dst_x1 = src_x0 - x0, src_x1 - x0
    out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def render_depth_normal(mesh, fragments):
    depth = fragments.zbuf[0, ..., 0:1]
    pix_to_face = fragments.pix_to_face[0, ..., 0]

    face_normals = mesh.faces_normals_packed()
    face_normals_pix = face_normals[pix_to_face.clamp(min=0)]

    mask = pix_to_face >= 0
    normal = F.normalize(face_normals_pix, dim=-1)
    normal[~mask] = 0.0
    depth[~mask] = 0.0
    return depth, normal


def render_face_patch(
    mesh,
    renderer,
    rasterizer,
    target_face_ids: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    h: int,
    w: int,
    min_target_pixels: int,
):
    cameras = FoVPerspectiveCameras(device=mesh.device, R=r, T=t)
    fragments = rasterizer(mesh, cameras=cameras)
    pix_to_face = fragments.pix_to_face[0, ..., 0]
    target_face_ids = target_face_ids.to(device=mesh.device, dtype=pix_to_face.dtype)
    target_mask = (pix_to_face.unsqueeze(-1) == target_face_ids.view(1, 1, -1)).any(dim=-1)
    target_pixels = int(target_mask.sum().item())

    if target_pixels < min_target_pixels:
        return None, None, None, False, target_pixels

    ys, xs = torch.where(target_mask)
    center_y = int(ys.float().mean().item())
    center_x = int(xs.float().mean().item())

    image = renderer(mesh, cameras=cameras)[0, ..., :3]
    depth, normal = render_depth_normal(mesh, fragments)

    patch = crop_with_padding(image, center_y, center_x, h, w, fill_value=1.0)
    depth_patch = crop_with_padding(depth, center_y, center_x, h, w, fill_value=0.0)
    normal_patch = crop_with_padding(normal, center_y, center_x, h, w, fill_value=0.0)
    patch_mask = crop_with_padding(target_mask.float().unsqueeze(-1), center_y, center_x, h, w, fill_value=0.0)
    patch_target_pixels = int((patch_mask[..., 0] > 0.5).sum().item())
    return patch, depth_patch, normal_patch, patch_target_pixels >= min_target_pixels, patch_target_pixels


def count_target_pixels(
    mesh,
    rasterizer,
    target_face_ids: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
) -> int:
    cameras = FoVPerspectiveCameras(device=mesh.device, R=r, T=t)
    fragments = rasterizer(mesh, cameras=cameras)
    pix_to_face = fragments.pix_to_face[0, ..., 0]
    target_face_ids = target_face_ids.to(device=mesh.device, dtype=pix_to_face.dtype)
    target_mask = (pix_to_face.unsqueeze(-1) == target_face_ids.view(1, 1, -1)).any(dim=-1)
    return int(target_mask.sum().item())


def build_face_texture_with_camera(
    mesh,
    face_idx: int,
    target_face_ids: torch.Tensor,
    renderer,
    rasterizer,
    coarse_rasterizer,
    p_views: int,
    h: int,
    w: int,
    dist_range: Tuple[float, float],
    jitter_angle: float,
    max_retry: int,
    min_target_pixels: int,
):
    """
    Heavy reference version with canonical multi-view directions.
    """
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    face_normals = mesh.faces_normals_packed()

    face = faces[face_idx]
    center = verts[face].mean(0)
    normal = face_normals[face_idx]

    patches = torch.ones((p_views, h, w, 3), device=mesh.device)
    depths = torch.ones((p_views, h, w, 1), device=mesh.device)
    normals = torch.ones((p_views, h, w, 3), device=mesh.device)
    valid = torch.zeros((p_views,), dtype=torch.bool, device=mesh.device)

    camera_R = torch.zeros((p_views, 3, 3), device=mesh.device)
    camera_T = torch.zeros((p_views, 3), device=mesh.device)

    total_retries = 0

    mesh_radius = (verts - verts.mean(dim=0, keepdim=True)).norm(dim=1).max().item()
    mesh_radius = max(mesh_radius, 1e-6)
    eff_dist_range = (dist_range[0] * mesh_radius, dist_range[1] * mesh_radius)

    if isinstance(coarse_rasterizer.raster_settings.image_size, (tuple, list)):
        render_size = int(coarse_rasterizer.raster_settings.image_size[0])
    else:
        render_size = int(coarse_rasterizer.raster_settings.image_size)
    adaptive_min_pixels = int((render_size * render_size) / max(1, int(faces.shape[0])))
    adaptive_min_pixels = max(1, min(min_target_pixels, adaptive_min_pixels))

    target_non_blank_views = max(1, p_views - 1)

    canonical_dirs = generate_canonical_view_dirs(
        face_normal=normal,
        num_views=p_views,
        spread=0.55,
        use_backside=False,
    )

    for view_idx in range(p_views):
        best_pixels = 0
        best_rt = None

        valid_so_far = int(valid[:view_idx].sum().item())
        views_left = p_views - view_idx
        must_succeed_now = (target_non_blank_views - valid_so_far) >= views_left
        retry_budget = max_retry * 2 if must_succeed_now else max_retry

        base_dir = canonical_dirs[view_idx]

        for retry_idx in range(retry_budget):
            # small perturbation around a canonical direction
            noise_scale = 0.04 if retry_idx == 0 else min(0.12, 0.04 + 0.02 * retry_idx)
            cur_dir = perturb_view_dir(base_dir, noise_scale=noise_scale)
            r, t = sample_camera_from_view_dir(center, cur_dir, eff_dist_range)

            target_pixels = count_target_pixels(mesh, coarse_rasterizer, target_face_ids, r, t)
            total_retries += 1

            if target_pixels > best_pixels:
                best_pixels = target_pixels
                best_rt = (r, t)

            if target_pixels >= adaptive_min_pixels:
                patch, depth_patch, normal_patch, ok, _ = render_face_patch(
                    mesh=mesh,
                    renderer=renderer,
                    rasterizer=rasterizer,
                    target_face_ids=target_face_ids,
                    r=r,
                    t=t,
                    h=h,
                    w=w,
                    min_target_pixels=1,
                )
                if patch is not None and ok:
                    patches[view_idx] = patch
                    depths[view_idx] = depth_patch
                    normals[view_idx] = normal_patch
                    valid[view_idx] = True
                    camera_R[view_idx] = r.squeeze(0)
                    camera_T[view_idx] = t.squeeze(0)
                    break

        if (not valid[view_idx]) and (best_rt is not None) and (best_pixels > 0):
            br, bt = best_rt
            weak_patch, depth_patch, normal_patch, ok, _ = render_face_patch(
                mesh=mesh,
                renderer=renderer,
                rasterizer=rasterizer,
                target_face_ids=target_face_ids,
                r=br,
                t=bt,
                h=h,
                w=w,
                min_target_pixels=1,
            )
            if weak_patch is not None:
                patches[view_idx] = weak_patch
                depths[view_idx] = depth_patch
                normals[view_idx] = normal_patch
                valid[view_idx] = True
                camera_R[view_idx] = br.squeeze(0)
                camera_T[view_idx] = bt.squeeze(0)

    depths = depths / (depths.max() + 1e-6)
    normals = (normals + 1.0) / 2.0
    return patches, depths, normals, valid, total_retries, camera_R, camera_T


def render_anchor_patches_with_given_cameras(
    mesh,
    renderer,
    rasterizer,
    target_face_ids: torch.Tensor,
    R_all: torch.Tensor,
    T_all: torch.Tensor,
    h: int,
    w: int,
    min_target_pixels: int,
):
    p_views = R_all.shape[0]
    patches = torch.ones((p_views, h, w, 3), device=mesh.device)
    depths = torch.zeros((p_views, h, w, 1), device=mesh.device)
    normals = torch.zeros((p_views, h, w, 3), device=mesh.device)
    valid = torch.zeros((p_views,), dtype=torch.bool, device=mesh.device)

    for view_idx in range(p_views):
        r = R_all[view_idx].unsqueeze(0)
        t = T_all[view_idx].unsqueeze(0)

        patch, depth_patch, normal_patch, ok, _ = render_face_patch(
            mesh=mesh,
            renderer=renderer,
            rasterizer=rasterizer,
            target_face_ids=target_face_ids,
            r=r,
            t=t,
            h=h,
            w=w,
            min_target_pixels=1,
        )
        if patch is not None:
            patches[view_idx] = patch
            depths[view_idx] = depth_patch
            normals[view_idx] = normal_patch
            valid[view_idx] = ok

    depths = depths / (depths.max() + 1e-6)
    normals = (normals + 1.0) / 2.0
    return patches, depths, normals, valid


# =========================================================
# Path helpers
# =========================================================
def guess_reference_obj_for_dist(mesh_path: str, dataset_name: str) -> Optional[str]:
    if dataset_name == "SJTU-TMQA":
        parts = mesh_path.split(os.sep)
        if "distortion_dataset" not in parts:
            return None
        idx = parts.index("distortion_dataset")
        root_dir = os.sep.join(parts[:idx])
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        model = mesh_name.split("_")[0]
        return os.path.join(root_dir, "reference_dataset", model, f"{model}.obj")

    elif dataset_name == "TSMD":
        parts = mesh_path.split(os.sep)
        if "Distortion" not in parts:
            return None
        idx = parts.index("Distortion")
        root_dir = os.sep.join(parts[:idx])
        source = parts[idx + 1]
        return os.path.join(root_dir, "Reference", source, f"{source}_C0-L5_deq_tri.obj")

    elif dataset_name == "VCMesh":
        parts = mesh_path.split(os.sep)
        if "Stimuli_Format_Obj" not in parts:
            return None
        idx = parts.index("Stimuli_Format_Obj")
        source = parts[idx + 1]
        root_dir = os.sep.join(parts[:idx])
        return os.path.join(root_dir, "Stimuli_Format_Obj", source, f"{source}_Ref_0.obj")

    elif dataset_name == "YN2023":
        parts = mesh_path.split(os.sep)
        try:
            source = parts[-4]
            root_dir = os.sep.join(parts[:-4])
            return os.path.join(root_dir, source, "source", f"{source}.obj")
        except Exception:
            return None

    return None


def is_reference_obj(mesh_path: str, dataset_name: str) -> bool:
    ref_guess = guess_reference_obj_for_dist(mesh_path, dataset_name)
    if ref_guess is None:
        return False
    return os.path.normpath(ref_guess) == os.path.normpath(mesh_path)


# =========================================================
# Reference preprocess
# =========================================================
def preprocess_reference_mesh_diverse(
    mesh_path: str,
    device: torch.device,
    renderer,
    raster_settings: RasterizationSettings,
    coarse_raster_settings: RasterizationSettings,
    dataset_name: str,
    output_replace: str,
    n_faces: int,
    k_neighbors: int,
    p_views: int,
    patch_h: int,
    patch_w: int,
    colored_mesh: bool,
    dist_range: Tuple[float, float],
    jitter_angle: float,
    max_retry: int,
    min_target_pixels: int,
    cover_exist: bool,
    saliency_keep_ratio: float,
    overlap_thresh: float,
    oversample_factor: int,
) -> str:
    npz_path = mesh_path.replace(dataset_name, output_replace).replace(".obj", ".npz")
    save_dir = os.path.dirname(npz_path)
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(npz_path) and (not cover_exist):
        log_info("Skip existing reference %s", npz_path)
        return npz_path

    mesh = load_objs_as_meshes([mesh_path], device=device, vcmesh=colored_mesh)
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    face_normals = mesh.faces_normals_packed()

    n_eff = min(n_faces, faces.shape[0])
    k_eff = min(k_neighbors, faces.shape[0])

    rasterizer = MeshRasterizer(cameras=renderer.rasterizer.cameras, raster_settings=raster_settings)
    coarse_rasterizer = MeshRasterizer(cameras=renderer.rasterizer.cameras, raster_settings=coarse_raster_settings)

    face_adjacency = get_face_adjacency(faces, verts)
    colors = get_colors_from_mesh(mesh, colored_mesh)

    fps_idx, knn_idx = sample_faces_fps_knn_diverse(
        faces=faces,
        verts=verts,
        face_normals=face_normals,
        colors=colors,
        face_adjacency=face_adjacency,
        n_faces=n_eff,
        k_neighbors=k_eff,
        saliency_keep_ratio=saliency_keep_ratio,
        overlap_thresh=overlap_thresh,
        oversample_factor=oversample_factor,
    )

    face_centers, face_normals_unit = get_face_centers_and_normals(faces, verts, face_normals)

    tex_list, depth_list, normal_list, valid_list = [], [], [], []
    camera_R_list, camera_T_list = [], []

    total_retries_all = 0

    for fi, face_idx in enumerate(fps_idx):
        group_face_ids = torch.unique(torch.cat([face_idx.view(1), knn_idx[fi]], dim=0))

        patches, depths, normals, valid, total_retries, camera_R, camera_T = build_face_texture_with_camera(
            mesh=mesh,
            face_idx=int(face_idx.item()),
            target_face_ids=group_face_ids,
            renderer=renderer,
            rasterizer=rasterizer,
            coarse_rasterizer=coarse_rasterizer,
            p_views=p_views,
            h=patch_h,
            w=patch_w,
            dist_range=dist_range,
            jitter_angle=jitter_angle,
            max_retry=max_retry,
            min_target_pixels=min_target_pixels,
        )

        tex_list.append(patches)
        depth_list.append(depths)
        normal_list.append(normals)
        valid_list.append(valid)
        camera_R_list.append(camera_R)
        camera_T_list.append(camera_T)
        total_retries_all += total_retries

        if (fi + 1) % 4 == 0 or (fi + 1) == len(fps_idx):
            cur_valid = torch.stack(valid_list, dim=0).float().mean().item()
            log_info(
                "Reference progress: %s | sampled_faces=%d/%d | retries=%d | valid_ratio=%.4f",
                mesh_path, fi + 1, len(fps_idx), total_retries_all, cur_valid
            )

    tex_tensor = torch.stack(tex_list, dim=0)
    depth_tensor = torch.stack(depth_list, dim=0)
    normal_tensor = torch.stack(normal_list, dim=0)
    tex_valid = torch.stack(valid_list, dim=0)

    camera_R = torch.stack(camera_R_list, dim=0)
    camera_T = torch.stack(camera_T_list, dim=0)

    all_geo_face_idx = collect_geo_indices(fps_idx, knn_idx)
    geo_flat = get_face_attributes(all_geo_face_idx, faces, verts, face_normals, face_adjacency, colors)
    geo_tensor = geo_flat.reshape(n_eff, k_eff + 1, 33)

    ref_anchor_centers = face_centers[fps_idx]
    ref_anchor_normals = face_normals_unit[fps_idx]

    meta: Dict[str, str] = {
        "mesh_path": mesh_path,
        "mode": "reference_diverse_mv",
        "n_faces": str(n_eff),
        "k_neighbors": str(k_eff),
        "p_views": str(p_views),
        "patch_h": str(patch_h),
        "patch_w": str(patch_w),
        "dist_min": str(dist_range[0]),
        "dist_max": str(dist_range[1]),
        "jitter_angle": str(jitter_angle),
        "min_target_pixels": str(min_target_pixels),
        "saliency_keep_ratio": str(saliency_keep_ratio),
        "overlap_thresh": str(overlap_thresh),
        "oversample_factor": str(oversample_factor),
    }

    np.savez_compressed(
        npz_path,
        geo=geo_tensor.cpu().numpy().astype(np.float32),
        tex=(tex_tensor.clamp(0.0, 1.0) * 255).byte().cpu().numpy(),
        depth=(depth_tensor.clamp(0.0, 1.0) * 255).byte().cpu().numpy(),
        normal=(normal_tensor.clamp(0.0, 1.0) * 255).byte().cpu().numpy(),
        tex_valid=tex_valid.cpu().numpy(),

        fps_idx=fps_idx.cpu().numpy(),
        knn_idx=knn_idx.cpu().numpy(),

        ref_anchor_centers=ref_anchor_centers.cpu().numpy().astype(np.float32),
        ref_anchor_normals=ref_anchor_normals.cpu().numpy().astype(np.float32),

        camera_R=camera_R.cpu().numpy().astype(np.float32),
        camera_T=camera_T.cpu().numpy().astype(np.float32),

        face_centers=face_centers.cpu().numpy().astype(np.float32),
        face_normals=face_normals_unit.cpu().numpy().astype(np.float32),

        meta=np.array([f"{k}={v}" for k, v in meta.items()], dtype=object),
    )

    log_info("Saved reference %s | retries=%d | valid_ratio=%.4f", npz_path, total_retries_all, tex_valid.float().mean().item())
    return npz_path


# =========================================================
# Distortion preprocess
# =========================================================
def preprocess_distortion_mesh_aligned_fast(
    mesh_path: str,
    ref_npz_path: str,
    device: torch.device,
    renderer,
    raster_settings: RasterizationSettings,
    dataset_name: str,
    output_replace: str,
    colored_mesh: bool,
    cover_exist: bool,
    alpha_center: float = 1.0,
    alpha_normal: float = 0.2,
    fallback_retry: int = 1,
):
    npz_path = mesh_path.replace(dataset_name, output_replace).replace(".obj", ".npz")
    save_dir = os.path.dirname(npz_path)
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(npz_path) and (not cover_exist):
        log_info("Skip existing distortion %s", npz_path)
        return npz_path

    ref_pack = np.load(ref_npz_path, allow_pickle=True)

    ref_fps_idx = torch.from_numpy(ref_pack["fps_idx"]).to(device=device, dtype=torch.long)
    ref_knn_idx = torch.from_numpy(ref_pack["knn_idx"]).to(device=device, dtype=torch.long)
    ref_anchor_centers = torch.from_numpy(ref_pack["ref_anchor_centers"]).to(device=device, dtype=torch.float32)
    ref_anchor_normals = torch.from_numpy(ref_pack["ref_anchor_normals"]).to(device=device, dtype=torch.float32)
    camera_R = torch.from_numpy(ref_pack["camera_R"]).to(device=device, dtype=torch.float32)
    camera_T = torch.from_numpy(ref_pack["camera_T"]).to(device=device, dtype=torch.float32)

    mesh = load_objs_as_meshes([mesh_path], device=device, vcmesh=colored_mesh)
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    face_normals = mesh.faces_normals_packed()

    rasterizer = MeshRasterizer(cameras=renderer.rasterizer.cameras, raster_settings=raster_settings)

    n_eff = ref_fps_idx.shape[0]
    k_eff = ref_knn_idx.shape[1]

    face_adjacency = get_face_adjacency(faces, verts)
    colors = get_colors_from_mesh(mesh, colored_mesh)
    dist_face_centers, dist_face_normals = get_face_centers_and_normals(faces, verts, face_normals)

    matched_anchor_idx = match_ref_anchors_to_dist_faces(
        ref_anchor_centers=ref_anchor_centers,
        ref_anchor_normals=ref_anchor_normals,
        dist_face_centers=dist_face_centers,
        dist_face_normals=dist_face_normals,
        alpha_center=alpha_center,
        alpha_normal=alpha_normal,
    )

    dist_knn_idx = build_knn_groups_from_anchor_faces(
        anchor_face_idx=matched_anchor_idx,
        face_centers=dist_face_centers,
        k_neighbors=k_eff,
    )

    tex_list, depth_list, normal_list, valid_list = [], [], [], []

    patch_h = int(ref_pack["tex"].shape[2])
    patch_w = int(ref_pack["tex"].shape[3])

    for fi in range(n_eff):
        group_face_ids = torch.unique(torch.cat([matched_anchor_idx[fi].view(1), dist_knn_idx[fi]], dim=0))

        patches, depths, normals, valid = render_anchor_patches_with_given_cameras(
            mesh=mesh,
            renderer=renderer,
            rasterizer=rasterizer,
            target_face_ids=group_face_ids,
            R_all=camera_R[fi],
            T_all=camera_T[fi],
            h=patch_h,
            w=patch_w,
            min_target_pixels=1,
        )

        if (~valid).any() and fallback_retry > 0:
            anchor_center = dist_face_centers[matched_anchor_idx[fi]]
            anchor_normal = dist_face_normals[matched_anchor_idx[fi]]

            mesh_radius = (verts - verts.mean(dim=0, keepdim=True)).norm(dim=1).max().item()
            mesh_radius = max(mesh_radius, 1e-6)
            eff_dist_range = (0.6 * mesh_radius, 1.4 * mesh_radius)

            canonical_dirs = generate_canonical_view_dirs(
                face_normal=anchor_normal,
                num_views=patches.shape[0],
                spread=0.55,
                use_backside=False,
            )

            for vi in torch.where(~valid)[0].tolist():
                best_patch = None
                best_depth = None
                best_normal = None
                best_ok = False
                best_pixels = -1

                base_dir = canonical_dirs[min(vi, len(canonical_dirs) - 1)]

                for retry_idx in range(fallback_retry):
                    cur_dir = perturb_view_dir(base_dir, noise_scale=0.08 + 0.03 * retry_idx)
                    r, t = sample_camera_from_view_dir(anchor_center, cur_dir, eff_dist_range)

                    patch, depth_patch, normal_patch, ok, pixels = render_face_patch(
                        mesh=mesh,
                        renderer=renderer,
                        rasterizer=rasterizer,
                        target_face_ids=group_face_ids,
                        r=r,
                        t=t,
                        h=patch_h,
                        w=patch_w,
                        min_target_pixels=1,
                    )
                    if pixels > best_pixels and patch is not None:
                        best_pixels = pixels
                        best_patch = patch
                        best_depth = depth_patch
                        best_normal = normal_patch
                        best_ok = ok

                if best_patch is not None:
                    patches[vi] = best_patch
                    depths[vi] = best_depth
                    normals[vi] = best_normal
                    valid[vi] = best_ok

        tex_list.append(patches)
        depth_list.append(depths)
        normal_list.append(normals)
        valid_list.append(valid)

        if (fi + 1) % 4 == 0 or (fi + 1) == n_eff:
            cur_valid = torch.stack(valid_list, dim=0).float().mean().item()
            log_info(
                "Distortion progress: %s | aligned_faces=%d/%d | valid_ratio=%.4f",
                mesh_path, fi + 1, n_eff, cur_valid
            )

    tex_tensor = torch.stack(tex_list, dim=0)
    depth_tensor = torch.stack(depth_list, dim=0)
    normal_tensor = torch.stack(normal_list, dim=0)
    tex_valid = torch.stack(valid_list, dim=0)

    all_geo_face_idx = collect_geo_indices(matched_anchor_idx, dist_knn_idx)
    geo_flat = get_face_attributes(all_geo_face_idx, faces, verts, face_normals, face_adjacency, colors)
    geo_tensor = geo_flat.reshape(n_eff, k_eff + 1, 33)

    meta: Dict[str, str] = {
        "mesh_path": mesh_path,
        "mode": "distortion_aligned_fast_mv",
        "ref_npz": ref_npz_path,
        "n_faces": str(n_eff),
        "k_neighbors": str(k_eff),
        "alpha_center": str(alpha_center),
        "alpha_normal": str(alpha_normal),
    }

    np.savez_compressed(
        npz_path,
        geo=geo_tensor.cpu().numpy().astype(np.float32),
        tex=(tex_tensor.clamp(0.0, 1.0) * 255).byte().cpu().numpy(),
        depth=(depth_tensor.clamp(0.0, 1.0) * 255).byte().cpu().numpy(),
        normal=(normal_tensor.clamp(0.0, 1.0) * 255).byte().cpu().numpy(),
        tex_valid=tex_valid.cpu().numpy(),

        matched_anchor_idx=matched_anchor_idx.cpu().numpy(),
        knn_idx=dist_knn_idx.cpu().numpy(),

        ref_anchor_centers=ref_anchor_centers.cpu().numpy().astype(np.float32),
        ref_anchor_normals=ref_anchor_normals.cpu().numpy().astype(np.float32),

        camera_R=camera_R.cpu().numpy().astype(np.float32),
        camera_T=camera_T.cpu().numpy().astype(np.float32),

        meta=np.array([f"{k}={v}" for k, v in meta.items()], dtype=object),
    )

    log_info("Saved distortion %s | valid_ratio=%.4f", npz_path, tex_valid.float().mean().item())
    return npz_path


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Aligned FR preprocessing with diverse anchor sampling and canonical multi-view")

    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cover_exist", action="store_true")
    parser.add_argument("--output_replace", type=str, required=True)
    parser.add_argument("--debug_one", action="store_true")

    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--P", type=int, default=4)
    parser.add_argument("--H", type=int, default=112)
    parser.add_argument("--W", type=int, default=112)

    parser.add_argument("--dist_min", type=float, default=0.6)
    parser.add_argument("--dist_max", type=float, default=1.4)
    parser.add_argument("--jitter_angle", type=float, default=18.0)
    parser.add_argument("--max_retry", type=int, default=14)
    parser.add_argument("--min_target_pixels", type=int, default=20)
    parser.add_argument("--render_size", type=int, default=512)
    parser.add_argument("--coarse_render_size", type=int, default=56)

    parser.add_argument("--alpha_center", type=float, default=1.0)
    parser.add_argument("--alpha_normal", type=float, default=0.2)
    parser.add_argument("--fallback_retry", type=int, default=1)

    parser.add_argument("--saliency_keep_ratio", type=float, default=0.5)
    parser.add_argument("--overlap_thresh", type=float, default=0.35)
    parser.add_argument("--oversample_factor", type=int, default=4)

    args = parser.parse_args()

    device = torch.device(args.device)

    raster_settings = RasterizationSettings(image_size=args.render_size, bin_size=0, max_faces_per_bin=None)
    coarse_raster_settings = RasterizationSettings(image_size=args.coarse_render_size, bin_size=0, max_faces_per_bin=None)
    rasterizer = MeshRasterizer(raster_settings=raster_settings)

    blend_params = BlendParams(background_color=torch.ones(3))
    lights = AmbientLights(ambient_color=((1.0, 1.0, 1.0),), device=device)
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(device=device, lights=lights, blend_params=blend_params),
    )

    colored_mesh = args.dataset_name == "VCMesh"
    dist_range = (args.dist_min, args.dist_max)

    obj_files = []
    for subdir, _, files in os.walk(args.root_dir):
        for file_name in files:
            if file_name.lower().endswith(".obj"):
                obj_files.append(os.path.join(subdir, file_name))
    obj_files.sort()

    ref_done = set()

    for idx, mesh_path in enumerate(tqdm(obj_files)):
        # if 'waterBuffalo' not in mesh_path:
        #     continue
        try:
            if is_reference_obj(mesh_path, args.dataset_name):
                if os.path.normpath(mesh_path) not in ref_done:
                    preprocess_reference_mesh_diverse(
                        mesh_path=mesh_path,
                        device=device,
                        renderer=renderer,
                        raster_settings=raster_settings,
                        coarse_raster_settings=coarse_raster_settings,
                        dataset_name=args.dataset_name,
                        output_replace=args.output_replace,
                        n_faces=args.N,
                        k_neighbors=args.K,
                        p_views=args.P,
                        patch_h=args.H,
                        patch_w=args.W,
                        colored_mesh=colored_mesh,
                        dist_range=dist_range,
                        jitter_angle=args.jitter_angle,
                        max_retry=args.max_retry,
                        min_target_pixels=args.min_target_pixels,
                        cover_exist=args.cover_exist,
                        saliency_keep_ratio=args.saliency_keep_ratio,
                        overlap_thresh=args.overlap_thresh,
                        oversample_factor=args.oversample_factor,
                    )
                    ref_done.add(os.path.normpath(mesh_path))

            else:
                ref_obj_path = guess_reference_obj_for_dist(mesh_path, args.dataset_name)
                if ref_obj_path is None or (not os.path.exists(ref_obj_path)):
                    log_info("Cannot infer reference for %s", mesh_path)
                    continue

                ref_npz_path = ref_obj_path.replace(args.dataset_name, args.output_replace).replace(".obj", ".npz")

                if (not os.path.exists(ref_npz_path)) and (os.path.normpath(ref_obj_path) not in ref_done):
                    preprocess_reference_mesh_diverse(
                        mesh_path=ref_obj_path,
                        device=device,
                        renderer=renderer,
                        raster_settings=raster_settings,
                        coarse_raster_settings=coarse_raster_settings,
                        dataset_name=args.dataset_name,
                        output_replace=args.output_replace,
                        n_faces=args.N,
                        k_neighbors=args.K,
                        p_views=args.P,
                        patch_h=args.H,
                        patch_w=args.W,
                        colored_mesh=colored_mesh,
                        dist_range=dist_range,
                        jitter_angle=args.jitter_angle,
                        max_retry=args.max_retry,
                        min_target_pixels=args.min_target_pixels,
                        cover_exist=args.cover_exist,
                        saliency_keep_ratio=args.saliency_keep_ratio,
                        overlap_thresh=args.overlap_thresh,
                        oversample_factor=args.oversample_factor,
                    )
                    ref_done.add(os.path.normpath(ref_obj_path))

                preprocess_distortion_mesh_aligned_fast(
                    mesh_path=mesh_path,
                    ref_npz_path=ref_npz_path,
                    device=device,
                    renderer=renderer,
                    raster_settings=raster_settings,
                    dataset_name=args.dataset_name,
                    output_replace=args.output_replace,
                    colored_mesh=colored_mesh,
                    cover_exist=args.cover_exist,
                    alpha_center=args.alpha_center,
                    alpha_normal=args.alpha_normal,
                    fallback_retry=args.fallback_retry,
                )

            if args.debug_one:
                break

        except Exception as exc:
            logging.exception("Failed on %s: %s", mesh_path, str(exc))
            if args.debug_one:
                break


if __name__ == "__main__":
    main()
