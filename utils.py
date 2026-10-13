import os
from pathlib import Path
from typing import Optional,Dict,List,Tuple

import numpy as np
import torch
from iopath.common.file_io import PathManager
from pytorch3d.common.datatypes import Device
from pytorch3d.io.mtl_io import make_mesh_texture_atlas
from pytorch3d.io.obj_io import (_Aux, _Faces, _format_faces_indices,
                                 _load_materials, _parse_face)
from pytorch3d.io.utils import _make_tensor, _open_file
from pytorch3d.renderer import TexturesAtlas, TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes, join_meshes_as_batch

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, kendalltau

# NOTE: This script rewrites some functions of the obj_io.py script from pytorch3d to:
#   1. Add support for loading vertex-color meshes (vcmesh), which is not supported by pytorch3d 0.7.5.
#   2. Add support for loading and saving tensors of obj files (for better loading efficiency).


def _parse_obj(f, data_dir: str, vcmesh: bool = False):
    """
    Extended version of PyTorch3D's `_parse_obj` that optionally supports
    loading per-vertex colors in addition to standard OBJ content.

    For detailed documentation about the returned values and OBJ formatting,
    refer to the official PyTorch3D `obj_io.py::_parse_obj` function.

    Args:
        vcmesh (bool, optional): If True, parse vertex colors (R, G, B) in addition
            to vertex positions. Defaults to False.
    """
    verts, normals, verts_uvs = [], [], []
    faces_verts_idx, faces_normals_idx, faces_textures_idx = [], [], []
    faces_materials_idx = []
    material_names = []
    mtl_path = None
    verts_color = [] if vcmesh else None

    lines = [line.strip() for line in f]

    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    materials_idx = -1

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("mtllib"):
            if len(tokens) < 2:
                raise ValueError("material file name is not specified")
            # NOTE: only allow one .mtl file per .obj.
            # Definitions for multiple materials can be included
            # in this one .mtl file.
            mtl_path = line[len(tokens[0]):].strip()  # Take the remainder of the line
            mtl_path = os.path.join(data_dir, mtl_path)
        elif len(tokens) and tokens[0] == "usemtl":
            material_name = tokens[1]
            # materials are often repeated for different parts
            # of a mesh.
            if material_name not in material_names:
                material_names.append(material_name)
                materials_idx = len(material_names) - 1
            else:
                materials_idx = material_names.index(material_name)
        elif line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
            if vcmesh:
                vert_col = [float(x) for x in tokens[4:7]]
                if len(vert_col) != 3:
                    msg = "Vertex color %s does not have 3 values. Line: %s"
                    raise ValueError(msg % (str(vert), str(line)))
                verts_color.append(vert_col)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            verts_uvs.append(tx)
        elif line.startswith("vn "):  # Line is a normal.
            try:
                norm = [float(x) for x in tokens[1:4]]
                if len(norm) != 3:
                    msg = "Normal %s does not have 3 values. Line: %s"
                    raise ValueError(msg % (str(norm), str(line)))
                normals.append(norm)
            except ValueError:
                normals.append([0.0, 0.0, 1.0])
            # if len(norm) != 3:
            #     msg = "Normal %s does not have 3 values. Line: %s"
            #     raise ValueError(msg % (str(norm), str(line)))
            # normals.append(norm)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            _parse_face(
                line,
                tokens,
                materials_idx,
                faces_verts_idx,
                faces_normals_idx,
                faces_textures_idx,
                faces_materials_idx,
            )

    return (
        verts,
        verts_color,
        normals,
        verts_uvs,
        faces_verts_idx,
        faces_normals_idx,
        faces_textures_idx,
        faces_materials_idx,
        material_names,
        mtl_path,
    )


def _load_obj_parse(
    f_obj,
    *,
    data_dir: str,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    path_manager: PathManager,
    device: Device = "cpu",
    load_tensor: bool = False,
    save_tensor: bool = False,
    vcmesh: bool = False,
):
    """
    Extended version of PyTorch3D's `_load_obj` that supports loading and saving
    object tensors, as well as loading vertex-colored meshes.

    For full documentation on core functionality, refer to:
        `pytorch3d.io.obj_io._load_obj`

    Args:
        load_tensor (bool, optional): If True, load mesh data from a pre-saved `.pt` tensor file
            instead of parsing the `.obj` file. Defaults to False.
        save_tensor (bool. optional): If True, saves the parsed result as a `.pt` file to enable
            faster loading next time. Defaults to False.
        vcmesh (bool, optional): If True, enables support for vertex-colored meshes by parsing
            per-vertex RGB colors. Defaults to False.
    """

    if texture_wrap is not None and texture_wrap not in ["repeat", "clamp"]:
        msg = "texture_wrap must be one of ['repeat', 'clamp'] or None, got %s"
        raise ValueError(msg % texture_wrap)

    if not load_tensor:
        (
            verts,
            verts_color,
            normals,
            verts_uvs,
            faces_verts_idx,
            faces_normals_idx,
            faces_textures_idx,
            faces_materials_idx,
            material_names,
            mtl_path,
        ) = _parse_obj(f_obj, data_dir, vcmesh=vcmesh)

        verts = _make_tensor(verts, cols=3, dtype=torch.float32, device=device)  # (V, 3)
        verts_color = _make_tensor(verts_color, cols=3, dtype=torch.float32, device=device) if vcmesh else None
        normals = _make_tensor(
            normals,
            cols=3,
            dtype=torch.float32,
            device=device,
        )  # (N, 3)
        verts_uvs = _make_tensor(
            verts_uvs,
            cols=2,
            dtype=torch.float32,
            device=device,
        )  # (T, 2)

        faces_verts_idx = _format_faces_indices(
            faces_verts_idx, verts.shape[0], device=device
        )

        # Repeat for normals and textures if present.
        if len(faces_normals_idx):
            faces_normals_idx = _format_faces_indices(
                faces_normals_idx, normals.shape[0], device=device, pad_value=-1
            )
        if len(faces_textures_idx):
            faces_textures_idx = _format_faces_indices(
                faces_textures_idx, verts_uvs.shape[0], device=device, pad_value=-1
            )
        if len(faces_materials_idx):
            faces_materials_idx = torch.tensor(
                faces_materials_idx, dtype=torch.int64, device=device
            )
    else:
        load_path = os.path.splitext(f_obj.name)[0] + '.pt'
        obj_pt = torch.load(load_path)
        verts = obj_pt['verts']
        verts_color = obj_pt['verts_color'] if vcmesh else None
        normals = obj_pt['normals']
        verts_uvs = obj_pt['verts_uvs']
        faces_verts_idx = obj_pt['faces_verts_idx']
        faces_normals_idx = obj_pt['faces_normals_idx']
        faces_textures_idx = obj_pt['faces_textures_idx']
        faces_materials_idx = obj_pt['faces_materials_idx']
        material_names = obj_pt['material_names']
        mtl_path = os.path.join(data_dir, './' + obj_pt['mtl_path'].split('/')[-1]) if not vcmesh else None

    if save_tensor:
        save_path = os.path.splitext(f_obj.name)[0] + '.pt'
        if not vcmesh:
            torch.save({'verts': verts, 'normals': normals, 'verts_uvs': verts_uvs, 'faces_verts_idx': faces_verts_idx,
                        'faces_normals_idx': faces_normals_idx, 'faces_textures_idx': faces_textures_idx,
                        'faces_materials_idx': faces_materials_idx, 'material_names': material_names,
                        'mtl_path': mtl_path}, save_path)
        else:
            torch.save({'verts': verts, 'verts_color': verts_color, 'normals': normals, 'verts_uvs': verts_uvs,
                        'faces_verts_idx': faces_verts_idx, 'faces_normals_idx': faces_normals_idx,
                        'faces_textures_idx': faces_textures_idx, 'faces_materials_idx': faces_materials_idx,
                        'material_names': material_names, 'mtl_path': mtl_path}, save_path)

    texture_atlas = None
    if not vcmesh:
        material_colors, texture_images = _load_materials(
            material_names,
            mtl_path,
            data_dir=data_dir,
            load_textures=load_textures,
            path_manager=path_manager,
            device=device,
        )
    else:
        material_colors, texture_images = None, None

    if material_colors and not material_names:
        # usemtl was not present but single material was present in the .mtl file
        material_names.append(next(iter(material_colors.keys())))
        # replace all -1 by 0 material idx
        if torch.is_tensor(faces_materials_idx):
            faces_materials_idx.clamp_(min=0)

    if create_texture_atlas:
        # Using the images and properties from the
        # material file make a per face texture map.

        # Create an array of strings of material names for each face.
        # If faces_materials_idx == -1 then that face doesn't have a material.
        idx = faces_materials_idx.cpu().numpy()
        face_material_names = np.array(material_names)[idx]  # (F,)
        face_material_names[idx == -1] = ""

        # Construct the atlas.
        texture_atlas = make_mesh_texture_atlas(
            material_colors,
            texture_images,
            face_material_names,
            faces_textures_idx,
            verts_uvs,
            texture_atlas_size,
            texture_wrap,
        )

    faces = _Faces(
        verts_idx=faces_verts_idx,
        normals_idx=faces_normals_idx,
        textures_idx=faces_textures_idx,
        materials_idx=faces_materials_idx,
    )
    aux = _Aux(
        normals=normals if len(normals) else None,
        verts_uvs=verts_uvs if len(verts_uvs) else None,
        material_colors=material_colors,
        texture_images=texture_images,
        texture_atlas=texture_atlas,
    )
    return verts, faces, aux, verts_color


def load_obj(
    f,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    device: Device = "cpu",
    path_manager: Optional[PathManager] = None,
    load_tensor: bool = False,
    save_tensor: bool = False,
    vcmesh: bool = False
):
    """
    Wrapper around PyTorch3D's `load_obj` function with added support for:
      - vertex-color meshes
      - loading/saving cached tensor representations

    For full documentation on standard arguments and return values,
    refer to: `pytorch3d.io.obj_io.load_obj`.

    Args:
        load_tensor (bool, optional): If True, loads mesh data from a pre-saved `.pt` file
            instead of parsing the `.obj` file. Defaults to False.
        save_tensor (bool, optional): If True, saves the parsed result into a `.pt` file
            for faster future loading. Defaults to False.
        vcmesh (bool, optional): If True, enables parsing of vertex colors
            (R, G, B) along with positions. Defaults to False.
    """
    data_dir = "./"
    if isinstance(f, (str, bytes, Path)):
        # pyre-fixme[6]: For 1st argument expected `PathLike[Variable[AnyStr <:
        #  [str, bytes]]]` but got `Union[Path, bytes, str]`.
        data_dir = os.path.dirname(f)
    if path_manager is None:
        path_manager = PathManager()
    with _open_file(f, path_manager, "r") as f:
        return _load_obj_parse(
            f,
            data_dir=data_dir,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
            device=device,
            load_tensor=load_tensor,
            save_tensor=save_tensor,
            vcmesh=vcmesh
        )


def load_objs_as_meshes(
    files: list,
    device: Optional[Device] = None,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    path_manager: Optional[PathManager] = None,
    load_tensor: bool = False,
    save_tensor: bool = False,
    normalize: bool = True,
    vcmesh: bool = False
):
    """
    Wrapper around PyTorch3D's `load_objs_as_meshes` that adds support for:
      - vertex-color meshes
      - loading from/saving to cached tensor (.pt) files

    For full documentation on mesh loading behavior, refer to:
        `pytorch3d.io.obj_io.load_objs_as_meshes`

    Args:
        load_tensor (bool, optional): If True, load mesh data from previously cached `.pt` files. Defaults to False.
        save_tensor (bool, optional): If True, save parsed mesh data to `.pt` files for reuse. Defaults to False.
        vcmesh (bool, optional): If True, supports loading vertex-color
            meshes using `TexturesVertex`. Defaults to False.
    """
    mesh_list = []
    for f_obj in files:
        verts, faces, aux, verts_color = load_obj(
            f_obj,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
            load_tensor=load_tensor,
            save_tensor=save_tensor,
            vcmesh=vcmesh
        )
        tex = None
        if create_texture_atlas:
            # TexturesAtlas type
            tex = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
        else:
            # TexturesUV type
            tex_maps = aux.texture_images
            if tex_maps is not None and len(tex_maps) > 0:
                verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
                faces_uvs = faces.textures_idx.to(device)  # (F, 3)
                image = list(tex_maps.values())[0].to(device)[None]
                tex = TexturesUV(
                    verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
                )
        if vcmesh:
            tex = TexturesVertex(verts_features=verts_color.unsqueeze(dim=0))
        if normalize:
            verts = verts - verts.mean(0)
            scale = max(verts.abs().max(0)[0])
            verts = verts / scale
        mesh = Meshes(
            verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex.to(device)
        )
        mesh_list.append(mesh)
    if len(mesh_list) == 1:
        return mesh_list[0]
    return join_meshes_as_batch(mesh_list)


def count_model_params(model: torch.nn.Module) -> str:
    """
    Generates a detailed summary of the number of trainable parameters in a model,
    broken down by submodules.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
        str: A formatted string summarizing parameter counts per submodule and the total count.
    """
    output_string = "\n"
    submodel_dict = {}
    output_string += (100 * "=") + "\n"
    output_string += (f"{model.__class__.__name__:<{60}} Num Params") + "\n"
    output_string += (100 * "=") + "\n"
    for name, param in model.named_parameters():
        if param.requires_grad:
            output_string += (f"{name:<{60}} {param.numel()}") + "\n"
            if name.split('.')[0] not in submodel_dict.keys():
                submodel_dict[name.split('.')[0]] = []
            submodel_dict[name.split('.')[0]].append(param.numel())
    output_string += (100 * "=") + "\n"
    total = 0
    for key, value in submodel_dict.items():
        output_string += (f"{key:<{60}} {sum(value)}") + "\n"
        total += sum(value)
    output_string += (100 * "=") + "\n"
    output_string += (f"{str('Total'):<{60}} {total}")

    return output_string


def accumulate_step_outputs(
    step_outputs: Dict,
    names_batched: List[str],
    pred_quality: torch.Tensor,
    sample_batched: Dict,
    reg_loss: torch.Tensor,
    rank_loss: torch.Tensor,
    loss: torch.Tensor
) -> Dict:
    """
    Accumulates step outputs (predictions, losses, targets) into a running dictionary
    during training or evaluation over a batch of samples.

    Args:
        step_outputs (Dict): Dictionary storing accumulated values.
        names_batched (List[str]): List of sample identifiers in the current batch.
        pred_quality (torch.Tensor): Predicted quality scores for the batch.
        sample_batched (Dict): Dictionary containing the ground-truth target scores.
        reg_loss (torch.Tensor): Regression loss for the batch.
        rank_loss (torch.Tensor): Ranking loss for the batch.
        loss (torch.Tensor): Total loss for the batch.

    Returns:
        Dict: Updated `step_outputs` dictionary with the current batch results appended.
    """
    step_outputs['names'] += names_batched
    step_outputs['preds'] = torch.cat((step_outputs['preds'], pred_quality), dim=0)
    step_outputs['targets'] = torch.cat((step_outputs['targets'], sample_batched['score'].to(step_outputs['targets'].device)), dim=0)
    step_outputs['reg_loss'] = torch.cat((step_outputs['reg_loss'], reg_loss.unsqueeze(0) * len(names_batched)), dim=0)
    step_outputs['rank_loss'] = torch.cat((step_outputs['rank_loss'],
                                           rank_loss.unsqueeze(0) * len(names_batched)), dim=0)
    step_outputs['loss'] = torch.cat((step_outputs['loss'], loss.unsqueeze(0) * len(names_batched)), dim=0)

    return step_outputs

class RankLoss(torch.nn.Module):
    """
    Rank-based loss function that penalizes incorrect ordering of prediction pairs.
    Given two sets of predictions and targets, the loss encourages preserving relative ranking.

    [Ref] Wei Sun, Xiongkuo Min, Wei Lu, and Guangtao Zhai. A deep learning based no-reference quality assessment model
    for ugc videos. In Proceedings of the 30th ACM International Conference on Multimedia.
    """

    def __init__(self, **kwargs):
        super(RankLoss, self).__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the pairwise ranking loss.

        Args:
            preds (torch.Tensor): Predicted scores of shape [N] or [N, 1].
            targets (torch.Tensor): Ground-truth scores of shape [N] or [N, 1].

        Returns:
            torch.Tensor: Scalar loss tensor representing the average pairwise ranking loss.
        """
        preds = preds.view(-1)
        targets = targets.view(-1)

        n = len(preds)
        preds = preds.unsqueeze(0).repeat(n, 1)
        preds_t = preds.t()
        targets = targets.unsqueeze(0).repeat(n, 1)
        targets_t = targets.t()
        masks = torch.sign(targets - targets_t)
        exclude_self = (torch.abs(targets - targets_t) > 0)
        rank_loss = exclude_self * torch.relu(torch.abs(targets - targets_t) - masks * (preds - preds_t))
        rank_loss = rank_loss.sum() / (exclude_self.sum() + 1e-08)
        return rank_loss

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic

def get_srcc_plcc(preds,targets):
    preds = preds.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    y_output_logistic = fit_function(preds,targets)
    srcc = spearmanr(y_output_logistic, preds)[0]
    plcc = pearsonr(y_output_logistic, preds)[0]
    return srcc, plcc





def plcc(pred: torch.Tensor, target: torch.Tensor, mapping: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the Pearson Linear Correlation Coefficient (PLCC) between predictions and targets.
    Optionally applies a 5-parameter logistic mapping before computing the correlation.

    Args:
        pred (torch.Tensor): Predicted quality scores (shape: [N]).
        target (torch.Tensor): Ground-truth quality scores (shape: [N]).
        mapping (bool, optional): If True, apply a logistic function to fit `pred` to `target`. Defaults to False.
        eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Scalar tensor representing the PLCC score.
    """
    if not mapping:
        pred = pred - torch.mean(pred)
        target = target - torch.mean(target)
        plcc = torch.sum(pred * target) / (torch.sqrt(torch.sum(pred ** 2)) * torch.sqrt(torch.sum(target ** 2)) + eps)
    else:
        x = pred.detach().cpu().numpy()
        y = target.detach().cpu().numpy()
        corr, _ = pearsonr(x, y)
        if corr > 0:
            c = np.mean(x)
            a = np.abs(np.max(y) - np.min(y))
            d = np.mean(y)
            b = 1 / np.std(x)
            e = 1
        else:
            c = np.mean(x)
            a = - np.abs(np.max(y) - np.min(y))
            d = np.mean(y)
            b = 1 / np.std(x)
            e = 1

        try:
            popt, _ = curve_fit(logistic_func_h, x, y, p0=[a, b, c, d, e])
            a = popt[0]
            b = popt[1]
            c = popt[2]
            d = popt[3]
            e = popt[4]
        except RuntimeError:
            pass

        x_log = logistic_func_h(x, a, b, c, d, e)
        plcc, _ = pearsonr(x_log, y)
        plcc = torch.tensor(plcc, device=pred.device)

    return plcc


def logistic_func_h(x: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    """
    Five-parameter logistic function used to fit prediction scores to target scores.

    Args:
        x (np.ndarray): Input array (typically prediction values).
        a (float): Amplitude coefficient.
        b (float): Slope coefficient.
        c (float): Inflection point (center).
        d (float): Offset/baseline value.
        e (float): Linear term coefficient.

    Returns:
        np.ndarray: Transformed output after applying the logistic function.
    """
    x_ = np.array(x)
    tmp = 0.5 - 1 / (1 + np.exp(b * (x_ - c)))
    y = a * tmp + d + e * x_
    return y


def rank_array(input_array: torch.Tensor) -> torch.Tensor:
    """
    Computes ranks of elements in the input tensor.

    Args:
        input_array (torch.Tensor): 1D tensor of values to be ranked.

    Returns:
        torch.Tensor: 1D tensor of the same shape containing rank indices.
    """
    sorted_indices = torch.argsort(input_array)
    ranks = torch.empty_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(len(input_array), device=input_array.device)

    return ranks


def srcc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the Spearman Rank Correlation Coefficient (SRCC) between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted scores (1D tensor or [N, 1]).
        target (torch.Tensor): Ground-truth scores (1D tensor or [N, 1]).

    Returns:
        torch.Tensor: Scalar tensor representing the SRCC score.
    """
    pred = pred.squeeze()
    target = target.squeeze()
    N = pred.size(0)

    pred_ranks = rank_array(pred)
    target_ranks = rank_array(target)

    num = 6 * torch.sum((pred_ranks - target_ranks) ** 2)
    denom = N * (N ** 2 - 1.0)
    srcc = 1 - num / denom

    return srcc


def get_srcc_plcc_h(preds,targets):
    s = srcc(preds,targets)
    p = plcc(preds,targets,mapping=True)
    return s,p

def patchify_and_generate_mask(
    projections: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    patch_size: int,
    background_intensity: float = 0.0,
    nonempty_ratio: float = 0.01,
    stride_ratio: float = 1.0
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Patchifies both rgb and feature projections, generates their bakground mask
    and discards empty patches.

    Args:
        projections (Tuple[torch.Tensor, torch.Tensor]): [rgb, feature] projections:
            rgb should be of shape [N, P, H, W, C1].
            feature should be of shape [N, P, H, W, C2].
        patch_size (int): size of patches
        background_intensity (float, optional): background intensity values. Defaults to 0.0.
        nonempty_ratio (float, optional): threshold ratio to consider a patch as nonempty. Defaults to 0.01.
        stride_ratio (float, optional): stride ratio for unfolding (patchifying) operation. Defaults to 1.0.

    Returns:
        Tuple[
            Tuple[torch.Tensor, torch.Tensor]],            # (patched RGB, patched features)
            torch.Tensor,                                  # non-empty patch indicator: [N*P, num_patches]
            torch.Tensor                                   # background mask: [num_nonempty_patches, Ps, Ps, 1]
        ]
    """
    # reshape projections
    N, P, C1, H, W = projections[0].shape
    rgb_proj = projections[0].view(N * P, C1, H, W) # output shape = [N*P, C1, H, W]
    dep_proj = projections[1].view(N * P, C1, H, W) # output shape = [N*P, C1, H, W]
    nor_proj = projections[0].view(N * P, C1, H, W) # output shape = [N*P, C1, H, W]

    # patchify projections
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=int(patch_size * stride_ratio))
    # output shape = [N*P, num_patches, C1, Ps, Ps]
    patch_rgb_proj = unfold(rgb_proj).view(N * P, C1, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
    patch_dep_proj = unfold(dep_proj).view(N * P, C1, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
    patch_nor_proj = unfold(nor_proj).view(N * P, C1, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
    num_patches = patch_rgb_proj.shape[1]

    # generate mask
    # output shape = [N*P, num_patches, Ps, Ps]
    ref_patch_rgb = patch_rgb_proj.view(N, P, num_patches, C1, patch_size, patch_size)[::2, :, :, :3, :, :]
    background_mask = ref_patch_rgb.any(dim=3) != background_intensity
    background_mask = torch.repeat_interleave(background_mask, repeats=2, dim=0)
    background_mask = background_mask.view(N*P, num_patches, patch_size, patch_size)

    # when less than nonempty_ratio is empty, consider the patch as empty, output shape = [N*P, num_patches]
    min_patches = 10
    background_mask_sum = torch.sum(background_mask, dim=(-1, -2)).view(N, P, num_patches)
    nonempty_patches = background_mask_sum >= ((patch_size ** 2) * nonempty_ratio)
    # find models that had less than "min_patches" valid patches
    model_indices = torch.nonzero(torch.sum(nonempty_patches, dim=(-1, -2)) < min_patches, as_tuple=True)[0]
    # select "min_patches" patches with the highest non-empty ratio for both reference and test models
    for idx in model_indices[::2]:
        patch_val, _ = torch.topk(background_mask_sum[idx].flatten(), k=min_patches)
        nonempty_patches[idx] = background_mask_sum[idx] > max(patch_val[-1] - 1, 0)  # refence model
        nonempty_patches[idx+1] = background_mask_sum[idx+1] > max(patch_val[-1] - 1, 0)  # test model
    nonempty_patches = nonempty_patches.view(N*P, num_patches)
    # output shape = [num_nonempty_patches, Ps, Ps, 1]
    background_mask = background_mask[nonempty_patches, :, :].unsqueeze(-1)

    # discard empty patches
    # output shape = [num_nonempty_patches, Ps, Ps, C1]
    patch_rgb_proj = patch_rgb_proj[nonempty_patches, :, :, :].permute(0, 2, 3, 1)
    patch_dep_proj = patch_dep_proj[nonempty_patches, :, :, :].permute(0, 2, 3, 1)
    patch_nor_proj = patch_nor_proj[nonempty_patches, :, :, :].permute(0, 2, 3, 1)
    # output shape = [num_nonempty_patches, Ps, Ps, C2]
    # patch_background_mask = background_mask[nonempty_patches, :, :]  # output shape = [num_nonempty_patches, Ps, Ps]

    return ((patch_rgb_proj, patch_dep_proj, patch_nor_proj), nonempty_patches, background_mask)


def pool_patch_features(
    N: int,
    num_proj: int,
    reg_feat: torch.Tensor,
    graph_feat: torch.Tensor,
    nonempty_patches: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    pools per-model regressor and graph features
    based on the non-empty patch indicator tensor.

    Args:
        N (int): number of models
        num_proj (int): number of projections (viewpoints) for each model
        reg_feat (torch.Tensor): regressor output features
            of shape [num_nonempty_patches, regressor_out_dim]
        graph_feat (torch.Tensor): graph pooled features
            of shape [num_nonempty_patches, graph_feat_out_dim]
        nonempty_patches (torch.Tensor): non-empty patch indicator tensor
            of shape [N*P, num_patches]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Pooled regressor features of shape [N, regressor_out_dim].
            - Pooled graph features of shape [N, graph_feat_out_dim].
    """
    # find patch index boundries for each model
    per_sample_patchmap = torch.sum(nonempty_patches.view(N, num_proj, nonempty_patches.shape[1]), dim=(-1, -2))
    patch_boundries = torch.cumsum(per_sample_patchmap, dim=0)

    # average patch features - first model
    pooled_reg_feat = torch.mean(reg_feat[0:patch_boundries[0], :], dim=0, keepdim=True)
    pooled_graph_feat = torch.mean(graph_feat[0:patch_boundries[0], :], dim=0, keepdim=True)
    # average patch features - other models
    for idx, up_bound in enumerate(patch_boundries[1:]):
        low_bound = patch_boundries[idx]
        model_reg_feat = torch.mean(reg_feat[low_bound:up_bound, :], dim=0, keepdim=True)
        pooled_reg_feat = torch.cat((pooled_reg_feat, model_reg_feat), dim=0)
        model_graph_feat = torch.mean(graph_feat[low_bound:up_bound, :], dim=0, keepdim=True)
        pooled_graph_feat = torch.cat((pooled_graph_feat, model_graph_feat), dim=0)

    return pooled_reg_feat, pooled_graph_feat


