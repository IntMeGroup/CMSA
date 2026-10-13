import os
from typing import Any, Dict, Optional, Type

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset


class TMQAMultiModalDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        normalize_score: bool = False,
        seed: Optional[int] = None,
        kfold_seed: Optional[int] = None,
        # geotex_dirname: str = "SJTU-TMQA-geotex",
        random_patch_shuffle: bool = True,
        patch_shuffle_prob: float = 1.0,
        view_dropout_prob: float = 0.0,
    ) -> None:
        self.root_dir = root_dir
        self.split = split
        self.normalize_score = normalize_score
        self.random_patch_shuffle = random_patch_shuffle
        self.patch_shuffle_prob = min(max(patch_shuffle_prob, 0.0), 1.0)
        self.view_dropout_prob = min(max(view_dropout_prob, 0.0), 1.0)
        self.dataset_name = os.path.basename(os.path.normpath(root_dir))
        # self.geotex_dirname = geotex_dirname

        self.ref_dir = os.path.join(root_dir, "reference_dataset")
        self.dist_dir = os.path.join(root_dir, "distortion_dataset")
        self.file = os.path.join(self.dist_dir, "SJTU-TMQA_MOS.csv")
        self.data = pd.read_csv(self.file)
        self.k = 8
        source_models = self.data["model"].unique()
        if kfold_seed is None:
            if seed is not None:
                rng = np.random.default_rng(seed)
                source_models = source_models.copy()
                rng.shuffle(source_models)
            split_idx = int(0.8 * len(source_models))
            source_splits = [source_models[:split_idx], source_models[split_idx:]]
        else:
            fold_index = 0 if seed is None else seed
            kfold = KFold(n_splits=5, shuffle=True, random_state=kfold_seed)
            source_splits = None
            for current_fold, (train_idx, test_idx) in enumerate(kfold.split(source_models)):
                if current_fold == fold_index:
                    source_splits = [source_models[train_idx], source_models[test_idx]]
                    break
            if source_splits is None:
                raise ValueError(f"Invalid fold index {fold_index}, expected range [0, 4].")

        target_sources = {"train": source_splits[0], "test": source_splits[1], "all": source_models}
        if split not in target_sources:
            raise ValueError(f"Unsupported split: {split}")
        self.data = self.data[self.data["model"].isin(target_sources[split])].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def set_epoch_k(self):
        k_min, k_max = 12, 28
        self.k = torch.randint(k_min, k_max + 1, (1,)).item()

    def _obj_to_npz_path(self, obj_path: str) -> str:
        # return obj_path.replace(self.dataset_name, self.geotex_dirname).replace(".obj", ".npz")
        return obj_path.replace(".obj", ".npz")


    def _resolve_dist_obj_path(self, sample_name: str) -> str:
        model, distortion, level = sample_name.split("_")
        if "ds" in distortion or "JPEG" in distortion:
            return os.path.join(self.dist_dir, distortion, model, level, model + ".obj")
        return os.path.join(self.dist_dir, distortion, model, sample_name + ".obj")

    def _load_npz(self, npz_path: str) -> Dict[str, torch.Tensor]:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing preprocessed file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        geometry = torch.from_numpy(data["geo"]).to(torch.float32)
        tex = torch.from_numpy(data["tex"]).to(torch.float32).permute(0, 1, 4, 2, 3) / 255.0
        depth = torch.from_numpy(data["depth"]).to(torch.float32).permute(0, 1, 4, 2, 3) / 255.0
        depth = depth.repeat(1, 1, 3, 1, 1)
        normal = torch.from_numpy(data["normal"]).to(torch.float32).permute(0, 1, 4, 2, 3) / 255.0

        return {
            "geometry": geometry,
            "tex": tex,
            "depth": depth,
            "normal": normal,
        }

    def _build_patch_order(self) -> torch.Tensor:
        order = torch.arange(4)
        # if self.split == "train" and self.random_patch_shuffle and torch.rand(1).item() < self.patch_shuffle_prob:
        #     order = order[torch.randperm(16)]
        if self.split == "train" and self.random_patch_shuffle:
            order = order[torch.randperm(4)]
        return order

    def _reorder_tex_valid(self, tex_valid: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        return tex_valid[:, order]

    def _apply_view_dropout(self, tex: torch.Tensor, depth: torch.Tensor, normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.split != "train" or self.view_dropout_prob <= 0.0:
            return tex, depth, normal

        # dropout_mask = torch.rand_like(tex_valid) < self.view_dropout_prob
        dropout_mask = torch.rand(*tex.shape[:2]) < self.view_dropout_prob
        keep_mask = ~dropout_mask
        keep_mask[:, 0] = True
        tex = tex * keep_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(tex.dtype)
        depth = depth * keep_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(depth.dtype)
        normal = normal * keep_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(normal.dtype)
        return tex, depth, normal

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        row = self.data.iloc[idx]
        sample_name = row["name"]
        score = torch.tensor(row["MOS"], dtype=torch.float32)

        dist_obj_path = self._resolve_dist_obj_path(sample_name)
        model_name = row["model"]
        ref_obj_path = os.path.join(self.ref_dir, model_name, model_name + ".obj")

        dist_data = self._load_npz(self._obj_to_npz_path(dist_obj_path))
        ref_data = self._load_npz(self._obj_to_npz_path(ref_obj_path))
        patch_order = self._build_patch_order()
        dist_tex, dist_depth, dist_normal = self._apply_view_dropout(dist_data["tex"][:, patch_order], dist_data["depth"][:, patch_order], dist_data["normal"][:, patch_order])
        ref_tex, ref_depth, ref_normal = self._apply_view_dropout(ref_data["tex"][:, patch_order], ref_data["depth"][:, patch_order], ref_data["normal"][:, patch_order])

        dist_geometry = dist_data["geometry"]
        ref_geometry = ref_data["geometry"]
        if self.split == 'train':
            selected_indices = torch.randperm(32)[:self.k]
            dist_tex = dist_tex[selected_indices]
            dist_depth = dist_depth[selected_indices]
            dist_normal = dist_normal[selected_indices]
            # selected_indices = torch.randperm(32)[:16]
            dist_geometry = dist_geometry[selected_indices]

            # selected_indices = torch.randperm(32)[:self.k]
            ref_tex = ref_tex[selected_indices]
            ref_depth = ref_depth[selected_indices]
            ref_normal = ref_normal[selected_indices]
            # selected_indices = torch.randperm(32)[:16]
            ref_geometry = ref_geometry[selected_indices]
        else:
            dist_tex = dist_tex
            dist_depth = dist_depth
            dist_normal = dist_normal
            dist_geometry = dist_geometry

            ref_tex = ref_tex
            ref_depth = ref_depth
            ref_normal = ref_normal
            ref_geometry = ref_geometry
        if self.normalize_score:
            # score = 1.0 - (score / 10.0)
            score = score / 10.0

        return {
            "dist_mosaic": dist_tex,
            "dist_depth": dist_depth,
            "dist_normal": dist_normal,
            "dist_geometry": dist_geometry,
            "ref_mosaic": ref_tex,
            "ref_depth": ref_depth,
            "ref_normal": ref_normal,
            "ref_geometry": ref_geometry,
            "score": score,
            "name": dist_obj_path,
        }

class TSMDMultiModalDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        normalize_score: bool = False,
        seed: Optional[int] = None,
        kfold_seed: Optional[int] = None,
        random_patch_shuffle: bool = True,
        patch_shuffle_prob: float = 1.0,
        view_dropout_prob: float = 0.0,
    ) -> None:
        self.root_dir = root_dir
        self.split = split
        self.normalize_score = normalize_score
        self.random_patch_shuffle = random_patch_shuffle
        self.patch_shuffle_prob = min(max(patch_shuffle_prob, 0.0), 1.0)
        self.view_dropout_prob = min(max(view_dropout_prob, 0.0), 1.0)
        self.dataset_name = os.path.basename(os.path.normpath(root_dir))
        # self.geotex_dirname = geotex_dirname
        self.k  = 8 

        self.ref_dir = os.path.join(self.root_dir, 'Reference')
        self.dist_dir = os.path.join(self.root_dir, 'Distortion')
        self.file = os.path.join(self.root_dir, 'TSMD_MOS.xlsx')
        self.data = pd.read_excel(self.file)

        split_list = self.data['PVS'].str.split('_')
        self.data['source'] = split_list.apply(lambda x: '_'.join(x[:-4]))
        self.data['dec'] = split_list.apply(lambda x: x[-4])
        self.data['qp'] = split_list.apply(lambda x: x[-3])
        self.data['qt'] = split_list.apply(lambda x: x[-2])
        self.data['cqlevel'] = split_list.apply(lambda x: x[-1])

        # if kfold_seed is None:
        #     if seed is not None:
        #         rng = np.random.default_rng(seed)
        #         source_models = source_models.copy()
        #         rng.shuffle(source_models)
        #     split_idx = int(0.8 * len(source_models))
        #     source_splits = [source_models[:split_idx], source_models[split_idx:]]
        # else:
        #     fold_index = 0 if seed is None else seed
        #     kfold = KFold(n_splits=5, shuffle=True, random_state=kfold_seed)
        #     source_splits = None
        #     for current_fold, (train_idx, test_idx) in enumerate(kfold.split(source_models)):
        #         if current_fold == fold_index:
        #             source_splits = [source_models[train_idx], source_models[test_idx]]
        #             break
        #     if source_splits is None:
        #         raise ValueError(f"Invalid fold index {fold_index}, expected range [0, 4].")

        # target_sources = {"train": source_splits[0], "test": source_splits[1], "all": source_models}
        # if split not in target_sources:
        #     raise ValueError(f"Unsupported split: {split}")
        # self.data = self.data[self.data["model"].isin(target_sources[split])].reset_index(drop=True)

        source_models = self.data['source'].unique()
        # rejecting models whose reference is under license and not publicaly available
        rejection_list = ['thomas_fr00170', 'mitch_fr00001', 'nathalie_fr00036']
        source_models = np.array([x for x in source_models if x not in rejection_list])

        if kfold_seed is None:
            # randomly shuffle the source models with a given seed
            if seed is not None:
                np.random.seed(seed)
                np.random.shuffle(source_models)
            source_splits = [source_models[:int(0.8 * len(source_models))],
                             source_models[int(0.8 * len(source_models)):]]
        else:  # switch to kfold
            # if kfold_seed is not None, seed should be in range [0, 4]: fold0 --> seed = 0, ..., fold4 --> seed = 4
            fold_index = 0 if seed is None else seed
            kfold = KFold(n_splits=5, shuffle=True, random_state=kfold_seed)
            source_splits = None
            for current_fold, (train_idx, test_idx) in enumerate(kfold.split(source_models)):
                if current_fold == fold_index:
                    source_splits = [source_models[train_idx], source_models[test_idx]]
                    break
            if source_splits is None:
                raise ValueError(f"Invalid fold index {fold_index}, expected range [0, 4].")
        target_sources = {'train': source_splits[0], 'test': source_splits[1], 'all': source_models}
        if split not in target_sources:
            raise ValueError(f"Unsupported split: {split}")
        self.data = self.data[self.data['source'].isin(target_sources[split])].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def set_epoch_k(self):
        k_min, k_max = 12, 28
        self.k = torch.randint(k_min, k_max + 1, (1,)).item()

    def _obj_to_npz_path(self, obj_path: str) -> str:
        return obj_path.replace(".obj", ".npz")

    def _resolve_dist_obj_path(self, sample_name: str) -> str:
        model, distortion, level = sample_name.split("_")
        if "ds" in distortion or "JPEG" in distortion:
            return os.path.join(self.dist_dir, distortion, model, level, model + ".obj")
        return os.path.join(self.dist_dir, distortion, model, sample_name + ".obj")

    def _load_npz(self, npz_path: str) -> Dict[str, torch.Tensor]:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing preprocessed file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        geometry = torch.from_numpy(data["geo"]).to(torch.float32)
        tex = torch.from_numpy(data["tex"]).to(torch.float32).permute(0, 1, 4, 2, 3) / 255.0
        depth = torch.from_numpy(data["depth"]).to(torch.float32).permute(0, 1, 4, 2, 3) / 255.0
        depth = depth.repeat(1, 1, 3, 1, 1)
        normal = torch.from_numpy(data["normal"]).to(torch.float32).permute(0, 1, 4, 2, 3) / 255.0

        return {
            "geometry": geometry,
            "tex": tex,
            "depth": depth,
            "normal": normal,
        }

    def _build_patch_order(self) -> torch.Tensor:
        order = torch.arange(4, dtype=torch.long)
        # if self.split == "train" and self.random_patch_shuffle and torch.rand(1).item() < self.patch_shuffle_prob:
        #     order = order[torch.randperm(16)]
        if self.split == "train" and self.random_patch_shuffle:
            order = order[torch.randperm(4)]
        return order

    def _reorder_tex_valid(self, tex_valid: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        return tex_valid[:, order]

    def _apply_view_dropout(self, tex: torch.Tensor, depth: torch.Tensor, normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.split != "train" or self.view_dropout_prob <= 0.0:
            return tex, depth, normal

        # dropout_mask = torch.rand_like(tex_valid) < self.view_dropout_prob
        dropout_mask = torch.rand(*tex.shape[:2]) < self.view_dropout_prob
        keep_mask = ~dropout_mask
        keep_mask[:, 0] = True
        tex = tex * keep_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(tex.dtype)
        depth = depth * keep_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(depth.dtype)
        normal = normal * keep_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(normal.dtype)
        return tex, depth, normal

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if torch.is_tensor(idx):
            idx = int(idx.item())


        sample = self.data.iloc[idx]
        sample_name = sample['source']
        dist_filename = os.path.join(
            self.dist_dir,
            sample['source'],
            f'{sample["PVS"][:-2]}_{sample["PVS"][-2:]}'
        )
        dist_obj_path = dist_filename+'.obj'
        ref_filename = os.path.join(self.ref_dir, sample['source'], f'{sample["source"]}_C0-L5_deq_tri')
        ref_obj_path = ref_filename+'.obj'
        score = torch.tensor(sample['MOS'], dtype=torch.float32)

        dist_data = self._load_npz(self._obj_to_npz_path(dist_obj_path))
        ref_data = self._load_npz(self._obj_to_npz_path(ref_obj_path))
        patch_order = self._build_patch_order()
        dist_tex, dist_depth, dist_normal = self._apply_view_dropout(dist_data["tex"][:, patch_order], dist_data["depth"][:, patch_order], dist_data["normal"][:, patch_order])
        ref_tex, ref_depth, ref_normal = self._apply_view_dropout(ref_data["tex"][:, patch_order], ref_data["depth"][:, patch_order], ref_data["normal"][:, patch_order])

        dist_geometry = dist_data["geometry"]
        ref_geometry = ref_data["geometry"]
        if self.split == 'train':
            # selected_indices = torch.randperm(32)[:16]

            selected_indices = torch.randperm(32)[:self.k]
            dist_tex = dist_tex[selected_indices]
            dist_depth = dist_depth[selected_indices]
            dist_normal = dist_normal[selected_indices]
            dist_geometry = dist_geometry[selected_indices]

            # selected_indices = torch.randperm(32)[:16]
            selected_indices = torch.randperm(32)[:self.k]
            ref_tex = ref_tex[selected_indices]
            ref_depth = ref_depth[selected_indices]
            ref_normal = ref_normal[selected_indices]
            ref_geometry = ref_geometry[selected_indices]
        else:
            dist_tex = dist_tex
            dist_depth = dist_depth
            dist_normal = dist_normal
            dist_geometry = dist_geometry

            ref_tex = ref_tex
            ref_depth = ref_depth
            ref_normal = ref_normal
            ref_geometry = ref_geometry
        if self.normalize_score:
            # score = 1.0 - (score / 10.0)
            score = (score - 1.0) / 4.0

        return {
            "dist_mosaic": dist_tex,
            "dist_depth": dist_depth,
            "dist_normal": dist_normal,
            "dist_geometry": dist_geometry,
            "ref_mosaic": ref_tex,
            "ref_depth": ref_depth,
            "ref_normal": ref_normal,
            "ref_geometry": ref_geometry,
            "score": score,
            "name": dist_obj_path,
        }

class VCMeshMultiModalDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        normalize_score: bool = False,
        seed: Optional[int] = None,
        kfold_seed: Optional[int] = None,
        random_patch_shuffle: bool = True,
        patch_shuffle_prob: float = 1.0,
        view_dropout_prob: float = 0.0,
    ) -> None:
        self.root_dir = root_dir
        self.split = split
        self.normalize_score = normalize_score
        self.random_patch_shuffle = random_patch_shuffle
        self.patch_shuffle_prob = min(max(patch_shuffle_prob, 0.0), 1.0)
        self.view_dropout_prob = min(max(view_dropout_prob, 0.0), 1.0)
        self.dataset_name = os.path.basename(os.path.normpath(root_dir))

        self.k = k
        self.data = pd.read_csv(os.path.join(self.root_dir, 'Per_Stimulus_MOS_and_MLE.csv'))
        source_models = self.data['source'].unique()
        # train_split = np.delete(source_models, seed)
        # test_split = [source_models[seed]]  # seed should be an integer from 0 to 4. Chooses the test source model.
        # target_sources = {'train': train_split, 'test': test_split, 'all': source_models}
        # self.data = self.data[self.data['source'].isin(target_sources[split])]

        if kfold_seed is None:
            # randomly shuffle the source models with a given seed
            random_num = np.random.randint(0, 5)
            train_split = np.delete(source_models, random_num)
            test_split = [source_models[random_num]]  # seed should be an integer from 0 to 4. Chooses the test source model.
            target_sources = {'train': train_split, 'test': test_split, 'all': source_models}
        else:  # switch to kfold
            # if kfold_seed is not None, seed should be in range [0, 4]: fold0 --> seed = 0, ..., fold4 --> seed = 4
            fold_index = 0 if seed is None else seed
            train_split = np.delete(source_models, fold_index)
            test_split = [source_models[fold_index]]  # seed should be an integer from 0 to 4. Chooses the test source model.
            target_sources = {'train': train_split, 'test': test_split, 'all': source_models}
        if split not in target_sources:
            raise ValueError(f"Unsupported split: {split}")
        self.data = self.data[self.data['source'].isin(target_sources[split])].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def set_epoch_k(self):
        k_min, k_max = 12, 28
        self.k = torch.randint(k_min, k_max + 1, (1,)).item()


    def _obj_to_npz_path(self, obj_path: str) -> str:
        return obj_path.replace(".obj", ".npz")

    def _load_npz(self, npz_path: str) -> Dict[str, torch.Tensor]:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing preprocessed file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        geometry = torch.from_numpy(data["geo"]).to(torch.float32)
        tex = torch.from_numpy(data["tex"]).to(torch.float32).permute(0, 1, 4, 2, 3) / 255.0
        depth = torch.from_numpy(data["depth"]).to(torch.float32).permute(0, 1, 4, 2, 3) / 255.0
        depth = depth.repeat(1, 1, 3, 1, 1)
        normal = torch.from_numpy(data["normal"]).to(torch.float32).permute(0, 1, 4, 2, 3) / 255.0

        return {
            "geometry": geometry,
            "tex": tex,
            "depth": depth,
            "normal": normal,
        }

    def _build_patch_order(self) -> torch.Tensor:
        order = torch.arange(4, dtype=torch.long)
        # if self.split == "train" and self.random_patch_shuffle and torch.rand(1).item() < self.patch_shuffle_prob:
        #     order = order[torch.randperm(16)]
        if self.split == "train" and self.random_patch_shuffle:
            order = order[torch.randperm(4)]
        return order

    def _reorder_tex_valid(self, tex_valid: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        return tex_valid[:, order]

    def _apply_view_dropout(self, tex: torch.Tensor, depth: torch.Tensor, normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.split != "train" or self.view_dropout_prob <= 0.0:
            return tex, depth, normal

        # dropout_mask = torch.rand_like(tex_valid) < self.view_dropout_prob
        dropout_mask = torch.rand(*tex.shape[:2]) < self.view_dropout_prob
        keep_mask = ~dropout_mask
        keep_mask[:, 0] = True
        tex = tex * keep_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(tex.dtype)
        depth = depth * keep_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(depth.dtype)
        normal = normal * keep_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(normal.dtype)
        return tex, depth, normal

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if torch.is_tensor(idx):
            idx = int(idx.item())


        sample = self.data.iloc[idx]
        sample_name = sample['source']
        root_name = os.path.join(self.root_dir, 'Stimuli_Format_Obj', sample['source'])
        dist_obj_path = os.path.join(root_name, f'{sample["Stimuli"]}.obj')
        ref_obj_path = os.path.join(root_name, f'{sample["source"]}_Ref_0.obj')
        score = torch.tensor(sample['MOS'], dtype=torch.float32)

        dist_data = self._load_npz(self._obj_to_npz_path(dist_obj_path))
        ref_data = self._load_npz(self._obj_to_npz_path(ref_obj_path))
        patch_order = self._build_patch_order()
        dist_tex, dist_depth, dist_normal = self._apply_view_dropout(dist_data["tex"][:, patch_order], dist_data["depth"][:, patch_order], dist_data["normal"][:, patch_order])
        ref_tex, ref_depth, ref_normal = self._apply_view_dropout(ref_data["tex"][:, patch_order], ref_data["depth"][:, patch_order], ref_data["normal"][:, patch_order])

        dist_geometry = dist_data["geometry"]
        ref_geometry = ref_data["geometry"]
        if self.split == 'train':
            # selected_indices = torch.randperm(32)[:16]
            selected_indices = torch.randperm(32)[:self.k]
            dist_tex = dist_tex[selected_indices]
            dist_depth = dist_depth[selected_indices]
            dist_normal = dist_normal[selected_indices]
            dist_geometry = dist_geometry[selected_indices]

            # selected_indices = torch.randperm(32)[:16]
            selected_indices = torch.randperm(32)[:self.k]
            ref_tex = ref_tex[selected_indices]
            ref_depth = ref_depth[selected_indices]
            ref_normal = ref_normal[selected_indices]
            ref_geometry = ref_geometry[selected_indices]
        else:
            dist_tex = dist_tex
            dist_depth = dist_depth
            dist_normal = dist_normal
            dist_geometry = dist_geometry

            ref_tex = ref_tex
            ref_depth = ref_depth
            ref_normal = ref_normal
            ref_geometry = ref_geometry
        if self.normalize_score:
            score = (score -1.0 )/ 4.0

        return {
            "dist_mosaic": dist_tex,
            "dist_depth": dist_depth,
            "dist_normal": dist_normal,
            "dist_geometry": dist_geometry,
            "ref_mosaic": ref_tex,
            "ref_depth": ref_depth,
            "ref_normal": ref_normal,
            "ref_geometry": ref_geometry,
            "score": score,
            "name": dist_obj_path,
        }

class YN2023MultiModalDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        normalize_score: bool = False,
        seed: Optional[int] = None,
        kfold_seed: Optional[int] = None,
        random_patch_shuffle: bool = True,
        patch_shuffle_prob: float = 1.0,
        view_dropout_prob: float = 0.0,
    ) -> None:
        self.root_dir = root_dir
        self.split = split
        self.normalize_score = normalize_score
        self.random_patch_shuffle = random_patch_shuffle
        self.patch_shuffle_prob = min(max(patch_shuffle_prob, 0.0), 1.0)
        self.view_dropout_prob = min(max(view_dropout_prob, 0.0), 1.0)
        self.dataset_name = os.path.basename(os.path.normpath(root_dir))

        self.k = 8
        self.data = pd.read_csv(os.path.join(self.root_dir, 'MOS+CI_3000stimuli.csv'))
        split_list = self.data['stimulus'].str.split('_')
        self.data['source'] = split_list.apply(lambda x: '_'.join(x[:-6]))
        self.data['simp'] = split_list.apply(lambda x: x[-6])
        self.data['qp'] = split_list.apply(lambda x: x[-5])
        self.data['qt'] = split_list.apply(lambda x: x[-4])
        self.data['res'] = split_list.apply(lambda x: x[-2])
        self.data['JPEG'] = split_list.apply(lambda x: x[-1][1:])

        source_models = self.data['source'].unique()
        # Graph-LPIPS train/test split for random_state == 1
        if kfold_seed is None:
            source_splits = train_test_split(source_models, train_size=0.8, random_state=seed, shuffle=True)
        else:  # switch to kfold
            # if kfold_seed is not None, seed should be in range [0, 4]: fold0 --> seed = 0, ..., fold4 --> seed = 4
            kfold = KFold(n_splits=5, shuffle=True, random_state=kfold_seed)
            for fold_ind, (train_ind, test_ind) in enumerate(kfold.split(source_models)):
                if fold_ind == seed:
                    source_splits = [source_models[train_ind], source_models[test_ind]]
        target_sources = {'train': source_splits[0], 'test': source_splits[1], 'all': source_models}
        if split not in target_sources:
            raise ValueError(f"Unsupported split: {split}")
        self.data = self.data[self.data['source'].isin(target_sources[split])].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def set_epoch_k(self):
        k_min, k_max = 12, 28
        self.k = torch.randint(k_min, k_max + 1, (1,)).item()

    def _obj_to_npz_path(self, obj_path: str) -> str:
        return obj_path.replace(".obj", ".npz")

    def _load_npz(self, npz_path: str) -> Dict[str, torch.Tensor]:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing preprocessed file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        geometry = torch.from_numpy(data["geo"]).to(torch.float32)
        tex = torch.from_numpy(data["tex"]).to(torch.float32).permute(0, 1, 4, 2, 3) / 255.0
        depth = torch.from_numpy(data["depth"]).to(torch.float32).permute(0, 1, 4, 2, 3) / 255.0
        depth = depth.repeat(1, 1, 3, 1, 1)
        normal = torch.from_numpy(data["normal"]).to(torch.float32).permute(0, 1, 4, 2, 3) / 255.0

        return {
            "geometry": geometry,
            "tex": tex,
            "depth": depth,
            "normal": normal,
        }

    def _build_patch_order(self) -> torch.Tensor:
        order = torch.arange(4, dtype=torch.long)
        # if self.split == "train" and self.random_patch_shuffle and torch.rand(1).item() < self.patch_shuffle_prob:
        #     order = order[torch.randperm(16)]
        if self.split == "train" and self.random_patch_shuffle:
            order = order[torch.randperm(4)]
        return order

    def _reorder_tex_valid(self, tex_valid: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        return tex_valid[:, order]

    def _apply_view_dropout(self, tex: torch.Tensor, depth: torch.Tensor, normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.split != "train" or self.view_dropout_prob <= 0.0:
            return tex, depth, normal

        # dropout_mask = torch.rand_like(tex_valid) < self.view_dropout_prob
        dropout_mask = torch.rand(*tex.shape[:2]) < self.view_dropout_prob
        keep_mask = ~dropout_mask
        keep_mask[:, 0] = True
        tex = tex * keep_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(tex.dtype)
        depth = depth * keep_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(depth.dtype)
        normal = normal * keep_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(normal.dtype)
        return tex, depth, normal

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if torch.is_tensor(idx):
            idx = int(idx.item())


        sample = self.data.iloc[idx]
        sample_name = sample['source']
        dist_obj_path = os.path.join(
            self.root_dir,
            sample['source'],
            f'distortions/JPEG_resize{sample["res"]}_quality{sample["JPEG"]}',
            f'{sample["source"]}_{sample["simp"]}_{sample["qp"]}_{sample["qt"]}.obj'
        )
        ref_obj_path = os.path.join(self.root_dir, sample['source'], 'source', f'{sample["source"]}.obj')
        score = torch.tensor(sample['MOS'], dtype=torch.float32)

        dist_data = self._load_npz(self._obj_to_npz_path(dist_obj_path))
        ref_data = self._load_npz(self._obj_to_npz_path(ref_obj_path))
        patch_order = self._build_patch_order()
        dist_tex, dist_depth, dist_normal = self._apply_view_dropout(dist_data["tex"][:, patch_order], dist_data["depth"][:, patch_order], dist_data["normal"][:, patch_order])
        ref_tex, ref_depth, ref_normal = self._apply_view_dropout(ref_data["tex"][:, patch_order], ref_data["depth"][:, patch_order], ref_data["normal"][:, patch_order])

        dist_geometry = dist_data["geometry"]
        ref_geometry = ref_data["geometry"]
        if self.split == 'train':
            # selected_indices = torch.randperm(32)[:16]


            selected_indices = torch.randperm(32)[:self.k]
            dist_tex = dist_tex[selected_indices]
            dist_depth = dist_depth[selected_indices]
            dist_normal = dist_normal[selected_indices]
            dist_geometry = dist_geometry[selected_indices]

            # selected_indices = torch.randperm(32)[:16]
            selected_indices = torch.randperm(32)[:self.k]
            ref_tex = ref_tex[selected_indices]
            ref_depth = ref_depth[selected_indices]
            ref_normal = ref_normal[selected_indices]
            ref_geometry = ref_geometry[selected_indices]
        else:
            dist_tex = dist_tex
            dist_depth = dist_depth
            dist_normal = dist_normal
            dist_geometry = dist_geometry

            ref_tex = ref_tex
            ref_depth = ref_depth
            ref_normal = ref_normal
            ref_geometry = ref_geometry
        if self.normalize_score:
            score = (score - 1.0 )/ 4.0

        return {
            "dist_mosaic": dist_tex,
            "dist_depth": dist_depth,
            "dist_normal": dist_normal,
            "dist_geometry": dist_geometry,
            "ref_mosaic": ref_tex,
            "ref_depth": ref_depth,
            "ref_normal": ref_normal,
            "ref_geometry": ref_geometry,
            "score": score,
            "name": dist_obj_path,
        }


DATASET_REGISTRY: Dict[str, Type[Dataset]] = {
    "TMQA": TMQAMultiModalDataset,
    "TSMD": TSMDMultiModalDataset,
    "VCMesh": VCMeshMultiModalDataset,
    "YN2023": YN2023MultiModalDataset,
}


class DatasetBuilder:
    @staticmethod
    def build(name: str, **kwargs: Any) -> Dataset:
        dataset_cls = DATASET_REGISTRY.get(name)
        if dataset_cls is None:
            raise ValueError(f"Unsupported dataset name '{name}'. Available options: {list(DATASET_REGISTRY.keys())}")
        return dataset_cls(**kwargs)
