import sys, os
import math
import imageio

import torch
import torchvision
import numpy as np

import pytorch3d
import pytorch3d.datasets

import open3d as o3d

def get_2d_mgrid(shape):
    pixel_coords = np.stack(np.mgrid[:shape[0], :shape[1]], axis=-1).astype(np.float32)

    # normalize pixel coords onto [-1, 1]
    pixel_coords[..., 0] = pixel_coords[..., 0] / max(shape[0] - 1, 1)
    pixel_coords[..., 1] = pixel_coords[..., 1] / max(shape[1] - 1, 1)
    pixel_coords -= 0.5
    pixel_coords *= 2.
    # flatten 
    pixel_coords = torch.tensor(pixel_coords).view(-1, 2)

    return pixel_coords

def get_3d_mgrid(shape):
    pixel_coords = np.stack(np.mgrid[:shape[0], :shape[1], :shape[2]], axis=-1).astype(np.float32)

    # normalize pixel coords onto [-1, 1]
    pixel_coords[..., 0] = pixel_coords[..., 0] / max(shape[0] - 1, 1)
    pixel_coords[..., 1] = pixel_coords[..., 1] / max(shape[1] - 1, 1)
    pixel_coords[..., 2] = pixel_coords[..., 2] / max(shape[2] - 1, 1)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    # flatten 
    pixel_coords = torch.tensor(pixel_coords).view(-1, 3)

    return pixel_coords

class LatticeDataset(torch.utils.data.Dataset):

    def __init__(self, image_shape):
        super().__init__()

        if len(image_shape) == 2:
            self.mgrid = get_2d_mgrid(image_shape)
        elif len(image_shape) == 3:
            self.mgrid = get_3d_mgrid(image_shape)
        else:
            raise NotImplementedError(f'{len(image_shape)}-dimension lattice is not implemented.')

    def __len__(self):
        return self.mgrid.shape[0]
    
    def __getitem__(self, idx):
        return self.mgrid[idx], torch.tensor(idx, dtype=torch.int64)

class IterableLatticeDataset(torch.utils.data.IterableDataset, LatticeDataset):

    def __init__(self, image_shape, batch_size, shuffle=False):

        torch.utils.data.IterableDataset.__init__(self)
        LatticeDataset.__init__(self, image_shape)

        if len(image_shape) == 2:
            self.mgrid = get_2d_mgrid(image_shape)
        elif len(image_shape) == 3:
            self.mgrid = get_3d_mgrid(image_shape)
        else:
            raise NotImplementedError(f'{len(image_shape)}-dimension lattice is not implemented.')

        self.batch_size = batch_size
        self.randomize = shuffle

        self.perm = torch.arange(self.mgrid.shape[0])
        self.current_idx = self.perm.shape[0]

        # shuffle data
        # if self.randomize:
        #     perm = np.random.permutation(self.batch_size)
        #     self.mgrid = self.mgrid[perm]
        #     self.data = self.data[perm]

    def __iter__(self):
        self.current_idx = 0
        if self.randomize:
            self.perm = torch.randperm(self.mgrid.shape[0])
        return self

    def __next__(self):
        if self.current_idx >= self.perm.shape[0]:
            raise StopIteration()

        end = min(self.current_idx + self.batch_size, self.perm.shape[0])

        i_sel = self.perm[self.current_idx:end]
        coords = self.mgrid[i_sel]

        self.current_idx = end

        return coords, i_sel

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, pointcloud_path, on_surface_points, normalize=False, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        if normalize:
            coords -= np.mean(coords, axis=0, keepdims=True)
            if keep_aspect_ratio:
                coord_max = np.amax(coords)
                coord_min = np.amin(coords)
            else:
                coord_max = np.amax(coords, axis=0, keepdims=True)
                coord_min = np.amin(coords, axis=0, keepdims=True)

            self.coords = (coords - coord_min) / (coord_max - coord_min)
            self.coords -= 0.5
            self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __iter__(self):
        while True:
            point_cloud_size = self.coords.shape[0]

            off_surface_samples = self.on_surface_points  # **2
            total_samples = self.on_surface_points + off_surface_samples

            # Random coords
            rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

            on_surface_coords = self.coords[rand_idcs, :]
            on_surface_normals = self.normals[rand_idcs, :]

            off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))

            yield {
                'on_surface_coords': torch.from_numpy(on_surface_coords).float(),
                'on_surface_normals': torch.from_numpy(on_surface_normals).float(),
                'off_surface_coords': torch.from_numpy(off_surface_coords).float()
            }

class ShapeNetDatasetBase:

    def __init__(self, on_surface_points, normalize=False, keep_aspect_ratio=True):

        self.on_surface_points = on_surface_points
        self.normalize = normalize
        self.keep_aspect_ratio = keep_aspect_ratio

    # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
    # sample efficiency)
    def normalize_mesh(self, coords):
        coords -= np.mean(coords, axis=0, keepdims=True)
        if self.keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        coords = (coords - coord_min) / (coord_max - coord_min)
        coords -= 0.5
        coords *= 2.

        return coords

    def sample_points(self, mesh):
        pcd = mesh.sample_points_uniformly(self.on_surface_points, use_triangle_normal=True)

        on_surface_coords = np.asarray(pcd.points)
        on_surface_normals = np.asarray(pcd.normals)
        if self.normalize:
            on_surface_coords = self.normalize_mesh(on_surface_coords)

        # Random coords
        off_surface_samples = self.on_surface_points
        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))

        return {
            'on_surface_coords': torch.from_numpy(on_surface_coords).float(),
            'on_surface_normals': torch.from_numpy(on_surface_normals).float(),
            'off_surface_coords': torch.from_numpy(off_surface_coords).float(),
        }

class ShapeNetDataset(torch.utils.data.Dataset, ShapeNetDatasetBase):
    def __init__(self, data_dir, on_surface_points, normalize=False, category=None, subsample=None, keep_aspect_ratio=True):
        torch.utils.data.Dataset.__init__(self)
        ShapeNetDatasetBase.__init__(self, on_surface_points, normalize, keep_aspect_ratio)

        synsets = [category] if category is not None else None
        self.dataset = pytorch3d.datasets.ShapeNetCore(data_dir, synsets=synsets, version=1, load_textures=False)
        if subsample is not None:
            if isinstance(subsample, list):
                indices = subsample
            elif isinstance(subsample, int):
                indices = list(range(0, len(self.dataset), subsample))
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        verts = o3d.utility.Vector3dVector(data['verts'].numpy())
        faces = o3d.utility.Vector3iVector(data['faces'].numpy())

        mesh = o3d.geometry.TriangleMesh(verts, faces)
        ret_dict = self.sample_points(mesh)
        ret_dict['model_ids'] = torch.tensor(idx, dtype=torch.int64)

        return ret_dict

class SingleShapeNetDataset(torch.utils.data.IterableDataset, ShapeNetDatasetBase):
    def __init__(self, data_dir, model_id, on_surface_points, normalize=False, category=None, subsample=None, keep_aspect_ratio=True):
        torch.utils.data.IterableDataset.__init__(self)
        ShapeNetDatasetBase.__init__(self, on_surface_points, normalize, keep_aspect_ratio)

        synsets = [category] if category is not None else None
        self.dataset = pytorch3d.datasets.ShapeNetCore(data_dir, synsets=synsets, version=1, load_textures=False)
        if subsample is not None:
            if isinstance(subsample, list):
                indices = subsample
            elif isinstance(subsample, int):
                indices = list(range(0, len(self.dataset), subsample))
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

        data = self.dataset[model_id]
        verts = o3d.utility.Vector3dVector(data['verts'].numpy())
        faces = o3d.utility.Vector3iVector(data['faces'].numpy())
        self.mesh = o3d.geometry.TriangleMesh(verts, faces)

    def __iter__(self):
        while True:
            yield self.sample_points(self.mesh)

def get_r2n2_split_path(r2n2_dir, category, ensure=True):
    split_path = os.path.join(r2n2_dir, f'{category}_splits.json')
    if not os.path.exists(split_path) and ensure:
        with open(os.path.join(r2n2_dir, 'pix2mesh_splits_val05.json'), 'r') as f:
            split_dict = json.load(f)
        inv_synset_dict = {label: offset for offset, label in synset_dict.items()}

        offset = inv_synset_dict[category]
        category_splits = {}
        category_splits['train'] = {offset: split_dict['train'][offset]}
        category_splits['val'] = {offset: split_dict['val'][offset]}
        category_splits['test'] = {offset: split_dict['test'][offset]}
        with open(split_path, 'w') as f:
            json.dump(category_splits, f)
    return split_path

class R2N2Dataset(torch.utils.data.Dataset, ShapeNetDatasetBase):
    def __init__(self, shapenet_dir, r2n2_dir, on_surface_points, normalize=False, category=None, subsample=None, keep_aspect_ratio=True):
        torch.utils.data.Dataset.__init__(self)
        ShapeNetDatasetBase.__init__(self, on_surface_points, normalize, keep_aspect_ratio)

        split_path = get_r2n2_split_path(r2n2_dir, category, ensure=True)
        dataset = pytorch3d.datasets.R2N2('train', shapenet_dir, r2n2_dir, split_path,
            return_all_views=False, return_voxels=False, load_textures=False)
        if subsample is not None:
            if isinstance(subsample, list):
                indices = subsample
            elif isinstance(subsample, int):
                indices = list(range(0, len(self.dataset), subsample))
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        verts = o3d.utility.Vector3dVector(data['verts'].numpy())
        faces = o3d.utility.Vector3iVector(data['faces'].numpy())

        mesh = o3d.geometry.TriangleMesh(verts, faces)
        ret_dict = self.sample_points(mesh)
        ret_dict['model_ids'] = torch.tensor(idx, dtype=torch.int64)

        return ret_dict

class SingleR2N2Dataset(torch.utils.data.IterableDataset, ShapeNetDatasetBase):
    def __init__(self, shapenet_dir, r2n2_dir, model_id, on_surface_points, normalize=False, category=None, subsample=None, keep_aspect_ratio=True):
        torch.utils.data.IterableDataset.__init__(self)
        ShapeNetDatasetBase.__init__(self, on_surface_points, normalize, keep_aspect_ratio)

        split_path = get_r2n2_split_path(r2n2_dir, category, ensure=True)
        dataset = pytorch3d.datasets.R2N2('train', shapenet_dir, r2n2_dir, split_path,
            return_all_views=False, return_voxels=False, load_textures=False)

        data = dataset[model_id]
        verts = o3d.utility.Vector3dVector(data['verts'].numpy())
        faces = o3d.utility.Vector3iVector(data['faces'].numpy())
        self.mesh = o3d.geometry.TriangleMesh(verts, faces)

    def __iter__(self):
        while True:
            yield self.sample_points(self.mesh)

class VolumeFolderDataset(torch.utils.data.Dataset):

    def __init__(self, root, lazy_load=False):
        super().__init__()

        self.file_names = []
        for filename in sorted(os.listdir(root)):
            basename, _ = os.path.splitext(filename)
            suffix = basename.split('.')[-1]
            if suffix.lower() not in ['sdf', 'occ', 'vol']:
                continue
            self.file_names.append(os.path.join(root, filename))
        assert len(self.file_names) > 0

        vol = self.load_volume(self.file_names[0]) # [D, H, W, C]
        self.volume_size = tuple(vol.shape[:3])
        self.num_channels = int(vol.shape[-1])

        self.volumes = None
        if not lazy_load:
            volumes = []
            for file_path in self.file_names:
                vol = self.load_volume(file_path)
                volumes.append(vol)
            self.volumes = torch.stack(volumes, 0) # [N, D, H, W, C]

    def load_volume(self, path):
        vol = np.load(path) # [D, H, W, C]
        if vol.ndim == 3:
            vol = vol[..., None] # [D, H, W] -> [D, H, W, 1]
        vol = torch.from_numpy(vol) # [D, H, W, C]
        return vol

    @property
    def num_volumes(self):
        return len(self.file_names)

    def __len__(self):
        return self.num_volumes

    def __getitem__(self, idx):

        if self.volumes is None:
            vol = self.load_volume(self.file_names[idx]) # [D, H, W, C]
        else:
            vol = self.volumes[idx]

        return vol, torch.tensor(idx, dtype=torch.int64)

class VolumeSliceDataset(torch.utils.data.Dataset):

    def __init__(self, vol_dataset, i_slice):
        super().__init__()

        self.dataset = vol_dataset
        self.i_slice = i_slice

    @property
    def num_channels(self):
        return self.dataset.num_channels

    @property
    def slice_size(self):
        return self.dataset.volume_size[1:]

    @property
    def num_volumes(self):
        return self.dataset.num_volumes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        vol, ids = self.dataset[idx]
        return vol[self.i_slice], ids
