import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import shutil
import time
import json

from tqdm import tqdm, trange
import argparse

import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import pytorch3d
import pytorch3d.datasets

import open3d as o3d
import manifold

import data_3d as data
import utils

synset_dict = {
    "04256520": "sofa",
    "02933112": "cabinet",
    "02828884": "bench",
    "03001627": "chair",
    "03211117": "display",
    "04090263": "rifle",
    "03691459": "loudspeaker",
    "03636649": "lamp",
    "04401088": "telephone",
    "02691156": "airplane",
    "04379243": "table",
    "02958343": "car",
    "04530566": "watercraft"
}

def normalize_points(coords, keep_aspect_ratio=True):
    coords -= np.mean(coords, axis=0, keepdims=True)
    if keep_aspect_ratio:
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
    else:
        coord_max = np.amax(coords, axis=0, keepdims=True)
        coord_min = np.amin(coords, axis=0, keepdims=True)

    coords = (coords - coord_min) / (coord_max - coord_min)
    coords -= 0.5
    coords *= 2.

    return coords

p = argparse.ArgumentParser()
p.add_argument('--shapenet_dir', type=str, required=True)
p.add_argument('--r2n2_dir', type=str, required=True)
p.add_argument('--out_dir', type=str, required=True)
p.add_argument('--category', type=str, default='chair')
p.add_argument('--subsample', type=int, default=-1)
p.add_argument('--version', type=int, default=1)
p.add_argument('--resolution', type=int, default=128)
p.add_argument('--manifold_depth', type=int, default=8)
args = p.parse_args()

split_path = os.path.join(args.r2n2_dir, f'{args.category}_splits.json')
if not os.path.exists(split_path):
    with open(os.path.join(args.r2n2_dir, 'pix2mesh_splits_val05.json'), 'r') as f:
        split_dict = json.load(f)
    inv_synset_dict = {label: offset for offset, label in synset_dict.items()}

    offset = inv_synset_dict[args.category]
    category_splits = {}
    category_splits['train'] = {offset: split_dict['train'][offset]}
    category_splits['val'] = {offset: split_dict['val'][offset]}
    category_splits['test'] = {offset: split_dict['test'][offset]}
    with open(split_path, 'w') as f:
        json.dump(category_splits, f)

os.makedirs(args.out_dir, exist_ok=True)

# dataset = pytorch3d.datasets.ShapeNetCore(args.shapenet_dir, synsets=[args.category], version=args.version, load_textures=False)
dataset = pytorch3d.datasets.R2N2('train', args.shapenet_dir, args.r2n2_dir, split_path,
    return_all_views=False, return_voxels=True, load_textures=False)
subset = range(len(dataset)) if args.subsample < 0 else range(args.subsample)
N = args.resolution

lattices = data.get_3d_mgrid((N, N, N)).numpy().reshape(N, N, N, 3)

for i in tqdm(subset):
    data_sample = dataset[i]
    model_id = data_sample['model_id']


    verts = data_sample['verts'].numpy().astype(np.float64)
    verts = normalize_points(verts)
    faces = data_sample['faces'].numpy().astype(np.int32)

    input_is_watertight = manifold.is_manifold(verts, faces)
    processor = manifold.Processor(verts, faces)
    verts, faces = processor.get_manifold_mesh(depth=args.manifold_depth)
    output_is_watertight = manifold.is_manifold(verts, faces)
    assert output_is_watertight, f'Failed to convert to watertight: {model_id}'

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts.astype(np.float32)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32))
    )

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    sdf_values = scene.compute_signed_distance(lattices)
    sdf_values = sdf_values.numpy()


    np.save(os.path.join(args.out_dir, f'{model_id}.sdf'), sdf_values)
    o3d.io.write_triangle_mesh(os.path.join(args.out_dir, f'{model_id}.ply'), mesh)

    verts, faces, normals = utils.convert_sdf_samples_to_mesh(
        sdf_values,
        voxel_grid_origin=[-1, -1, -1],
        voxel_size=2.0 / (N - 1),
        offset=None,
        scale=None,
    )

    recon_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    o3d.io.write_triangle_mesh(os.path.join(args.out_dir, f'{model_id}_mc.ply'), recon_mesh)