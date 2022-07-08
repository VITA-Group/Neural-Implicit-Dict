import os
import shutil
import time

from tqdm import tqdm, trange
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import pytorch3d
import pytorch3d.datasets

import open3d as o3d

p = argparse.ArgumentParser()
p.add_argument('--shapenet_dir', type=str, required=True)
p.add_argument('--r2n2_dir', type=str, required=True)
p.add_argument('--out_dir', type=str, required=True)
p.add_argument('--category', type=str, default='chair')
p.add_argument('--subsample', type=int, default=8)
p.add_argument('--version', type=int, default=1)
p.add_argument('--N_points', type=int, default=10000)
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

# dataset = pytorch3d.datasets.ShapeNetCore(args.data_dir, synsets=[args.category], version=args.version, load_textures=False)
dataset = pytorch3d.datasets.R2N2('train', args.shapenet_dir, args.r2n2_dir, split_path,
    return_all_views=False, return_voxels=False, load_textures=False)

subset = range(len(dataset)) if args.subsample < 0 else range(args.subsample)
for i in tqdm(subset):
    data = dataset[i]
    verts = o3d.utility.Vector3dVector(data['verts'].numpy())
    faces = o3d.utility.Vector3iVector(data['faces'].numpy())
    model_id = data['model_id']

    mesh = o3d.geometry.TriangleMesh(verts, faces)
    pcd = mesh.sample_points_uniformly(args.N_points, use_triangle_normal=True)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    pcd_arr = np.concatenate([points, normals], -1)

    save_path = os.path.join(args.out_dir, model_id+'.xyz')
    np.savetxt(save_path, pcd_arr)
