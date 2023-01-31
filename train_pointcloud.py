import os
import shutil
import time

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
import configargparse
import trimesh
from functools import partial

import models
import data_3d as data

import diff_ops
import utils

@torch.no_grad()
def evaluate(model, sdf_loader, coords_loader, args, device, pbar=None):

    N = args.resolution

    model.eval()

    batched_sdf_values = []
    for i, coords in enumerate(coords_loader):
        if pbar is not None:
            pbar.set_description(f'Coord Iter: {i+1} / {len(coords_loader)}')
            pbar.refresh()

        coords = coords[0].to(device) # [N_chunk, 3]
        preds = model(coords) # [N_chunk, 1]
        batched_sdf_values.append(preds.cpu())

    sdf_values = torch.cat(batched_sdf_values, 0) # [N_coords, 1]
    sdf_values = sdf_values.reshape(N, N, N).numpy()

    if pbar is not None:
        pbar.set_description('Marching cube ...')
        pbar.refresh()

    start_time = time.time()
    verts, faces, normals = utils.convert_sdf_samples_to_mesh(
        sdf_values,
        voxel_grid_origin=[-1, -1, -1],
        voxel_size=2.0 / (N - 1),
        offset=None,
        scale=None,
    )
    mesh = {'verts': verts, 'faces': faces, 'normals': normals}

    if pbar is not None:
        pbar.set_description(f'Done. marching cube took {time.time() - start_time} s')
        pbar.refresh()

    sdf_batch = next(sdf_loader)
    gt_coords, gt_normals = sdf_batch['on_surface_coords'], sdf_batch['on_surface_normals']
    metrics = utils.compute_pcd_distance(verts, gt_coords.numpy(), normals, gt_normals.numpy())

    return mesh, sdf_values, metrics

def train_one_step(model, optimizer, batch, args, device):

    total_loss = 0.

    model.train()

    on_surface_coords = batch['on_surface_coords'].to(device)
    on_surface_normals = batch['on_surface_normals'].to(device)
    off_surface_coords = batch['off_surface_coords'].to(device)

    coords_org = on_surface_coords.clone().detach().requires_grad_(True)
    on_surface_coords = coords_org

    on_surface_pred = model(on_surface_coords)
    off_surface_pred = model(off_surface_coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.abs(on_surface_pred)
    inter_constraint = torch.exp(-1e2 * torch.abs(off_surface_pred))

    loss = sdf_constraint.mean() * 3e3 \
        + inter_constraint.mean() * 1e2
    if args.loss_norm:
        gradient = diff_ops.gradient(on_surface_pred, on_surface_coords)
        normal_constraint = 1 - F.cosine_similarity(gradient, on_surface_normals, dim=-1)
        grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

        loss = loss + normal_constraint.mean() * 1e2 \
            + grad_constraint.mean() * 5e1

    total_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
            
    return total_loss

def main(args):
    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')
    
    if args.dataset == 'pointcloud':
        sdf_dataset = data.PointCloudDataset(args.pointcloud_path, on_surface_points=args.batch_size, normalize=True)
    else:
        sdf_dataset = data.SingleR2N2Dataset(args.shapenet_dir, args.r2n2_dir, model_id=args.model_id, on_surface_points=args.batch_size,
            category=args.category, normalize=True)
    dataloader = iter(sdf_dataset)

    if args.restart:
        shutil.rmtree(args.log_dir, ignore_errors=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # build model and optimizer
    if args.siren:
        args.pos_emb = 'Id'
        args.act_type = 'sine'
    model = models.INRNet(args, out_features=1, in_features=3).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # load checkpoints
    start_step = 0
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        ckpts = sorted(os.listdir(checkpoint_dir))
        ckpts = [os.path.join(checkpoint_dir, f) for f in ckpts if f.endswith('.ckpt')]
        if len(ckpts) > 0 and not args.restart:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            start_step = ckpt['current_step']
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
    os.makedirs(checkpoint_dir, exist_ok=True)

    # tensorboard logger
    summaries_dir = os.path.join(args.log_dir, 'tensorboard')
    os.makedirs(summaries_dir, exist_ok=True)
    writer = SummaryWriter(summaries_dir, purge_step=start_step)

    if args.test_only:
        model.eval()
        
        N = args.resolution
        coords = data.get_3d_mgrid((N, N, N))
        coords_dataset = torch.utils.data.TensorDataset(coords)
        coords_loader = DataLoader(coords_dataset, shuffle=False, batch_size=args.eval_batch_size, pin_memory=True, num_workers=8)

        with tqdm(total=len(coords_loader)) as pbar:
            mesh, sdf_values, metrics = evaluate(model, dataloader, coords_loader, args, device, pbar)
            cd, hd, nc = metrics['chamfer'], metrics['hausdorff'], metrics['normal']
            print(f'[TEST] CD: {cd:.4f} HD: {hd:.4f} NC: {nc:.4f}')

            mesh = trimesh.Trimesh(vertices=mesh['verts'], faces=mesh['faces'], vertex_normals=mesh['normals'])
            trimesh.exchange.export.export_mesh(mesh, os.path.join(args.log_dir, f'{start_step:04d}.ply'))

            np.save(os.path.join(args.log_dir, f'{start_step:04d}.npy'), sdf_values)
        return

    # training

    with tqdm(total=args.num_steps) as pbar:

        pbar.update(start_step)

        time0 = time.time()

        for current_step in range(start_step, args.num_steps):
            batch = next(dataloader)
            total_loss = train_one_step(model, optimizer, batch, args, device)
            pbar.set_description(f'LOSS: {total_loss:.4f}')
            pbar.update(1)

            if current_step % args.steps_til_ckpt == 0:
                pbar.set_description('Checkpointing ...')
                pbar.refresh()
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'current_step': current_step
                }
                torch.save(save_dict, os.path.join(checkpoint_dir, f'{current_step:04d}.ckpt'))


if __name__ == '__main__':

    p = configargparse.ArgumentParser()

    p.add_argument('--config', is_config_file=True, help='config file path')
    p.add_argument('--log_dir', type=str, required=True, help='directory path for logging')
    p.add_argument('--gpuid', type=int, default=0, help='cuda device number')
    p.add_argument('--test_only', action='store_true', help='test only (without training)')
    p.add_argument('--restart', action='store_true', help='do not reload from checkpoints')

    # dataset options
    p.add_argument('--dataset', type=str, default='shapenet', choices=['shapenet', 'pointcloud'], help='dataset type')
    p.add_argument('--category', type=str, default='chair', help='subclass for shapenet')
    p.add_argument('--model_id', type=int, default=0, help='model index in the shapenet')
    p.add_argument('--shapenet_dir', type=str, help='root path for shapenet dataset')
    p.add_argument('--r2n2_dir', type=str, help='root path for r2n2 dataset')
    p.add_argument('--pointcloud_path', type=str, help='root path to point clouds')

    # general training options
    p.add_argument('--batch_size', type=int, default=1024, help='batch size of sampled points when training')
    p.add_argument('--eval_batch_size', type=int, default=102400, help='batch size of sampled points when evaluation')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
    p.add_argument('--weight_decay', type=float, default=0., help='weight decay. default=0.')
    p.add_argument('--num_steps', type=int, default=50000, help='Number of iterations to train network')
    p.add_argument('--loss_norm', type=bool, default=True, help='turn on loss to rectify SDF norm')
    p.add_argument('--no_loss_norm', action='store_false', dest='loss_norm', help='disable loss to rectify SDF norm')
    p.set_defaults(loss_norm=True)
    p.add_argument('--loss_l1', type=float, default=3e1, help='coefficient for L1 sparisty')

    # network architecture specific options
    p.add_argument('--num_layers', type=int, default=4, help='number of layers of network')
    p.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension of network')
    p.add_argument('--code_dim', type=int, default=1024, help='dimension of coding (num basis in dictionary)')
    p.add_argument('--pos_emb', type=str, default='ffm', choices=['Id', 'rbf', 'pe', 'ffm', 'gffm'],
                help='coordinate embedding function applied before FC layers.')
    p.add_argument('--act_type', type=str, default='relu', choices=['relu', 'sine'],
                help='activation function between FC layers')
    p.add_argument('--siren', action='store_true', help='substitute relu activation function with sin')

    # rbf specific options
    p.add_argument('--rbf_centers', type=int, default=1024,
                help='number of center of rbf')

    # PE specific options
    p.add_argument('--num_freqs', type=int, default=-1,
                help='number of frequncy bands. Calculated by default.')
    p.add_argument('--use_nyquist', action='store_true',
                help='Use Nyquist theorem to estimate appropriate number of frequencies.')

    # ffm specific options
    p.add_argument('--ffm_map_size', type=int, default=4096,
                help='mapping dimension of ffm')
    p.add_argument('--ffm_map_scale', type=float, default=16,
                help='Gaussian mapping scale of positional input')

    # gffm specific options
    p.add_argument('--kernel', type=str, default="exp", help='choose from [exp], [exp2], [matern], [gamma_exp], [rq], [poly]')
    p.add_argument('--gffm_map_size', type=int, default=4096,
                help='mapping dimension of gffm')
    p.add_argument('--length_scale', type=float, default=64, help='(inverse) length scale of [exp,matern,gamma] kernel')
    p.add_argument('--matern_order', type=float, default=0.5, help='\nu in Matern class kernel function')
    p.add_argument('--gamma_order', type=float, default=1, help='gamma in gamma-exp kernel')
    p.add_argument('--rq_order', type=float, default=4, help='order in rational-quadratic kernel')
    p.add_argument('--poly_order', type=float, default=4, help='order in polynomial kernel')

    # logging/saving options
    p.add_argument('--steps_til_summary', type=int, default=1,
                help='Step interval until loss is printed')
    p.add_argument('--steps_til_eval', type=int, default=1,
                help='Step interval until evaluation')
    p.add_argument('--steps_til_ckpt', type=int, default=100,
                help='Step interval until checkpoint is saved')

    p.add_argument('--resolution', type=int, default=256,
                help='Mesh resolution when exporting')


    args = p.parse_args()
    main(args)