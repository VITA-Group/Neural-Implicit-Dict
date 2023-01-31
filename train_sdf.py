import os
import shutil
import time

from tqdm import tqdm
import configargparse
from functools import partial
import imageio
import open3d as o3d

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
import data_3d as data
import utils

@torch.no_grad()
def evaluate(model, test_loader, args, current_epoch, device, pbar=None):
    model.eval()
    total_mse = total_psnr = 0.
    N_total = 0
    warmup = current_epoch <= args.warmup_epochs

    N = max(test_loader.dataset.volume_size)
    lattice_dataset = data.LatticeDataset((N, N, N))
    coords_loader = DataLoader(lattice_dataset, pin_memory=True, batch_size=args.chunk_size, shuffle=False)

    save_dir = os.path.join(args.log_dir, f'eval_{current_epoch:05d}')
    os.makedirs(save_dir, exist_ok=True)

    for i, (vol, sample_ids) in enumerate(test_loader):
        sample_ids = sample_ids.to(device)
        batch_vol_values = []
        for coords, i_sel in coords_loader:
            coords, i_sel = coords.to(device), i_sel.to(device)

            model_input = {
                'voxels': vol,
                'sample_ids': sample_ids,
                'coords': coords
            }

            model_output = model(model_input, topk_sparse=(not warmup))
            batch_vol_values.append(model_output['preds'].cpu())

        B, C = vol.shape[0], vol.shape[-1]
        pred_vol = torch.cat(batch_vol_values, 1) # [B, NxNxN, 1]

        mse = torch.mean((pred_vol.reshape(B, -1) - vol.reshape(B, -1)) ** 2., -1) # [B]
        psnr = -10. * torch.log10(mse)  # [B]

        if pbar is not None:
            pbar.set_description(f'[TEST] EPOCH {current_epoch} ITER: {i}/{len(test_loader)} '
                    f'MSE: {torch.mean(mse).item():.4f} PSNR: {torch.mean(psnr).item():.4f}')
        else:
            print(f'[TEST] EPOCH {current_epoch} ITER: {i}/{len(test_loader)} MSE: {torch.mean(mse).item():.4f} '
                f'PSNR: {torch.mean(psnr).item():.4f}')

        total_mse += torch.sum(mse).item()
        total_psnr += torch.sum(psnr).item()
        N_total += mse.shape[0]

        pred_vol = pred_vol.reshape(B, N, N, N)
        for i, v in zip(sample_ids, pred_vol):
            np.save(os.path.join(save_dir, f'{i:04d}.npy'), v.numpy())

            if pbar is not None:
                pbar.set_description('Marching cube ...')
                pbar.refresh()

            start_time = time.time()
            verts, faces, normals = utils.convert_sdf_samples_to_mesh(
                v.numpy(),
                voxel_grid_origin=[-1, -1, -1],
                voxel_size=2.0 / (N - 1),
                offset=None,
                scale=None,
            )

            if pbar is not None:
                pbar.set_description(f'Done. marching cube took {time.time() - start_time} s')
                pbar.refresh()

            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(verts),
                triangles=o3d.utility.Vector3iVector(faces)
            )
            o3d.io.write_triangle_mesh(os.path.join(save_dir, f'{i:04d}.ply'), mesh)

    return {'mse': total_mse / N_total, 'psnr': total_psnr / N_total}

def train_one_step(model, optimizer, data_batch, coords_batch, warmup, args, device):

    vol, sample_ids = data_batch
    coords, i_sel = coords_batch

    vol, sample_ids = vol.to(device), sample_ids.to(device)
    coords, i_sel = coords.to(device), i_sel.to(device)

    model_input = {
        'voxels': vol,
        'sample_ids': sample_ids,
        'coords': coords
    }

    B, C = vol.shape[0], vol.shape[-1]
    y = vol.reshape(B, -1, C)[:, i_sel, :]

    model.train()
    model_output = model(model_input, topk_sparse=(not warmup))
    y_pred = model_output['preds']
    if args.loss_type == 'l2':
        mse = torch.mean((y_pred - y) ** 2.)
    elif args.loss_type == 'l1':
        mse = torch.mean(torch.abs(y_pred - y))
    psnr = -10. * torch.log10(mse)

    loss = mse

    if args.loss_cv > 0:
        importance = model_output['importance']
        cv_squared = models.cv_squared_loss(importance)
        if not warmup:
            load = model_output['load']
            cv_squared = cv_squared + models.cv_squared_loss(load)
        loss = loss + args.loss_cv * cv_squared

    if args.loss_l1 > 0 and warmup:
        gates = model_output['gates']
        sparsity = torch.mean(torch.abs(gates))
        loss = loss + args.loss_l1 * sparsity

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {'loss': loss.item(), 'mse': mse.item(), 'psnr': psnr.item()}

def train_one_epoch(model, optimizer, train_loader, coords_loader, args, writer, current_epoch, device, pbar):

    warmup = current_epoch <= args.warmup_epochs

    coords_iter = iter(coords_loader)
    for i, data_batch in enumerate(train_loader):

        try:
            coords_batch = next(coords_iter)
        except StopIteration:
            coords_iter = iter(coords_loader)
            coords_batch = next(coords_iter)

        train_log = train_one_step(model, optimizer, data_batch, coords_batch, warmup, args, device)
        loss, mse, psnr = train_log['loss'], train_log['mse'], train_log['psnr']

        if pbar is not None:
            pbar.update()
            pbar.set_description(f'[TRAIN] EPOCH: {current_epoch} LOSS: {loss:.4f} MSE: {mse:.4f} PSNR: {psnr:.4f}')

        if (i+len(train_loader)*current_epoch) % args.steps_til_summary == 0:
            with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
                print(f'[TRAIN] EPOCH: {current_epoch} ITER: {i}/{len(train_loader)} LOSS: {loss:.4f} '
                    f'MSE: {mse:.4f} PSNR: {psnr:.4f}', file=f)
            writer.add_scalar('train_loss', loss.item(), len(train_loader) * current_epoch + i)


def train_one_epoch_recursive(model, optimizer, train_loader, coords_loader, args, writer, current_epoch, device, pbar):

    warmup = current_epoch <= args.warmup_epochs
    for i, data_batch in enumerate(train_loader):
        all_loss, all_mse, all_psnr = [], [], []
        for coords_batch in coords_loader:
            train_log = train_one_step(model, optimizer, data_batch, coords_batch, warmup, args, device)
            loss, mse, psnr = train_log['loss'], train_log['mse'], train_log['psnr']

            all_loss.append(loss)
            all_mse.append(mse)
            all_psnr.append(psnr)

        loss, mse, psnr = np.array(all_loss).mean(), np.array(all_mse).mean(), np.array(all_psnr).mean()

        pbar.update()
        pbar.set_description(f'EPOCH: {current_epoch} LOSS: {loss:.4f} MSE: {mse:.4f} PSNR: {psnr:.4f}')

        if (i+len(train_loader)*current_epoch) % args.steps_til_summary == 0:
            with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
                print(f'[TRAIN] EPOCH: {current_epoch} ITER: {i}/{len(train_loader)} LOSS: {loss:.4f} '
                    f'MSE: {mse:.4f} PSNR: {psnr:.4f}', file=f)
            writer.add_scalar('train_loss', loss.item(), len(train_loader) * current_epoch + i)

def main(args):
    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')

    if args.restart and os.path.exists(args.log_dir):
        answer = utils.ask_input(f'Are you sure to clear log directory: {args.log_dir}', ['y', 'n'])
        if answer == 'y':
            shutil.rmtree(args.log_dir, ignore_errors=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # prepare data loader
    train_dataset = data.VolumeFolderDataset(root=os.path.join(args.data_dir))
    train_loader = DataLoader(train_dataset, pin_memory=True, batch_size=args.batch_size, shuffle=True)

    if args.inner_loop == 'recursive':
        lattice_dataset = data.LatticeDataset(train_dataset.volume_size)
        train_coords = DataLoader(lattice_dataset, pin_memory=True, batch_size=args.chunk_size, shuffle=True)
    else:
        train_coords = data.IterableLatticeDataset(train_dataset.volume_size, batch_size=args.chunk_size, shuffle=True)

    # build model and optimizer
    if args.gate_type == 'codebook':
        gate_module = partial(models.CodebookEncoder, num_images=len(train_dataset), max_norm=1., norm_type=2.)
    elif args.gate_type == 'pointnet':
        raise NotImplementedError('PointNet not implemented yet.')
    model = models.INRMoE(args, in_dim=3, out_dim=train_dataset.num_channels, bias=True, gate_module=gate_module)
    model = model.to(device)

    # freeze dictionary
    if args.finetune:
        model.freeze_dict()

    model_params = model.code_parameters() if args.finetune else model.parameters()
    optimizer = torch.optim.Adam(params=model_params, lr=args.lr, weight_decay=args.weight_decay)

    # load checkpoints
    start_epoch = 0
    ckpt_path = args.ckpt_path
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir) and not os.path.exists(ckpt_path):
        ckpts = sorted(os.listdir(checkpoint_dir))
        ckpts = [os.path.join(checkpoint_dir, f) for f in ckpts if f.endswith('.ckpt')]
        if len(ckpts) > 0 and not args.restart:
            ckpt_path = ckpts[-1]
    if os.path.exists(ckpt_path):
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['current_epoch']        
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    os.makedirs(checkpoint_dir, exist_ok=True)

    # tensorboard logger
    summaries_dir = os.path.join(args.log_dir, 'tensorboard')
    os.makedirs(summaries_dir, exist_ok=True)
    writer = SummaryWriter(summaries_dir, purge_step=start_epoch*len(train_loader))

    # training
    if args.test_only:
        # make full testing
        print("Running full validation set...")
        metrics = evaluate(model, train_loader, args, start_epoch, device)
        print(f'[TEST] MSE: {metrics["mse"]:.4f} PSNR: {metrics["psnr"]:.4f}')
        return
    
    with tqdm(total=len(train_loader) * args.num_epochs) as pbar:

        pbar.update(len(train_loader) * start_epoch)

        for current_epoch in range(start_epoch, args.num_epochs+1):
            if args.inner_loop == 'recursive':
                train_one_epoch_recursive(model, optimizer, train_loader, train_coords, args, writer, current_epoch, device, pbar)
            else:
                train_one_epoch(model, optimizer, train_loader, train_coords, args, writer, current_epoch, device, pbar)

            if current_epoch % args.epochs_til_ckpt == 0:
                pbar.set_description('Checkpointing ...')
                pbar.refresh()
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'current_epoch': current_epoch
                }
                torch.save(save_dict, os.path.join(checkpoint_dir, f'{current_epoch:04d}.ckpt'))

if __name__ == '__main__':
    p = configargparse.ArgumentParser()

    p.add_argument('--config', is_config_file=True, help='config file path')
    p.add_argument('--data_dir', type=str, required=True, help='root path for dataset')
    p.add_argument('--log_dir', type=str, required=True, help='directory path for logging')
    p.add_argument('--ckpt_path', type=str, default='', help='path to the reloaded checkpoint')
    p.add_argument('--gpuid', type=int, default=0, help='cuda device number')
    p.add_argument('--restart', action='store_true', help='do not reload from checkpoints')

    # general training options
    p.add_argument('--batch_size', type=int, default=64, help='batch size of images')
    p.add_argument('--chunk_size', type=int, default=1024, help='number of pixels for images')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
    p.add_argument('--weight_decay', type=float, default=0., help='weight decay. default=0.')
    p.add_argument('--num_epochs', type=int, default=5000, help='Nnmber of epochs to train network')
    p.add_argument('--warmup_epochs', type=int, default=0, help='Nnmber of epochs to warm-up without sparsity')
    p.add_argument('--loss_type', type=str, default='l2', help='loss type to minimize regression difference')
    p.add_argument('--loss_cv', type=float, default=0.01, help='coefficient for CV penality')
    p.add_argument('--loss_l1', type=float, default=0.01, help='coefficient for L1 sparsity')
    p.add_argument('--inner_loop', type=str, default='recursive', choices=['random', 'recursive'],
                help=' inner loop strategy for traversing coords batchs')
    p.add_argument('--test_only', action='store_true', help='test only (without training)')
    p.add_argument('--finetune', action='store_true', help='freeze the dictionary while training')

    # network architecture specific options
    p.add_argument('--num_layers', type=int, default=4, help='number of layers of network')
    p.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension of network')
    p.add_argument('--num_topk', type=int, default=128, help='dimension of coding (num experts to be used)')
    p.add_argument('--num_experts', type=int, default=1024, help='number of experts (num basis in dictionary)')
    p.add_argument('--pos_emb', type=str, default='ffm', choices=['Id', 'rbf', 'pe', 'ffm', 'gffm'],
                help='coordinate embedding function applied before FC layers.')
    p.add_argument('--act_type', type=str, default='relu', choices=['relu', 'sine'],
            help='activation function between FC layers')
    p.add_argument('--siren', action='store_true', help='substitute relu activation function with sin')
    p.add_argument('--gate_type', type=str, default='conv', choices=['pointnet', 'codebook'],
            help='type of gating network')

    p.add_argument('--kernel', type=str, default="exp", help='choose from [exp], [exp2], [matern], [gamma_exp], [rq], [poly]')
    p.add_argument('--ffm_map_size', type=int, default=4096,
                help='mapping dimension of ffm')
    p.add_argument('--ffm_map_scale', type=float, default=16,
                help='Gaussian mapping scale of positional input')
    p.add_argument('--gffm_map_size', type=int, default=4096,
                help='mapping dimension of gffm')
    # gffm specific options
    p.add_argument('--length_scale', type=float, default=64, help='(inverse) length scale of [exp,matern,gamma] kernel')
    p.add_argument('--matern_order', type=float, default=0.5, help='\nu in Matern class kernel function')
    p.add_argument('--gamma_order', type=float, default=1, help='gamma in gamma-exp kernel')
    p.add_argument('--rq_order', type=float, default=4, help='order in rational-quadratic kernel')
    p.add_argument('--poly_order', type=float, default=4, help='order in polynomial kernel')

    # logging/saving options
    p.add_argument('--epochs_til_eval', type=int, default=1,
                help='Epoch interval until evaluation')
    p.add_argument('--epochs_til_render', type=int, default=100,
                help='Epoch interval until rendering')
    p.add_argument('--epochs_til_ckpt', type=int, default=100,
                help='Epoch interval until checkpoint is saved')
    p.add_argument('--steps_til_summary', type=int, default=100,
                help='Step interval until loss is printed')

    args = p.parse_args()

    main(args)