import os
import shutil
import time
import math

from tqdm import tqdm
import configargparse
import imageio
from functools import partial

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import siren
import data_2d as data

import utils

@torch.no_grad()
def render(model, render_loader, args, current_epoch, save_dir, device):
    model.eval()

    j = 0
    list_rgb = []
    H, W = render_loader.dataset.image_size
    C = render_loader.dataset.num_channels
    for i, (coords, rgbs, _) in enumerate(render_loader):
        coords, rgbs = coords.to(device), rgbs.to(device)
        y_pred = model(coords)
        list_rgb.append(y_pred.cpu())
    image = torch.cat(list_rgb, 0).reshape(H, W, C)
    image = np.clip(image.numpy() * 255, 0, 255).astype(np.uint8)
    imageio.imwrite(os.path.join(save_dir, f'test_{current_epoch:04d}.png'), image)

@torch.no_grad()
def evaluate(model, test_loader, args, current_epoch, device, pbar=None):
    model.eval()
    total_mse = total_psnr = 0.
    N_total = 0

    j = 0
    list_gt = []
    list_rgb = []
    H, W = test_loader.dataset.image_size
    C = test_loader.dataset.num_channels
    for i, (coords, rgbs, _) in enumerate(test_loader):
        coords, y_gt = coords.to(device), rgbs.to(device)
        y_pred = model(coords)
        list_gt.append(y_gt.cpu())
        list_rgb.append(y_pred.cpu())

    pred = torch.cat(list_rgb, 0).reshape(H, W, C)[None, ...]
    gt = torch.cat(list_gt, 0).reshape(H, W, C)[None, ...]

    mse = torch.mean((pred - gt) ** 2.).item()
    psnr = utils.psnr(pred, gt, format='NHWC').mean().item()
    ssim = utils.ssim(pred, gt, format='NHWC').mean().item()
    lpips  = utils.lpips(pred, gt, format='NHWC').mean().item()

    if pbar is not None:
        pbar.set_description(f'[TEST] EPOCH {current_epoch} MSE: {total_mse:.4f} PSNR: {psnr:.4f} '
            f'SSIM: {ssim:.4f} LPIPS: {lpips:.4f}')
    with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
        print(f'[TEST] EPOCH {current_epoch} MSE: {total_mse:.4f} PSNR: {psnr:.4f} '
            f'SSIM: {ssim:.4f} LPIPS: {lpips:.4f}', file=f)

    return total_mse, psnr, ssim, lpips

def train_one_epoch(model, optimizer, train_loader, args, writer, current_epoch, device, pbar):

    H, W = train_loader.dataset.image_size
    C = train_loader.dataset.num_channels

    for i, (coords, rgbs, _) in enumerate(train_loader):
        coords, y = coords.to(device), rgbs.to(device)

        model.train()
        y_pred = model(coords)
        if args.loss_type == 'l2':
            mse = torch.mean((y_pred - y) ** 2.)
        elif args.loss_type == 'l1':
            mse = torch.mean(torch.abs(y_pred - y))
        psnr = -10. * torch.log10(mse)

        loss = mse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update()
        pbar.set_description(f'[TRAIN] EPOCH: {current_epoch} LOSS: {loss.item():.4f} MSE: {mse.item():.4f} '
            f'PSNR: {psnr.item():.4f}')

def main(args):
    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')

    if args.restart:
        shutil.rmtree(args.log_dir, ignore_errors=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # prepare data loader
    dataset = data.ImageDataset(args.image_path, crop=True)

    train_loader = DataLoader(dataset, pin_memory=True, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, pin_memory=True, batch_size=args.batch_size, shuffle=False)
    render_loader = DataLoader(dataset, pin_memory=True, batch_size=args.batch_size, shuffle=False)

    # build model and optimizer
    model = siren.INRNet(args, in_features=2, out_features=dataset.num_channels)
    model = model.to(device)

    print(f'# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'# FLOPs: {model.flops * dataset.image_size[0] * dataset.image_size[1]}')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # load checkpoints
    start_epoch = 0
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        ckpts = sorted(os.listdir(checkpoint_dir))
        ckpts = [os.path.join(checkpoint_dir, f) for f in ckpts if f.endswith('.ckpt')]
        if len(ckpts) > 0 and not args.restart:
            ckpt_path = ckpts[-1]
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
        time0 = time.time()
        mse, psnr, ssim, lpips = evaluate(model, val_loader, args, start_epoch, device)
        print(f'[TEST] EPOCH {start_epoch} MSE: {mse:.4f} PSNR: {psnr:.4f} '
            f'SSIM: {ssim:.4f} LPIPS: {lpips:.4f} Throughput: {1.0 / (time.time() - time0):4f}')
        return
    
    with tqdm(total=len(train_loader) * args.num_epochs) as pbar:

        pbar.update(len(train_loader) * start_epoch)

        for current_epoch in range(start_epoch, args.num_epochs+1):

            if current_epoch % args.epochs_til_eval == 0:
                pbar.set_description('Evaluating ...')
                pbar.refresh()
                mse, psnr, ssim, lpips = evaluate(model, val_loader, args, current_epoch, device, pbar)
                with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
                    print(f'[TEST] EPOCH {current_epoch} MSE: {mse:.4f} PSNR: {psnr:.4f} '
                        f'SSIM: {ssim:.4f} LPIPS: {lpips:.4f}', file=f)

            if current_epoch % args.epochs_til_render == 0:
                pbar.set_description('Rendering ...')
                pbar.refresh()
                save_dir = os.path.join(args.log_dir, f'render_{current_epoch:04d}')
                os.makedirs(save_dir, exist_ok=True)
                render(model, render_loader, args, current_epoch, save_dir, device)

            if current_epoch % args.epochs_til_ckpt == 0:
                pbar.set_description('Checkpointing ...')
                pbar.refresh()
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'current_epoch': current_epoch
                }
                torch.save(save_dict, os.path.join(checkpoint_dir, f'{current_epoch:04d}.ckpt'))

            train_one_epoch(model, optimizer, train_loader, args, writer, current_epoch, device, pbar)

if __name__ == '__main__':
    p = configargparse.ArgumentParser()

    p.add_argument('--config', is_config_file=True, help='config file path')
    p.add_argument('--image_path', type=str, required=True, help='root path for dataset')
    p.add_argument('--log_dir', type=str, required=True, help='directory path for logging')
    p.add_argument('--gpuid', type=int, default=0, help='cuda device number')
    p.add_argument('--test_only', action='store_true', help='test only (without training)')
    p.add_argument('--restart', action='store_true', help='do not reload from checkpoints')

    # general training options
    p.add_argument('--batch_size', type=int, default=1024, help='batch size of images')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
    p.add_argument('--weight_decay', type=float, default=0., help='weight decay. default=0.')
    p.add_argument('--num_epochs', type=int, default=5000, help='Nnmber of epochs to train network')
    p.add_argument('--loss_type', type=str, default='l2', help='loss type to minimize regression difference')
    p.add_argument('--loss_cv', type=float, default=0.01, help='coefficient for CV penality')
    p.add_argument('--loss_l1', type=float, default=0.01, help='coefficient for L1 sparsity')
    p.add_argument('--inner_loop', type=str, default='recursive', choices=['random', 'recursive'],
                help=' inner loop strategy for traversing coords batchs')

    # network architecture specific options
    p.add_argument('--num_layers', type=int, default=4, help='number of layers of network')
    p.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension of network')
    p.add_argument('--pos_emb', type=str, default='ffm', choices=['Id', 'rbf', 'pe', 'ffm', 'gffm'],
                help='coordinate embedding function applied before FC layers.')
    p.add_argument('--act_type', type=str, default='relu', choices=['relu', 'sine'],
            help='activation function between FC layers')
    p.add_argument('--siren', action='store_true', help='substitute relu activation function with sin')

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