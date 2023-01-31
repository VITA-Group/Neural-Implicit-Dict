import os
import shutil
import time

from tqdm import tqdm
import configargparse
import imageio
from functools import partial

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import INRMoE, SimpleConvImgEncoder, CodebookImgEncoder, ResConvImgEncoder, cv_squared_loss
from data_2d import LatticeDataset, IterableLatticeDataset, YaleFaceDataset, \
    OlivettiFaceDataset, CIFAR10Dataset, CelebADataset, ImageFolderDataset
import utils


@torch.no_grad()
def render(model, render_loader, coords_loader, args, current_epoch, save_dir, device):
    model.eval()

    warmup = current_epoch <= args.warmup_epochs
    
    j = 0
    for i, (rgb, img_ids) in enumerate(render_loader):
        rgb, img_ids = rgb.to(device), img_ids.to(device)
        B, H, W, C = rgb.shape
        list_rgb = []
        for coords, i_sel in coords_loader:
            coords, i_sel = coords.to(device), i_sel.to(device)
            model_input = {
                'imgs': rgb,
                'img_ids': img_ids,
                'coords': coords
            }
            model_output = model(model_input, topk_sparse=(not warmup))
            y_pred = model_output['preds']
            list_rgb.append(y_pred.cpu())

        N = render_loader.dataset.num_patches_per_img
        orig_H, orig_W = render_loader.dataset.full_image_size

        # we assume the batch_size = N x k
        assert (B % N == 0), "Batch size should be a multiple of number of patches per image"

        # merge patches
        images = torch.cat(list_rgb, 1).reshape(B // N, N, H, W, C).permute([0, 4, 2, 3, 1]).reshape(B // N, -1, N) # [N, CxHxW, L]
        images = F.fold(images, (orig_H, orig_W), kernel_size=(H, W), stride=(H, W)).permute([0, 2, 3, 1]) # [N, H, W, C]
        for img in images:
            img = np.clip(img.numpy() * 255, 0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(save_dir, f'test_{j:04d}.png'), img)
            j += 1

@torch.no_grad()
def evaluate(model, test_loader, coords_loader, args, current_epoch, device, pbar=None):
    model.eval()
    total_mse = total_psnr = total_ssim = total_lpips = 0.
    N_total = 0
    warmup = current_epoch <= args.warmup_epochs
    for i, (rgb, img_ids) in enumerate(test_loader):
        rgb, img_ids = rgb.to(device), img_ids.to(device)
        # list_mse = []
        list_rgb = []
        list_gt = []
        for coords, i_sel in coords_loader:
            coords, i_sel = coords.to(device), i_sel.to(device)
            
            model_input = {
                'imgs': rgb,
                'img_ids': img_ids,
                'coords': coords
            }

            y_gt = rgb.reshape(rgb.shape[0], -1, rgb.shape[-1])[:, i_sel, :]

            model_output = model(model_input, topk_sparse=(not warmup))
            y_pred = model_output['preds']

            list_rgb.append(y_pred.cpu())
            list_gt.append(y_gt.cpu())

        B, C = rgb.shape[0], rgb.shape[-1]
        H, W = test_loader.dataset.image_size
        N = test_loader.dataset.num_patches_per_img
        orig_H, orig_W = test_loader.dataset.full_image_size

        # we assume the batch_size = N x k
        assert (B % N == 0), "Batch size should be a multiple of number of patches per image"

        # Get unfolded patches
        pred = torch.cat(list_rgb, 1).reshape(B // N, N, H, W, C).permute([0, 4, 2, 3, 1]).reshape(B // N, -1, N) # [N, CxHxW, L]
        gt = torch.cat(list_gt, 1).reshape(B // N, N, H, W, C).permute([0, 4, 2, 3, 1]).reshape(B // N, -1, N) # [N, CxHxW, L]

        # Merge patches
        pred = F.fold(pred, (orig_H, orig_W), kernel_size=(H, W), stride=(H, W))
        gt = F.fold(gt, (orig_H, orig_W), kernel_size=(H, W), stride=(H, W))

        mse = ((pred - gt) ** 2.).reshape(pred.shape[0], -1).mean(-1)
        psnr = utils.psnr(pred, gt, format='NCHW')
        ssim = utils.ssim(pred, gt, format='NCHW', size_average=False)
        lpips  = utils.lpips(pred, gt, format='NCHW')

        if pbar is not None:
            pbar.set_description(f'[TEST] EPOCH {current_epoch} ITER: {i}/{len(test_loader)} '
                    f'MSE: {torch.mean(mse).item():.4f} PSNR: {torch.mean(psnr).item():.4f} '
                    f'SSIM: {torch.mean(ssim).item():.4f} LPIPS: {torch.mean(lpips).item():.4f}')
        else:
            print(f'[TEST] EPOCH {current_epoch} ITER: {i}/{len(test_loader)} '
                f'MSE: {torch.mean(mse).item():.4f} PSNR: {torch.mean(psnr).item():.4f} '
                f'SSIM: {torch.mean(ssim).item():.4f} LPIPS: {torch.mean(lpips).item():.4f}')
        with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
            print(f'[TEST] EPOCH {current_epoch} ITER: {i}/{len(test_loader)} MSE: {torch.mean(mse).item():.4f} '
                f'PSNR: {torch.mean(psnr).item():.4f} SSIM: {torch.mean(ssim).item():.4f} '
                f'LPIPS: {torch.mean(lpips).item():.4f}', file=f)

        total_mse += torch.sum(mse).item()
        total_psnr += torch.sum(psnr).item()
        total_ssim += torch.sum(ssim).item()
        total_lpips += torch.sum(lpips).item()
        N_total += mse.shape[0]

    return total_mse / N_total, total_psnr / N_total, total_ssim / N_total, total_lpips / N_total

def train_one_epoch(model, optimizer, train_loader, coords_loader, args, writer, current_epoch, device, pbar):

    warmup = current_epoch <= args.warmup_epochs

    coords_iter = iter(coords_loader)
    for i, (rgb, img_ids) in enumerate(train_loader):

        try:
            coords, i_sel = next(coords_iter)
        except StopIteration:
            coords_iter = iter(coords_loader)
            coords, i_sel = next(coords_iter)

        rgb, img_ids = rgb.to(device), img_ids.to(device)
        coords, i_sel = coords.to(device), i_sel.to(device)

        model_input = {
            'imgs': rgb,
            'img_ids': img_ids,
            'coords': coords
        }

        B, H, W, C = rgb.shape
        y = rgb.reshape(B, -1, C)[:, i_sel, :]

        model.train()
        model_output = model(model_input, topk_sparse=(not warmup))
        y_pred = model_output['preds']
        if args.loss_type == 'l2':
            mse = torch.mean((y_pred - y) ** 2.)
        elif args.loss_type == 'l1':
            mse = torch.mean(torch.abs(y_pred - y))
        psnr = -10. * torch.log10(mse)

        if warmup:
            gates, importance = model_output['gates'], model_output['importance']
            weights = 1. / torch.pow(torch.full((gates.shape[-1],), args.l1_exp, device=gates.device), \
                torch.arange(gates.shape[-1], device=gates.device))
            sparsity = torch.mean(torch.abs(gates) * weights[None, ...])
            cv_squared = cv_squared_loss(importance)
            loss = mse + args.loss_l1 * sparsity + args.loss_cv * cv_squared
        else:
            load, importance = model_output['load'], model_output['importance']
            cv_squared = cv_squared_loss(load) + cv_squared_loss(importance)
            loss = mse + args.loss_cv * cv_squared

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update()
        pbar.set_description(f'[TRAIN] EPOCH: {current_epoch} LOSS: {loss.item():.4f} MSE: {mse.item():.4f} '
            f'PSNR: {psnr.item():.4f}')

        if (i+len(train_loader)*current_epoch) % args.steps_til_summary == 0:
            with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
                print(f'[TRAIN] EPOCH: {current_epoch} ITER: {i}/{len(train_loader)} LOSS: {loss.item():.4f} '
                    f'MSE: {mse.item():.4f} PSNR: {psnr.item():.4f}', file=f)
            writer.add_scalar('train_loss', loss.item(), len(train_loader) * current_epoch + i)
            writer.add_scalar('train_mse', mse.item(), len(train_loader) * current_epoch + i)
            writer.add_scalar('train_psnr', psnr.item(), len(train_loader) * current_epoch + i)


def train_one_epoch_recursive(model, optimizer, train_loader, coords_loader, args, writer, current_epoch, device, pbar):

    warmup = current_epoch <= args.warmup_epochs
    for i, (rgb, img_ids) in enumerate(train_loader):
        rgb, img_ids = rgb.to(device), img_ids.to(device)
        
        for coords, i_sel in coords_loader:

            coords, i_sel = coords.to(device), i_sel.to(device)

            model_input = {
                'imgs': rgb,
                'img_ids': img_ids,
                'coords': coords
            }

            B, H, W, C = rgb.shape
            y = rgb.reshape(B, -1, C)[:, i_sel, :]

            model.train()
            model_output = model(model_input, topk_sparse=(not warmup))
            y_pred = model_output['preds']
            if args.loss_type == 'l2':
                mse = torch.mean((y_pred - y) ** 2.)
            elif args.loss_type == 'l1':
                mse = torch.mean(torch.abs(y_pred - y))
            psnr = -10. * torch.log10(mse)

            if warmup:
                gates, importance = model_output['gates'], model_output['importance']
                weights = 1. / torch.pow(torch.full((gates.shape[-1],), args.l1_exp, device=gates.device), \
                    torch.arange(gates.shape[-1], device=gates.device))
                sparsity = torch.mean(torch.abs(gates) * weights[None, ...])
                cv_squared = cv_squared_loss(importance)
                loss = mse + args.loss_l1 * sparsity + args.loss_cv * cv_squared
            else:
                load, importance = model_output['load'], model_output['importance']
                cv_squared = cv_squared_loss(load) + cv_squared_loss(importance)
                loss = mse + args.loss_cv * cv_squared

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.update()
        pbar.set_description(f'EPOCH: {current_epoch} LOSS: {loss.item():.4f} MSE: {mse.item():.4f} '
            f'PSNR: {psnr.item():.4f}')

        if (i+len(train_loader)*current_epoch) % args.steps_til_summary == 0:
            psnr = -10. * torch.log10(mse)
            with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
                print(f'[TRAIN] EPOCH: {current_epoch} ITER: {i}/{len(train_loader)} LOSS: {loss.item():.4f} '
                    f'MSE: {mse.item():.4f} PSNR: {psnr.item():.4f}', file=f)
            writer.add_scalar('train_loss', loss.item(), len(train_loader) * current_epoch + i)
            writer.add_scalar('train_mse', mse.item(), len(train_loader) * current_epoch + i)
            writer.add_scalar('train_psnr', psnr.item(), len(train_loader) * current_epoch + i)

def main(args):
    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')

    if args.restart:
        shutil.rmtree(args.log_dir, ignore_errors=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # prepare data loader
    if args.dataset == 'yaleface':
        train_dataset = YaleFaceDataset(root=args.data_dir, subclass='all', split='train')
        test_dataset = train_dataset
        render_dataset = test_dataset
    elif args.dataset == 'olivetti':
        train_dataset = OlivettiFaceDataset(root=args.data_dir, split='train')
        test_dataset = train_dataset
        render_dataset = test_dataset
    elif args.dataset == 'cifar10':
        train_dataset = CIFAR10Dataset(root=args.data_dir, split='train', subset=args.train_subset)
        test_dataset = CIFAR10Dataset(root=args.data_dir, split='test', subset=args.test_subset)
        render_dataset = CIFAR10Dataset(root=args.data_dir, split='test', subset=args.render_subset)
    elif args.dataset == 'celeba':
        train_dataset = CelebADataset(root=args.data_dir, split='train', subset=args.train_subset, downsampled_size=(128, 128), patch_size=(args.patch_size, args.patch_size))
        test_dataset = CelebADataset(root=args.data_dir, split='test', subset=args.test_subset, downsampled_size=(128, 128), patch_size=(args.patch_size, args.patch_size))
        render_dataset = CelebADataset(root=args.data_dir, split='test', subset=args.render_subset, downsampled_size=(128, 128), patch_size=(args.patch_size, args.patch_size))
    elif args.dataset == 'imagefolder':
        train_dataset = ImageFolderDataset(root=os.path.join(args.data_dir, 'train'))
        test_dataset = ImageFolderDataset(root=os.path.join(args.data_dir, 'test'))
        render_dataset = test_dataset
    else:
        raise ValueError(f'Unknown dataset type: {args.dataset}')
    train_loader = DataLoader(train_dataset, pin_memory=True, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, pin_memory=True, batch_size=args.batch_size, shuffle=False)
    render_loader = DataLoader(render_dataset, pin_memory=True, batch_size=args.batch_size, shuffle=False)

    if args.inner_loop == 'recursive':
        lattice_dataset = LatticeDataset(train_dataset.image_size)
        train_coords = DataLoader(lattice_dataset, pin_memory=True, batch_size=args.chunk_size, shuffle=True)
        val_coords = DataLoader(lattice_dataset, pin_memory=True, batch_size=args.chunk_size, shuffle=False)
    else:
        train_coords = IterableLatticeDataset(train_dataset.image_size, batch_size=args.chunk_size, shuffle=True)
        val_coords = IterableLatticeDataset(train_dataset.image_size, batch_size=args.chunk_size, shuffle=False)

    # build model and optimizer
    if args.gate_type == 'codebook':
        gate_module = partial(CodebookImgEncoder, num_images=train_dataset.num_images, max_norm=1., norm_type=2.)
    elif args.gate_type == 'conv':
        gate_module = partial(SimpleConvImgEncoder, input_size=train_dataset.num_channels, num_layers=2, hidden_dim=256)
    elif args.gate_type == 'resnet':
        gate_module = partial(ResConvImgEncoder, input_size=train_dataset.num_channels, image_resolution=min(train_dataset.image_size))
    model = INRMoE(args, in_dim=2, out_dim=train_dataset.num_channels, bias=True, gate_module=gate_module)
    model = model.to(device)

    # freeze dictionary
    if args.finetune:
        model.freeze_dict()

    model_params = model.code_parameters() if args.finetune else model.parameters()
    optimizer = torch.optim.Adam(params=model_params, lr=args.lr, weight_decay=args.weight_decay)

    # load checkpoints
    start_epoch = 0
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
    ckpt_path = args.ckpt_path
    if not ckpt_path and os.path.exists(checkpoint_dir):
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
        time0 = time.time()
        mse, psnr, ssim, lpips = evaluate(model, val_loader, val_coords, args, start_epoch, device)
        print(f'[TEST] EPOCH {start_epoch} MSE: {mse:.4f} PSNR: {psnr:.4f} '
            f'SSIM: {ssim:.4f} LPIPS: {lpips:.4f} Throughput: {len(val_loader.dataset) / (time.time() - time0):4f}')

        save_dir = os.path.join(args.log_dir, f'render_{start_epoch:04d}_testonly')
        os.makedirs(save_dir, exist_ok=True)
        render(model, render_loader, val_coords, args, start_epoch, save_dir, device)
        return
    
    with tqdm(total=len(train_loader) * args.num_epochs) as pbar:

        pbar.update(len(train_loader) * start_epoch)

        for current_epoch in range(start_epoch, args.num_epochs+1):
            if args.inner_loop == 'recursive':
                train_one_epoch_recursive(model, optimizer, train_loader, train_coords, args, writer, current_epoch, device, pbar)
            else:
                train_one_epoch(model, optimizer, train_loader, train_coords, args, writer, current_epoch, device, pbar)

            if current_epoch > 0 and current_epoch % args.epochs_til_eval == 0:
                pbar.set_description('Evaluating ...')
                pbar.refresh()
                mse, psnr, ssim, lpips = evaluate(model, val_loader, val_coords, args, current_epoch, device, pbar)
                with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
                    print(f'[TEST] EPOCH {current_epoch} MSE: {mse:.4f} PSNR: {psnr:.4f} '
                        f'SSIM: {ssim:.4f} LPIPS: {lpips:.4f}', file=f)

            if current_epoch > 0 and current_epoch % args.epochs_til_render == 0:
                pbar.set_description('Rendering ...')
                pbar.refresh()
                save_dir = os.path.join(args.log_dir, f'render_{current_epoch:04d}')
                os.makedirs(save_dir, exist_ok=True)
                render(model, render_loader, val_coords, args, current_epoch, save_dir, device)

            if current_epoch > 0 and current_epoch % args.epochs_til_ckpt == 0:
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
    p.add_argument('--log_dir', type=str, required=True, help='directory path for logging')
    p.add_argument('--ckpt_path', type=str, default='', help='path to load checkpoint')
    p.add_argument('--gpuid', type=int, default=0, help='cuda device number')
    p.add_argument('--restart', action='store_true', help='do not reload from checkpoints')

    # dataset options
    p.add_argument('--dataset', type=str, default='yaleface', choices=['yaleface', 'olivetti', 'cifar10', 'celeba', 'imagefolder'],
            help='dataset type: yaleface, olivetti, or cifar10')
    p.add_argument('--data_dir', type=str, required=True, help='root path for dataset')
    p.add_argument('--train_subset', type=int, default=500, help='subsample rate of training dataset')
    p.add_argument('--test_subset', type=int, default=500, help='subsample rate of testing dataset')
    p.add_argument('--render_subset', type=int, default=100, help='subsample rate of rendering dataset')
    p.add_argument('--patch_size', type=int, default=16, help='patch size to crop datasets')

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
    p.add_argument('--l1_exp', type=float, default=1., help='base for expoential L1 sparsity')
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
    p.add_argument('--gate_type', type=str, default='conv', choices=['conv', 'resnet', 'codebook'],
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
