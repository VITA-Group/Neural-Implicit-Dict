import os
import shutil
import time

from tqdm import tqdm, trange
import argparse
import imageio

from PIL import Image, ImageDraw

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_2d import YaleFaceDataset, OlivettiFaceDataset, CIFAR10Dataset, CelebADataset

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=0)
p.add_argument('--data_dir', type=str, required=True)
p.add_argument('--dataset', type=str, default='yaleface', choices=['yaleface', 'olivetti', 'cifar10', 'celeba'])
p.add_argument('--split', type=str, default='test', choices=['train', 'test'])
p.add_argument('--out_dir', type=str, required=True)
p.add_argument('--perturb_occ', action='store_true', default=False)
p.add_argument('--occ_size', type=int, default=64)
p.add_argument('--occ_stripes', type=int, default=10)
p.add_argument('--perturb_color', action='store_true', default=False)
p.add_argument('--mag_s', type=float, default=0.2)
p.add_argument('--mag_b', type=float, default=0.2)

args = p.parse_args()

def add_perturbation(img_np, perturbation, seed=0, num_channels=3, occ_size=64, occ_stripes=10, mag_s=0.2, mag_b=0.2):
    if 'color' in perturbation:
        np.random.seed(seed)
        s = np.random.uniform(1.0-mag_s, 1.0+mag_s, size=num_channels)
        b = np.random.uniform(-mag_b, mag_b, size=num_channels)
        img_np[..., :num_channels] = np.clip(s*img_np[..., :num_channels]+b, 0, 1)
    if 'occ' in perturbation:
        if num_channels == 1:
            img_np = img_np.squeeze(-1)
        img = Image.fromarray((255*img_np).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        left = np.random.randint(0, img_np.shape[1]-occ_size)
        top = np.random.randint(0, img_np.shape[0]-occ_size)
        w = occ_size / occ_stripes
        for i in range(occ_stripes):
            np.random.seed(occ_stripes*seed+i)
            random_color = tuple(np.random.choice(range(256), num_channels))
            if num_channels == 1:
                random_color = int(random_color[0])
            draw.rectangle(((left+w*i, top), (left+w*(i+1), top+occ_size)), fill=random_color)
        img_np = np.array(img)/255.0
    return img_np

os.makedirs(args.out_dir, exist_ok=True)

if args.dataset == 'yaleface':
    dataset = YaleFaceDataset(root=args.data_dir, subclass='all', split=args.split)
if args.dataset == 'olivetti':
    dataset = OlivettiFaceDataset(root=args.data_dir, split=args.split)
elif args.dataset == 'cifar10':
    dataset = CIFAR10Dataset(root=args.data_dir, split=args.split)
elif args.dataset == 'celeba':
    dataset = CelebADataset(root=args.data_dir, split=args.split)
else:
    raise ValueError(f'Unknown dataset type: {args.dataset}')

for i, data in enumerate(tqdm(dataset)):
    image = data[0].numpy()
    perturbation = []
    if args.perturb_occ:
        perturbation.append('occ')
    if args.perturb_color:
        perturbation.append('color')
    if len(perturbation) > 0:
        image = add_perturbation(image, perturbation, seed=args.seed+i, num_channels=image.shape[-1],
            occ_size=args.occ_size, occ_stripes=args.occ_stripes, mag_s=args.mag_s, mag_b=args.mag_b)
    imageio.imwrite(os.path.join(args.out_dir, f'{i:03d}.png'), (255*image).astype(np.uint8))
