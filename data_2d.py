import sys, os
import torch
import torchvision
import numpy as np

from PIL import Image
import imageio
import csv

from sklearn.datasets import fetch_olivetti_faces

class LatticeDataset(torch.utils.data.Dataset):

    def __init__(self, image_shape):
        super().__init__()

        self.mgrid = self.get_2d_mgrid(image_shape)

    def get_2d_mgrid(self, shape):
        pixel_coords = np.stack(np.mgrid[:shape[0], :shape[1]], axis=-1).astype(np.float32)

        # normalize pixel coords onto [-1, 1]
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(shape[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / max(shape[1] - 1, 1)
        pixel_coords -= 0.5
        pixel_coords *= 2.
        # flatten 
        pixel_coords = torch.tensor(pixel_coords).view(-1, 2)

        return pixel_coords

    def __len__(self):
        return self.mgrid.shape[0]
    
    def __getitem__(self, idx):
        return self.mgrid[idx], torch.tensor(idx, dtype=torch.int64)

class IterableLatticeDataset(torch.utils.data.IterableDataset, LatticeDataset):

    def __init__(self, image_shape, batch_size, shuffle=False):

        torch.utils.data.IterableDataset.__init__(self)
        LatticeDataset.__init__(self, image_shape)

        self.mgrid = self.get_2d_mgrid(image_shape)
        self.batch_size = batch_size
        self.randomize = shuffle

        self.perm = torch.arange(self.mgrid.shape[0])
        self.current_idx = self.perm.shape[0]
        
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

        yield coords, i_sel

class YaleFaceDataset(torch.utils.data.Dataset):

    __subclass_names__ = ['all', 'glasses', 'happy', 'leftlight', 'noglasses',
        'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']

    def __init__(self, root, subclass='all', split='all'):
        super().__init__()

        assert subclass in self.__subclass_names__, f'Unrecognized sub-class name: {subclass}'

        images = []
        for filename in os.listdir(root):
            if filename.endswith('.gif') or filename.endswith('.txt'):
                continue
            if subclass != 'all' and not filename.endswith('.'+subclass):
                continue
            img = imageio.imread(os.path.join(root, filename)) # [H, W]
            img = img.astype(np.float32) / 255.
            img = torch.from_numpy(img[..., None]) # [H, W, 1]
            images.append(img)

        self.images = torch.stack(images, dim=0) # [N, H, W, 1]

        split_idx = {}
        split_idx['all'] = list(range(self.images.shape[0]))
        split_idx['test'] = list(range(1, self.images.shape[0], 8))
        split_idx['train'] = [i for i in split_idx['all'] if i not in split_idx['test']]
        self.images = self.images[split_idx[split]]

    @property
    def num_images(self):
        return self.images.shape[0]

    @property
    def num_channels(self):
        return self.images.shape[-1]

    @property
    def image_size(self):
        return tuple(self.images.shape[1:3])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(idx, dtype=torch.int64)

class OlivettiFaceDataset(torch.utils.data.Dataset):

    def __init__(self, root, split='all'):
        super().__init__()

        assert split in ['all', 'train', 'test']

        olivetti = fetch_olivetti_faces(data_home=root)
        self.images = torch.from_numpy(olivetti.images)
        self.images = self.images[..., None] # [N, H, W, C=1]

        split_idx = {}
        split_idx['all'] = list(range(self.images.shape[0]))
        split_idx['test'] = list(range(1, self.images.shape[0], 8))
        split_idx['train'] = [i for i in split_idx['all'] if i not in split_idx['test']]
        self.images = self.images[split_idx[split]]

    @property
    def num_images(self):
        return self.images.shape[0]

    @property
    def num_channels(self):
        return self.images.shape[-1]

    @property
    def image_size(self):
        return tuple(self.images.shape[1:3])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(idx, dtype=torch.int64)

class CIFAR10Dataset(torch.utils.data.Dataset):

    def __init__(self, root, split='train'):
        super().__init__()

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            TransposeTransform(in_fmt='CHW', out_fmt='HWC')
        ])
        self.images = torchvision.datasets.CIFAR10(root=root, train=(split == 'train'),
                download=True, transform=transforms)
        # dataset = torchvision.datasets.CIFAR10(root=root, train=(split == 'train'),
        #         download=True, transform=transforms)
        # subset = np.linspace(0, len(dataset)-1, 100, dtype=np.int64)
        # self.images = torch.utils.data.Subset(dataset, subset)

    @property
    def num_images(self):
        return len(self.images)

    @property
    def num_channels(self):
        return 3

    @property
    def image_size(self):
        return (32, 32)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # return torch.mean(self.images[idx][0], dim=-1, keepdim=True), torch.tensor(idx, dtype=torch.int64)
        return self.images[idx][0], torch.tensor(idx, dtype=torch.int64)

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root, split, subset=-1, downsampled_size=None):
        # SIZE (178 x 218)
        super().__init__()
        assert split in ['train', 'test', 'val']

        self.img_dir = os.path.join(root, 'img_align_celeba')
        self.img_channels = 3
        self.file_names = []

        with open(os.path.join(root, 'list_eval_partition.txt'), newline='') as csvfile:
            rowreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in rowreader:
                if split == 'train' and row[1] == '0':
                    self.file_names.append(row[0])
                elif split == 'val' and row[1] == '1':
                    self.file_names.append(row[0])
                elif split == 'test' and row[1] == '2':
                    self.file_names.append(row[0])
        if isinstance(subset, int):
            if subset > 0:
                self.file_names = self.file_names[:subset]
        elif isinstance(subset, list):
            self.file_names = [self.file_names[i] for i in subset]

        self.downsampled_size = downsampled_size

    @property
    def num_images(self):
        return len(self.file_names)

    @property
    def num_channels(self):
        return 3

    @property
    def image_size(self):
        return (178, 178) if self.downsampled_size is None else self.downsampled_size

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.file_names[idx])
        assert os.path.exists(path), 'Index does not specify any images in the dataset'
        
        img = Image.open(path)

        width, height = img.size  # Get dimensions

        s = min(width, height)
        left = (width - s) / 2
        top = (height - s) / 2
        right = (width + s) / 2
        bottom = (height + s) / 2
        img = img.crop((left, top, right, bottom))

        if self.downsampled_size is not None:
            img = img.resize(self.downsampled_size)

        img = np.asarray(img).astype(np.float32) / 255.

        return torch.from_numpy(img), torch.tensor(idx, dtype=torch.int64)

class ImageFolderDataset(torch.utils.data.Dataset):

    __support_formats__ = ['.jpg', '.jpeg', '.png', '.gif', '.bmp'] 

    def __init__(self, root):
        super().__init__()

        if root.endswith('.npy'):
            images = []
            for filename in sorted(os.listdir(root)):
                _, suffix = os.path.splitext(filename)
                if suffix.lower() not in self.__support_formats__:
                    continue
                img = imageio.imread(os.path.join(root, filename)) # [H, W]
                img = img.astype(np.float32) / 255.
                if img.ndim == 2:
                    img = img[..., None] # [H, W] -> [H, W, 1]
                img = torch.from_numpy(img) # [H, W, C]
                images.append(img)
            self.images = torch.stack(images, dim=0) # [N, H, W, C]
        else:
            images = np.load(root)
            if img.ndim == 3:
                img = img[..., None] # [N, H, W] -> [N, H, W, 1]
            self.images = torch.from_numpy(images) # [N, H, W, C]

    @property
    def num_images(self):
        return self.images.shape[0]

    @property
    def num_channels(self):
        return self.images.shape[-1]

    @property
    def image_size(self):
        return tuple(self.images.shape[1:3])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(idx, dtype=torch.int64)

class CTSheppDataset(torch.utils.data.Dataset):

    def __init__(self, root, image_only=False):
        super().__init__()

        images = np.load(os.path.join(root, 'images.npy'))
        if images.ndim == 3:
            images = images[..., None] # [N, H, W] -> [N, H, W, 1]
        self.images = torch.from_numpy(images) # [N, H, W, C]

    @property
    def num_images(self):
        return self.images.shape[0]

    @property
    def num_channels(self):
        return self.images.shape[-1]

    @property
    def image_size(self):
        return tuple(self.images.shape[1:3])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(idx, dtype=torch.int64)

class CTSheppDataset(torch.utils.data.Dataset):

    def __init__(self, root, image_only=False, slices=None):
        super().__init__()

        images = np.load(os.path.join(root, 'images.npy'))
        if images.ndim == 3:
            images = images[..., None] # [N, H, W] -> [N, H, W, 1]
        self.images = torch.from_numpy(images).float() # [N, H, W, C]
        if slices is not None:
            self.images = self.images[slices]

    @property
    def num_images(self):
        return self.images.shape[0]

    @property
    def num_channels(self):
        return self.images.shape[-1]

    @property
    def image_size(self):
        return tuple(self.images.shape[1:3])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(idx, dtype=torch.int64)

class CTSliceDataset(torch.utils.data.Dataset):

    def __init__(self, ct_size, num_thetas):
        super().__init__()

        thetas = torch.linspace(0.0, np.pi, num_thetas)[:, None, None]
        x, y = np.mgrid[:ct_size[0], :ct_size[1]]
        x = torch.from_numpy((x / max(ct_size[0] - 1, 1) - 0.5) * 2.).float()
        y = torch.from_numpy((y / max(ct_size[1] - 1, 1) - 0.5) * 2.).float()

        x, y = x[None, ...], y[None, ...] # [N, H, W]
        x_rot = x * torch.cos(thetas) - y * torch.sin(thetas) # [N, H, W]
        y_rot = x * torch.sin(thetas) + y * torch.cos(thetas) # [N, H, W]

        self.N, self.H, self.W = x_rot.shape
        sample_coords = torch.stack([x_rot, y_rot], dim=-1) # [N, H, W, 2]
        self.sample_coords = sample_coords.reshape([-1, self.W, 2]) # [NxH, W, 2]

    @property
    def num_thetas(self):
        return self.N

    @property
    def image_size(self):
        return tuple(self.H, self.W)

    def __len__(self):
        return self.sample_coords.shape[0]

    def __getitem__(self, idx):
        return self.sample_coords[idx], torch.tensor(idx, dtype=torch.int64)

class ImageDataset(LatticeDataset):
    def __init__(self, image, crop=False):

        if isinstance(image, str):
            if image.endswith('.npy'):
                image = np.load(image)
            else:
                image = np.asarray(imageio.imread(image)).astype(np.float32) / 255.

        if crop:
            height, width = image.shape[:2]  # Get dimensions
            s = min(width, height)
            left = (width - s) // 2
            top = (height - s) // 2
            right = (width + s) // 2
            bottom = (height + s) // 2
            image = image[top:bottom, left:right]

        super().__init__(image.shape[:-1])

        self.image_size = image.shape[:2]
        self.num_channels = image.shape[-1]
        self.data = image.reshape(-1, image.shape[-1])

        assert self.data.shape[0] == self.mgrid.shape[0]

    def __getitem__(self, idx):
        coords, isel = super().__getitem__(idx)
        return coords, self.data[idx], isel

class TransposeTransform:

    def __init__(self, in_fmt='CHW', out_fmt='HWC'):
        self.order = [in_fmt.index(c) for c in out_fmt]

    def __call__(self, img):
        return img.permute(self.order)