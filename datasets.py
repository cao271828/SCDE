# -*- coding: utf-8 -*-
"""
v4: GPU batch-level augmentation.
    __getitem__ returns raw (C,H,W) patches, no numpy flip/rad.
    flip + radiation moved to GPU in training loop.
"""
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
from scipy.linalg import sqrtm
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
from utils_HSI import open_file
import matplotlib.pyplot as plt
import random

DATASETS_CONFIG = {
    'Houston13': {'img': 'Houston13.mat', 'gt': 'Houston13_7gt.mat'},
    'Houston18': {'img': 'Houston18.mat', 'gt': 'Houston18_7gt.mat'},
    'paviaU':    {'img': 'paviaU.mat',    'gt': 'paviaU_7gt.mat'},
    'paviaC':    {'img': 'paviaC.mat',    'gt': 'paviaC_7gt.mat'},
}

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG
    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)

def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    palette = None
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))
    dataset = datasets[dataset_name]
    folder = target_folder
    if dataset.get('download', False):
        if not os.path.isdir(folder): os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename, reporthook=t.update_to)
    elif not os.path.isdir(folder):
       print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'Houston13':
        img = np.asarray(open_file(folder + 'Houston13.mat')['ori_data']).transpose(1, 2, 0)
        rgb_bands = [13,20,33]
        gt = np.asarray(open_file(folder + 'Houston13_7gt.mat')['map'])
        label_values = ["grass healthy", "grass stressed", "trees", "water",
                        "residential buildings", "non-residential buildings", "road"]
        label_queue = {k: [f'{k} desc1', f'{k} desc2'] for k in label_values}
        ignored_labels = [0]
    elif dataset_name == 'Houston18':
        img = np.asarray(open_file(folder + 'Houston18.mat')['ori_data']).transpose(1, 2, 0)
        rgb_bands = [13,20,33]
        gt = np.asarray(open_file(folder + 'Houston18_7gt.mat')['map'])
        label_values = ["grass healthy", "grass stressed", "trees", "water",
                        "residential buildings", "non-residential buildings", "road"]
        label_queue = {k: [f'{k} desc1', f'{k} desc2'] for k in label_values}
        ignored_labels = [0]
    elif dataset_name == 'paviaU':
        img = np.asarray(open_file(folder + 'paviaU.mat')['paviaU'])[:, :, :102]
        rgb_bands = [20,30,30]
        gt = np.asarray(open_file(folder + 'paviaU_7gt.mat')['map'])
        label_values = ["tree", "asphalt", "brick", "bitumen", "shadow", 'meadow', 'bare soil']
        label_queue = {k: [f'{k} desc1', f'{k} desc2'] for k in label_values}
        ignored_labels = [0]
    elif dataset_name == 'paviaC':
        img = np.asarray(open_file(folder + 'paviaC.mat')['ori_data']).transpose(1, 2, 0)
        rgb_bands = [20,30,30]
        gt = np.asarray(open_file(folder + 'paviaC_7gt.mat')['map'])
        label_values = ["tree", "asphalt", "brick", "bitumen", "shadow", 'meadow', 'bare soil']
        label_queue = {k: [f'{k} desc1', f'{k} desc2'] for k in label_values}
        ignored_labels = [0]
    else:
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](folder)
        label_queue = {k: [f'{k} desc1', f'{k} desc2'] for k in label_values}

    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN found, disabled.")
    img[nan_mask] = 0; gt[nan_mask] = 0
    ignored_labels.append(0); ignored_labels = list(set(ignored_labels))

    img = np.asarray(img, dtype='float32')
    m, n, d = img.shape[0], img.shape[1], img.shape[2]
    img = img.reshape((m*n,-1))
    img = img/img.max()
    img_temp = np.sqrt(np.asarray((img**2).sum(1)))
    img_temp = np.expand_dims(img_temp, axis=1)
    img_temp = img_temp.repeat(d, axis=1)
    img_temp[img_temp==0] = 1
    img = img/img_temp
    img = np.reshape(img, (m, n, -1))
    return img, gt, label_values, label_queue, ignored_labels, rgb_bands, palette


class HyperX(torch.utils.data.Dataset):
    def __init__(self, data, gt, transform=None, **hyperparams):
        super(HyperX, self).__init__()
        self.transform = transform
        self.data = data          # (H, W, C)
        self.label = gt
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        self.mixture_augmentation = hyperparams['mixture_augmentation']
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos)
                                 if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]

        # --- Precompute patches as (C, H, W) float32 ---
        cache_patches = hyperparams.get('cache_patches', True)
        if cache_patches and self.patch_size > 1 and len(self.indices) > 0:
            n = len(self.indices)
            ps = self.patch_size
            bands = data.shape[2]
            self.cached_patches = np.zeros((n, bands, ps, ps), dtype='float32')
            print(f"[HyperX] Precomputing {n} patches ({bands}x{ps}x{ps}) ...")
            for idx in tqdm(range(n), desc='Cache patches'):
                x, y = self.indices[idx]
                x1, y1 = x - p, y - p
                x2, y2 = x1 + ps, y1 + ps
                self.cached_patches[idx] = data[x1:x2, y1:y2].transpose(2, 0, 1)
            nbytes = self.cached_patches.nbytes / (1024**3)
            print(f"[HyperX] Cached {n} patches ({nbytes:.2f} GB)")
        else:
            self.cached_patches = None

    def mixture_noise_chw(self, data, x_idx, y_idx, beta=1/25):
        """data: (C, H, W). Uses self.data (H,W,C) for pixel-level mixing."""
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        ps = self.patch_size
        p = ps // 2
        x1, y1 = x_idx - p, y_idx - p
        for di in range(ps):
            for dj in range(ps):
                gi, gj = x1 + di, y1 + dj
                lbl = self.label[gi, gj]
                if lbl not in self.ignored_labels:
                    l_indices = np.nonzero(np.array(self.labels) == lbl)[0]
                    if len(l_indices) > 0:
                        l_idx = np.random.choice(l_indices)
                        sx, sy = self.indices[l_idx]
                        data2[:, di, dj] = self.data[sx, sy]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]

        if self.cached_patches is not None:
            # v4: raw patch — flip + rad moved to GPU batch-level
            data = self.cached_patches[i]  # (C, H, W)

            if self.mixture_augmentation and np.random.random() < 0.5:
                data = self.mixture_noise_chw(data, x, y)
                data = np.ascontiguousarray(data).astype('float32', copy=False)

            data = torch.from_numpy(data)  # zero-copy reference
        else:
            # Fallback: original code path (H, W, C)
            x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
            x2, y2 = x1 + self.patch_size, y1 + self.patch_size
            data = self.data[x1:x2, y1:y2]
            label = self.label[x1:x2, y1:y2]
            if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
                horizontal = np.random.random() > 0.5; vertical = np.random.random() > 0.5
                if horizontal: data = np.fliplr(data); label = np.fliplr(label)
                if vertical: data = np.flipud(data); label = np.flipud(label)
            if self.radiation_augmentation and np.random.random() < 0.5:
                data = self.radiation_noise(data)
            if self.mixture_augmentation and np.random.random() < 0.5:
                alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
                noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
                data2 = np.zeros_like(data)
                for idx, value in np.ndenumerate(label):
                    if value not in self.ignored_labels:
                        l_indices = np.nonzero(np.array(self.labels) == value)[0]
                        l_indice = np.random.choice(l_indices)
                        x_s, y_s = self.indices[l_indice]
                        data2[idx] = self.data[x_s, y_s]
                data = (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise
            data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
            label = np.asarray(np.copy(label), dtype='int64')
            data = torch.from_numpy(data)
            label = torch.from_numpy(label)
            if self.center_pixel and self.patch_size > 1:
                label = label[self.patch_size // 2, self.patch_size // 2]
            elif self.patch_size == 1:
                data = data[:, 0, 0]; label = label[0, 0]
            else:
                label = self.labels[i]
            return data, label

        # Label (same for cached path)
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        label = self.label[x1:x2, y1:y2]
        label = np.asarray(np.copy(label), dtype='int64')
        label = torch.from_numpy(label)
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        elif self.patch_size == 1:
            data = data[:, 0, 0]; label = label[0, 0]
        else:
            label = self.labels[i]
        return data, label

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
    def preload(self):
        try:
            self.data, self.label = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.data = self.data.cuda(non_blocking=True)
            self.label = self.label.cuda(non_blocking=True)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data; label = self.label
        self.preload()
        return data, label
