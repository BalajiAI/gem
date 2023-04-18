import h5py
import config
from tqdm import tqdm
import cv2
import imageio
from imageio import imread, imwrite
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from skimage.transform import resize
from multiprocessing.pool import Pool
import random
import pickle as pck
from glob import glob
import pandas as pd
import utils
import pickle
import io
from copy import deepcopy
from torchvision.datasets.utils import download_file_from_google_drive
from tqdm.autonotebook import tqdm
from torchmeta.datasets.utils import get_asset
import pandas as pd
import os.path as osp

import torch.utils.data as data
import os
import os.path
import errno
import h5py
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop,ToPILImage
import torchvision
from torchvision.transforms import functional as trans_fn

from sklearn.model_selection import train_test_split
from multiprocessing import Pool


class CelebAHQGAN(torch.utils.data.Dataset):
    def __init__(self, sidelength=64, cache=None, cache_mask=None, split='train', subset=None):
        self.name = 'celebahq'
        # This is just a toy dataloader
        self.sidelength = sidelength
        subsample = 256 // self.sidelength
        self.mgrid = utils.get_mgrid((256, 256), dim=2, subsample=subsample)
        self.im_size = sidelength

    def __len__(self):
        return 27000

    def read_frame(self, path, item):
        frame = torch.zeros(64*64, 3)
        return frame

    def __getitem__(self, item):
        rgb = self.read_frame("", item)

        query_dict = {"idx": torch.Tensor([item]).long()}
        return {'context': query_dict, 'query': query_dict}, query_dict


class Cifar10(Dataset):
    def __init__(
            self,
            sidelength=1024,
            cache=None,
            cache_mask=None,
            split='train',
            subset=None):

        self.name = 'celebahq'
        self.channels = 3
        self.sidelength = sidelength

        cache = np.ctypeslib.as_array(cache.get_obj())
        cache = cache.reshape(50000, 3, 32, 32)
        cache = cache.astype("uint8")
        self.cache = torch.from_numpy(cache)

        self.im_size = 32

        cache_mask = np.ctypeslib.as_array(cache_mask.get_obj())
        cache_mask = cache_mask.reshape(50000)
        cache_mask = cache_mask.astype("uint8")
        self.cache_mask = torch.from_numpy(cache_mask)

        transform = transforms.ToTensor()

        self.data = CIFAR10(
            "data/cifar10",
            transform=transform,
            train=True,
            download=True)
        self.test_data = CIFAR10(
            "data/cifar10",
            transform=transform,
            train=False,
            download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.cache_mask is not None:
            # cache, cache_mask = self.generate_array()
            if self.cache_mask[index] == 0:
                im, _ = self.data[index]
                frame = (im * 255).numpy()
                self.cache[index] = torch.from_numpy(frame.astype(np.uint8))
                self.cache_mask[index] = 1

            frame = np.array(self.cache[index])
        else:
            im, label = self.data[index]
            frame = (im * 255).numpy()

        frame = torch.from_numpy(frame).float()
        frame /= 255.
        frame -= 0.5
        frame *= 2.
        # np.random.seed((index + int(time.time() * 1e7)) % 2**32)

        return_dict = {'rgb': frame}

        return return_dict

class CelebAHQ(torch.utils.data.Dataset):
    def __init__(self, sidelength=1024, cache=None, cache_mask=None, split='train', subset=None):
        self.name = 'celebahq'
        self.channels = 3
        self.im_size = 64

        if split == "train":
            self.im_paths = sorted(glob('/data/vision/billf/scratch/yilundu/dataset/celebahq/celebahq_train/data128x128_small/*.jpg'))
        else:
            self.im_paths = sorted(glob('/data/vision/billf/scratch/yilundu/dataset/celebahq/celebahq_test/*.jpg'))

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, item):
        path = self.im_paths[item]
        rgb = imread(path)
        rgb = resize(rgb, (64, 64))
        rgb = (torch.Tensor(rgb).float() - 0.5) * 2
        rgb = rgb.permute(2, 0, 1)

        return {"rgb":rgb}


class CelebA(torch.utils.data.Dataset):
    def __init__(self, sidelength, split='train'):
        transform = Compose([
            Resize((sidelength, sidelength)),
            CenterCrop((sidelength, sidelength)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.transform = transform
        self.path = "/datasets01/CelebA/CelebA/072017/img_align_celeba/"
        self.labels = pd.read_csv("/private/home/yilundu/list_attr_celeba.txt", sep="\s+", skiprows=1)
        self.name = 'celeba'

        self.channels = 3
        self.mgrid = utils.get_mgrid(sidelength, dim=2)
        self.sidelength = sidelength

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        info = self.labels.iloc[index]
        fname = info.name
        path = osp.join(self.path, fname)
        im = self.transform(Image.open(path))

        return {"rgb": im}


class GeneralizationWrapper(torch.utils.data.Dataset):
    '''Assumes that 2D modalities from dataset are ordered (ch, height, width)'''
    def __init__(self, dataset, context_sparsity, query_sparsity, sparsity_range=(10, 200),
                 inner_loop_supervision_key=None, persistent_mask=False, idx_offset=0, padding=False,
                 cache=None):
        self.dataset = dataset
        self.im_size = self.dataset.im_size

        img = self.dataset[0]["rgb"]
        self.sidelength = img.shape[-1]

        # subsample = 256 // self.sidelength
        # self.mgrid = utils.get_mgrid((256, 256), dim=len(img.shape)-1, subsample=subsample)
        self.mgrid = utils.get_mgrid((self.im_size, self.im_size), dim=len(img.shape)-1)

        self.per_key_channels = {key:value.shape[0] for key, value in self.dataset[0].items()}
        self.padding = padding
        if 'semantic' in list(self.dataset[0].keys()):
            self.total_channels = len(self.dataset.img_dataset.classes)
            self.total_channels += int(np.sum([value for key, value in self.per_key_channels.items() if key != 'semantic']))
        else:
            self.total_channels = int(np.sum(list(self.per_key_channels.values())))

        self.per_key_channels['x'] = 2

        self.persistent_mask = persistent_mask
        # self.mask_dir = os.path.join('/om2/user/sitzmann/', self.dataset.name + f'_masks_{padding}_{sparsity_range[0]}_{sparsity_range[1]}_{self.sidelength}')
        self.mask_dir = os.path.join('/tmp/'+ f'_masks_{padding}_{sparsity_range[0]}_{sparsity_range[1]}_{self.sidelength}')
        # if not os.path.exists(self.mask_dir):
        #     os.mkdir(self.mask_dir)

        self.context_sparsity = context_sparsity
        self.query_sparsity = query_sparsity
        self.sparsity_range = sparsity_range
        self.inner_loop_supervision_key = inner_loop_supervision_key
        self.idx_offset = idx_offset

        self.cache = cache

    def __len__(self):
        return len(self.dataset)

    def flatten_dict_entries(self, dict):
        return {key: value.permute(1, 2, 0).reshape(-1, self.per_key_channels[key]) for key, value in dict.items()}

    def sparsify(self, sample_dict, mgrid, sparsity, idx):
        if sparsity == 'full':
            result_dict = sample_dict
            result_dict = self.flatten_dict_entries(result_dict)
            result_dict['x'] = mgrid
            result_dict['mask'] = torch.ones_like(result_dict['x'][...,:1])
            return result_dict
        elif sparsity == 'half':
            result_dict = {key: value[:, :, :self.sidelength // 2] for key, value in sample_dict.items()}
            mgrid = mgrid.view(self.sidelength, self.sidelength, 2).permute(2, 0, 1)
            result_dict['x'] = mgrid[:, :, :self.sidelength // 2]
            mask = torch.ones_like(mgrid[:1, ...])
            mask = mask[:, :, :self.sidelength // 2].contiguous()
            result_dict = self.flatten_dict_entries(result_dict)
            result_dict['mask'] = mask.view(-1, 1)
            return result_dict
        elif sparsity == 'context':
            mask = np.ones((self.sidelength, self.sidelength)).astype(np.bool)
            mask[16:48, 16:48] = 0
            result_dict = {key: value[:, mask].transpose(1, 0).contiguous() for key, value in sample_dict.items()}
            nelem = result_dict['rgb'].shape[0]
            rix = np.random.permutation(nelem)[:1024]

            result_dict = {key: value[rix] for key, value in result_dict.items()}
            # mgrid = mgrid.view(self.sidelength, self.sidelength, 2).permute(2, 0, 1)

            result_dict['x'] = mgrid[mask.flatten()][rix]
            result_dict['mask'] = mask[mask==1][rix, None]
            return result_dict
        elif sparsity == 'sampled':
            if self.sparsity_range[0] == self.sparsity_range[1]:
                subsamples = self.sparsity_range[0]
            else:
                subsamples = np.random.randint(self.sparsity_range[0], self.sparsity_range[1])

            if not self.padding:
                # Sample upper_limit pixel idcs at random.
                lower_rand_idcs = np.random.choice(self.sidelength ** 2, size=self.sparsity_range[1], replace=False)
                upper_rand_idcs = np.random.choice(self.sparsity_range[1], size=subsamples, replace=False)

                mask_filepath = os.path.join(self.mask_dir, "%09d"%idx)
                if self.persistent_mask:
                    if not os.path.exists(mask_filepath):
                        with open(mask_filepath, 'wb') as mask_file:
                            pck.dump((lower_rand_idcs, upper_rand_idcs), mask_file)
                    else:
                        with open(mask_filepath, 'rb') as mask_file:
                            lower_rand_idcs, upper_rand_idcs = pck.load(mask_file)

                flat_dict = self.flatten_dict_entries(sample_dict)
                result_dict = {key: value[lower_rand_idcs] for key, value in flat_dict.items()}

                result_dict['mask'] = torch.zeros(self.sparsity_range[1], 1)
                result_dict['mask'][upper_rand_idcs, 0] = 1.
                result_dict['x'] = mgrid.view(-1, 2)[lower_rand_idcs, :]
                return result_dict
            else:
                rand_idcs = np.random.choice(self.sidelength**2, size=subsamples, replace=False)
                mask_filepath = os.path.join(self.mask_dir, "%09d"%idx)
                if self.persistent_mask:
                    if not os.path.exists(mask_filepath):
                        with open(mask_filepath, 'wb') as mask_file:
                            pck.dump(rand_idcs, mask_file)
                    else:
                        with open(mask_filepath, 'rb') as mask_file:
                            rand_idcs = pck.load(mask_file)

                result_dict = self.flatten_dict_entries(sample_dict)
                result_dict['mask'] = torch.zeros(self.sidelength**2, 1)
                result_dict['mask'][rand_idcs, 0] = 1.
                result_dict['x'] = mgrid.view(-1, 2)
                return result_dict

    def __getitem__(self, idx):
        if self.cache is not None:
            if idx not in self.cache:
                self.cache[idx] = self.dataset[idx]
            # else:
            #     print('used cache')

            sample_dict = self.cache[idx]
        else:
            sample_dict = self.dataset[idx]


        idx_other = random.randint(0, len(self.dataset) - 1)
        if self.cache is not None:
            if idx not in self.cache:
                self.cache[idx_other] = self.dataset[idx_other]
            # else:
            #     print('used cache')

            sample_dict_other = self.cache[idx_other]
        else:
            sample_dict_other = self.dataset[idx_other]

        mgrid = self.mgrid
        dist_mse = (sample_dict_other['rgb'].reshape(-1) - sample_dict['rgb'].reshape(-1)).pow(2).mean()
        ctxt_dict = self.sparsify(sample_dict, mgrid, self.context_sparsity, idx)
        query_dict = self.sparsify(sample_dict, mgrid, self.query_sparsity, idx)

        if self.inner_loop_supervision_key is not None:
            ctxt_dict['y'] = ctxt_dict[self.inner_loop_supervision_key]
            query_dict['y'] = query_dict[self.inner_loop_supervision_key]

        ctxt_dict['idx'] = torch.Tensor([idx]).long() + self.idx_offset
        query_dict['idx'] = torch.Tensor([idx]).long() + self.idx_offset

        ctxt_dict['idx_other'] = torch.Tensor([idx_other]).long() + self.idx_offset
        query_dict['idx_other'] = torch.Tensor([idx_other]).long() + self.idx_offset
        query_dict['mse'] = dist_mse
        ctxt_dict['mse'] = dist_mse

        query_dict = ctxt_dict

        return {'context': ctxt_dict, 'query': query_dict}, query_dict


class ImplicitGANDataset():
    def __init__(self, real_dataset, fake_dataset):
        self.real_dataset = real_dataset
        self.fake_dataset = fake_dataset
        self.im_size = 64

    def __len__(self):
        return len(self.fake_dataset)

    def __getitem__(self, idx):
        real = self.real_dataset[idx]
        fake = self.fake_dataset[idx]
        return fake, real


class DatasetSampler(torch.utils.data.sampler.Sampler):
    # modified from: https://stackoverflow.com/questions/57913825/how-to-select-specific-labels-in-pytorch-mnist-dataset
    def __init__(self, mask, data_source):
        # mask is a tensor specifying which classes to include
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)
