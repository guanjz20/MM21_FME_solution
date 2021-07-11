from unicodedata import name
import cv2
import os
import pdb
import torch
import time
import pywt
import glob
import numpy as np
import os.path as osp
from tqdm import tqdm
from torch.utils.data import Dataset
from torch import nn as nn

from . import params
from . import utils

WT_CHANNEL = 4
sm_kernel = np.load(params.GAUSS_KERNEL_PATH['sm_kernel'])
dr1_kernel = np.load(params.GAUSS_KERNEL_PATH['dr1_kernel'])
dr2_kernel = np.load(params.GAUSS_KERNEL_PATH['dr2_kernel'])
dr1_kernel = dr1_kernel[:, None, None]
dr2_kernel = dr2_kernel[:, None, None]


class SAMMDataset(Dataset):
    def __init__(self,
                 mode,
                 img_dirs,
                 seq_len=64,
                 step=32,
                 time_len=12,
                 input_size=256,
                 data_aug=False,
                 data_option=None,
                 dataset_name='SAMM'):
        super().__init__()
        self.dataset_name = dataset_name
        self.mode = mode
        self.seq_len = seq_len
        self.step = step
        assert mode == 'train' or (mode == 'test'
                                   and self.seq_len <= self.step)

        self.time_len = time_len  # observate time_len//2 frames before and after
        self.size = input_size if data_option == 'diff' else input_size * 2
        self.img_dirs = img_dirs  # imgs files dirs
        if not isinstance(self.img_dirs, list):
            self.img_dirs = [self.img_dirs]
        self.img_ps_dict = self._get_img_ps_dict()
        self.seq_list = self._get_seq_list()
        self.label_dict = np.load(osp.join(params.SAMM_ROOT, 'label_dict.npy'),
                                  allow_pickle=True).item()
        self.anno_dict = np.load(osp.join(params.SAMM_ROOT, 'anno_dict.npy'),
                                 allow_pickle=True).item()
        # print('Load {} clips, {} frames from {}'.format(
        #     len(self.seq_list),
        #     len(self.seq_list) * self.seq_len, dataset_name))

        self.transform = utils.get_group_transform(
            mode) if data_aug else utils.Identity()
        self.data_option = data_option

    def _get_img_ps_dict(self):
        ret_dict = {}
        for img_dir in self.img_dirs:
            img_ps = utils.scan_jpg_from_img_dir(img_dir)
            ret_dict[img_dir] = tuple(img_ps)
        return ret_dict

    def _get_seq_list(self):
        ret_list = []
        for img_dir, img_ps in self.img_ps_dict.items():
            front = 0
            tail = front + self.seq_len  # [front, tail), tail not include
            while tail <= len(img_ps):
                ret_list.append([img_dir, front,
                                 tail])  # (img dir, front_idx, tail_idx)
                front += self.step
                tail = front + self.seq_len
        return ret_list

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        img_dir, front, tail = self.seq_list[
            index]  # [front, tail), tail not include
        seq_info = (img_dir, front, tail)

        # insert and append extra imgs for temporal conv
        _old_len = len(self.img_ps_dict[img_dir])
        img_ps = list(self.img_ps_dict[img_dir][front:tail])
        for i in range(1, self.time_len // 2 + 1):
            img_ps.insert(0, self.img_ps_dict[img_dir][max(0, front - i)])
            img_ps.append(self.img_ps_dict[img_dir][min(
                _old_len - 1, tail - 1 + i)])
        _cur_len = len(self.img_ps_dict[img_dir])
        assert _old_len == _cur_len  # make sure the dict has not been changed

        # read seqence features, annos and labels
        img_features = np.stack([
            np.load(p.replace('.jpg', '.npy'))
            for p in img_ps[self.time_len // 2:-self.time_len // 2]
        ], 0)
        annos = self.anno_dict[img_dir][front:tail]
        labels = self.label_dict[img_dir][front:tail]
        assert img_features.shape == (self.seq_len, 2048)  # resnet50 features

        # read sequence imgs
        flat_imgs = np.empty(
            (self.seq_len + (self.time_len // 2) * 2, self.size, self.size),
            dtype=np.float32)
        for i, p in enumerate(img_ps):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if not img.shape[0] == img.shape[1]:
                # crop to square
                h, w = img.shape
                wide = abs(h - w) // 2
                if h > w:
                    img = img[wide:wide + w, :]
                else:
                    img = img[:, wide:wide + h]
            try:
                assert img.shape[0] == img.shape[1]
            except:
                print('Error in cropping image {}'.format(p))
            img = cv2.resize(img, (self.size, self.size))
            flat_imgs[i] = img

        # transform
        flat_imgs = self.transform(flat_imgs)
        if self.data_option is not None and 'wt' in self.data_option:
            flat_wts = np.stack([dwt2(img) for img in flat_imgs], 0)

        # expand falt imgs
        i = 0
        front = 0
        tail = front + self.time_len  # [front, tail], tail include
        if self.data_option is not None and 'wt' in self.data_option:
            seq_wts = np.empty((self.seq_len, self.time_len + 1, WT_CHANNEL,
                                self.size // 2, self.size // 2),
                               dtype=np.float32)
        elif self.data_option == 'diff':
            seq_imgs = np.empty(
                (self.seq_len, self.time_len + 1, self.size, self.size),
                dtype=np.float32)
        while tail < len(flat_imgs):
            if self.data_option is not None and 'wt' in self.data_option:
                seq_wts[i] = flat_wts[front:tail + 1].copy()
            elif self.data_option == 'diff':
                seq_imgs[i] = flat_imgs[front:tail + 1].copy()
            i += 1
            front += 1
            tail += 1
        assert i == self.seq_len

        # data options
        if self.data_option == 'diff':
            ret_coefs = np.stack([get_diff(imgs) for imgs in seq_imgs], 0)
        elif self.data_option == 'wt_diff':
            ret_coefs = np.stack([get_diff(coefs) for coefs in seq_wts],
                                 0).reshape(self.seq_len,
                                            self.time_len * WT_CHANNEL,
                                            self.size // 2, self.size // 2)
        elif self.data_option == 'wt_dr':
            ret_coefs = seq_wts.transpose(0, 2, 1, 3, 4)
            ret_coefs = np.asarray([[
                get_smoothing_and_dr_coefs(coefs_dim2)
                for coefs_dim2 in coefs_dim1
            ] for coefs_dim1 in ret_coefs])
            assert ret_coefs.shape[:3] == (self.seq_len, WT_CHANNEL, 3 * 2)
            ret_coefs = ret_coefs.transpose(0, 2, 1, 3, 4)
            ret_coefs = ret_coefs.reshape(self.seq_len, -1, self.size // 2,
                                          self.size // 2)
        elif self.data_option is None:
            print('Require data option...')
            exit()
        else:
            raise NotImplementedError

        ret_coefs = torch.FloatTensor(ret_coefs)
        img_features = torch.FloatTensor(img_features)
        annos = torch.FloatTensor(annos)
        labels = torch.LongTensor(labels)
        return ret_coefs, img_features, annos, labels, seq_info


class CASME_2Dataset(SAMMDataset):
    def __init__(self,
                 mode,
                 img_dirs,
                 seq_len=64,
                 step=32,
                 time_len=12,
                 input_size=256,
                 data_aug=False,
                 data_option=None,
                 dataset_name='CASME_2'):
        super().__init__(mode,
                         img_dirs,
                         seq_len=seq_len,
                         step=step,
                         time_len=time_len,
                         input_size=input_size,
                         data_aug=data_aug,
                         data_option=data_option,
                         dataset_name=dataset_name)
        self.label_dict = np.load(osp.join(params.CASME_2_LABEL_DIR,
                                           'label_dict.npy'),
                                  allow_pickle=True).item()
        self.anno_dict = np.load(osp.join(params.CASME_2_LABEL_DIR,
                                          'anno_dict.npy'),
                                 allow_pickle=True).item()


class SAMMImageDataset(Dataset):
    def __init__(self, img_ps):
        super().__init__()
        self.img_ps = img_ps
        self.bi_label = np.load(
            osp.join(params.SAMM_ROOT, 'bi_label.npy'),
            allow_pickle=True).item()  # imgs_dir -> [<target img_p> ... ]

    def __len__(self):
        return len(self.img_ps)

    def __getitem__(self, index):
        img_p = self.img_ps[index]
        npy_p = img_p.replace('.jpg', '.npy')
        feature = np.load(npy_p)
        feature = torch.tensor(feature, dtype=torch.float32)

        imgs_dir = osp.dirname(img_p)
        label = 1 if img_p in self.bi_label[
            imgs_dir] else 0  # 1 for spotting region
        label = torch.tensor(label, dtype=torch.long)

        return feature, label, img_p


class CASME_2ImageDataset(SAMMImageDataset):
    def __init__(self, img_ps):
        super().__init__(img_ps)
        self.bi_label = np.load(
            osp.join(params.CASME_2_LABEL_DIR, 'bi_label.npy'),
            allow_pickle=True).item()  # imgs_dir -> [<target img_p> ... ]


def get_diff(imgs):
    if len(imgs.shape) == 3:
        assert imgs.shape[1] == imgs.shape[2]  # imgs
    elif len(imgs.shape) == 4:
        assert imgs.shape[2] == imgs.shape[
            3] and imgs.shape[1] == WT_CHANNEL  # wt_coefs
    imgs1 = imgs[:-1]
    imgs2 = imgs[1:]
    return imgs2 - imgs1


def dwt2(img, wave_name='haar'):
    assert isinstance(img, np.ndarray)
    coefs = pywt.dwt2(img, wave_name)
    coefs = np.array([coefs[0], *coefs[1]])
    return coefs  # (4, w//2, h//2)


def get_smoothing_and_dr_coefs(imgs):
    '''
    GAUSS_KERNEL_PATH
    '''

    global sm_kernel, dr1_kernel, dr2_kernel
    sm_imgs = np.array([cv2.filter2D(img, -1, sm_kernel) for img in imgs])
    dr_ks = dr1_kernel.shape[0]

    dr1_res = []
    dr2_res = []
    for i in range(len(imgs) - dr_ks + 1):
        _imgs = sm_imgs[i:i + dr_ks]
        dr1_res.append((_imgs * dr1_kernel).sum(axis=0))
        dr2_res.append((_imgs * dr2_kernel).sum(axis=0))
    res = np.stack((*dr1_res, *dr2_res), 0)
    return res
