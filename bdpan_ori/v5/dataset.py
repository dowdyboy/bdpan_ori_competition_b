import paddle
import paddle.vision.transforms as transforms
from paddle.io import Dataset

from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import numpy as np


class OriDataset(Dataset):

    def __init__(self,
                 source_list,
                 source_weight_list,
                 data_size,
                 scale_size_list=[520],
                 img_size_list=[512],
                 hflip_p=0.5,
                 color_jitter=True,
                 is_val=False,
                 ):
        super(OriDataset, self).__init__()
        self.source_list = source_list
        self.source_weight_list = source_weight_list
        self.data_size = data_size
        self.scale_size_list = scale_size_list
        self.img_size_list = img_size_list
        self.is_val = is_val
        self.hflip_p = hflip_p
        self.color_jitter = color_jitter
        assert len(self.source_list) == len(self.source_weight_list)
        assert len(self.scale_size_list) == len(self.img_size_list)
        self.source_img_list = self._get_img_list()
        # transform_list = [
        #     transforms.Resize(self.scale_size),
        #     transforms.RandomCrop((self.img_size, self.img_size)) if not self.is_val else transforms.CenterCrop((self.img_size, self.img_size))
        # ]
        # if hflip_p is not None:
        #     transform_list.append(
        #         transforms.RandomHorizontalFlip(hflip_p)
        #     )
        # if color_jitter:
        #     transform_list.append(transforms.ColorJitter(0.4, 0.4, 0.7, 0.015))
        # self.transforms = transforms.Compose(transform_list)
        self.color_jitter_trans = transforms.ColorJitter(0.4, 0.4, 0.7, 0.015)
        self.get_tensor = transforms.ToTensor()

    def _transform(self, im):
        def _get_param(img, output_size):
            if isinstance(output_size, int):
                output_size = (output_size, output_size)
            w, h = img.size
            th, tw = output_size
            if w == tw and h == th:
                return 0, 0, h, w
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            return i, j, th, tw

        rnd_idx = random.randint(0, len(self.scale_size_list) - 1)
        scale_size = self.scale_size_list[rnd_idx]
        img_size = self.img_size_list[rnd_idx]
        im = transforms.functional.resize(im, scale_size)
        if not self.is_val:
            i, j, h, w = _get_param(im, img_size)
            im = transforms.functional.crop(im, i, j, h, w)
        else:
            im = transforms.functional.center_crop(im, img_size)
        if self.hflip_p is not None and random.random() < self.hflip_p:
            im = transforms.functional.hflip(im)
        if self.color_jitter:
            im = self.color_jitter_trans(im)
        return im

    def _get_img_list(self):
        ret = []
        for source_path in self.source_list:
            source_list = []
            for filename in os.listdir(source_path):
                source_list.append(os.path.join(source_path, filename))
            ret.append(source_list)
        return ret

    def _get_source_idx(self):
        from dowdyboy_lib.rand import wheel_rand_index
        return wheel_rand_index(self.source_weight_list)

    # def _get_rotate(self, im):
    #     angles = [0, -90, -180, -270]
    #     labels = [0, 1, 2, 3]
    #     r_idx = random.randint(0, len(angles) - 1)
    #     im = im.rotate(angles[r_idx])
    #     lb = labels[r_idx]
    #     return im, lb

    def _get_rotate(self, im):
        angles = [None, Image.ROTATE_270, Image.ROTATE_180, Image.ROTATE_90]
        labels = [0, 1, 2, 3]
        r_idx = random.randint(0, len(angles) - 1)
        if angles[r_idx] is not None:
            im = im.transpose(angles[r_idx])
        lb = labels[r_idx]
        return im, lb

    def __getitem__(self, idx):
        source_idx = self._get_source_idx()
        filepath_list = self.source_img_list[source_idx]
        f_idx = random.randint(0, len(filepath_list) - 1)
        file_path = filepath_list[f_idx]
        im = Image.open(file_path).convert("RGB")
        # im = self.transforms(im)
        im = self._transform(im)
        im, lb = self._get_rotate(im)
        im = self.get_tensor(im)
        return im, lb

    def __len__(self):
        return self.data_size


class OriTestDataset(Dataset):

    def __init__(self,
                 source_path,
                 scale_size=520,
                 img_size=512,
                 ):
        super(OriTestDataset, self).__init__()
        self.source_path = source_path
        self.scale_size = scale_size
        self.img_size = img_size
        self.file_list = self._get_file_list()
        self.transforms = transforms.Compose([
            transforms.Resize(self.scale_size),
            transforms.CenterCrop((self.img_size, self.img_size))
        ])
        self.get_tensor = transforms.ToTensor()

    def _get_file_list(self):
        ret = []
        for filename in os.listdir(self.source_path):
            ret.append(
                os.path.join(self.source_path, filename)
            )
        return ret

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        im = Image.open(file_path).convert("RGB")
        im = self.transforms(im)
        im = self.get_tensor(im)
        return im, file_path

    def __len__(self):
        return len(self.file_list)


