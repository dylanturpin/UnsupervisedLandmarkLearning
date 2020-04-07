"""Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Dataset classes for handling the BBCPose data
"""

from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import scipy.io as sio
from .base_datasets import BaseVideoDataset



class PretrainDataset(BaseVideoDataset):
    def __init__(self, config, partition):
        self.config = config
        super(PretrainDataset, self).__init__(config, partition)

    def setup_frame_array(self, config, partition):
        if partition == 'train':
            self.input_vids = list(range(config.n_train_vids))
        elif partition == 'validation':
            self.input_vids = list(range(config.n_train_vids, config.n_train_vids+config.n_val_vids))
        self.input_vids = [str(idx) for idx in self.input_vids]

        # first bin is 0
        self.num_frames_array = [0]
        frac = 1
        if partition == 'val':
            frac = self.config['val_frac']
        for folder in self.input_vids:
            # truncate validation if frac is specified
            num_frames = len(os.listdir(os.path.join(self.config.dataset_path, folder)))
            print(num_frames)
            self.num_frames_array.append(num_frames)
        self.num_frames_array = np.array(self.num_frames_array).cumsum()

        return self.num_frames_array

    def process_batch(self, vid_idx, img_idx):
        vid_path = os.path.join(self.dataset_path, self.input_vids[vid_idx])
        num_frames = len(os.listdir(vid_path))

        img_idx2_offset = self.sample_temporal(num_frames, img_idx, 3, 40)

        img_1 = os.path.join(vid_path, f'{img_idx}.jpg')
        img_2 = os.path.join(vid_path, f'{img_idx + img_idx2_offset}.jpg')

        #bbox_x_min = gt_kpts[0].min() - 60
        #bbox_x_max = gt_kpts[0].max() + 60
        #bbox_y_min = gt_kpts[1].min() - 60
        #bbox_y_max = gt_kpts[1].max() + 60

        bbox_x_min = 0
        bbox_x_max = 127
        bbox_y_min = 0
        bbox_y_max = 127

        # clip the bounding boxes
        img_a = Image.open(img_1).convert('RGB')
        wh = img_a.size
        bbox_x_min = max(0, bbox_x_min)
        bbox_y_min = max(0, bbox_y_min)
        bbox_x_max = min(wh[0], bbox_x_max)
        bbox_y_max = min(wh[1], bbox_y_max)

        bbox_a = (bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max)
        img_a = img_a.crop(bbox_a)
        img_temporal = Image.open(img_2).convert('RGB')
        img_temporal = img_temporal.crop(bbox_a)
        # randomly flip
        if np.random.rand() <= self.flip_probability:
            # flip both images
            img_a = transforms.functional.hflip(img_a)
            img_temporal = transforms.functional.hflip(img_temporal)

        bbox_w = bbox_x_max - bbox_x_min
        bbox_h = bbox_y_max - bbox_y_min

        img_temporal = self.to_tensor(self.resize(img_temporal))
        img_temporal = self.normalize(img_temporal)

        img_a_color_jittered, img_a_warped, img_a_warped_offsets, target=self.construct_color_warp_pair(img_a)

        return {'input_a': img_a_color_jittered, 'input_b': img_a_warped,
                'input_temporal': img_temporal, 'target': target,
                'imname': self.input_vids[vid_idx] + '_' + str(img_idx) + '.jpg',
                'warping_tps_params': img_a_warped_offsets}
