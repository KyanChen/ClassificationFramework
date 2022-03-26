# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os

import numpy as np

from mmcls.datasets.pipelines.compose import Compose

from ..base_dataset import BaseDataset
from ..builder import DATASETS
from torch.utils.data import Dataset


@DATASETS.register_module()
class RepFolderDataset(Dataset):
    def __init__(self,
                 data_prefix,
                 pipeline,
                 ann_file,
                 test_mode=False):
        super(RepFolderDataset, self).__init__()
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        with open(self.ann_file) as f:
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            # info['gt_label'] = np.array(self.class_to_idx[gt_label], dtype=np.int64)
            data_infos.append(info)
        return data_infos
    
    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)
