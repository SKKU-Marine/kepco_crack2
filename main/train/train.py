##
#Library import
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from PIL import Image
from mmseg.apis import set_random_seed
import natsort


from pycocotools.coco import COCO
import os
import cv2
from mmseg.apis import set_random_seed
import torch

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp
from functools import reduce

import numpy as np
from mmcv.utils import print_log
from torch.utils.data import Dataset

from mmseg.core import mean_iou
from mmseg.utils import get_root_logger
#from .registry import DATASETS
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
##
os.chdir('/home/shm/hjh/kepco_crack2/')
classes = ('background','crack',)
palette = [[0,0,0],[1,1,1]] # crack pixel of sample datasets has [1,1,1] pixel value
##
@DATASETS.register_module()
class CrackDataset0(CustomDataset):
  CLASSES = ('background','crack',)
  PALETTE = [[0,0,0],[1,1,1]]
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
                     split=None,reduce_zero_label=False,
                     **kwargs)
    assert osp.exists(self.img_dir) and self.split is None
##
#Dataset format for tiff extension
@DATASETS.register_module()
class CrackDataset1(CustomDataset):
  CLASSES = ('background','crack',)
  PALETTE = [[0,0,0],[1,1,1]]
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.tiff', seg_map_suffix='.png',
                     split=None,reduce_zero_label=False,
                     **kwargs)
    assert osp.exists(self.img_dir) and self.split is None

##
# Dataset config
from mmcv import Config
cfg = Config.fromfile('configs/swin/swin_kepco3-2.py')


##
'''
Total number of train data : 6356 
Batch Size : 4  
epoch = 2
iter = (images * epoch) / batch size  
iters = 303,331
epoch = (iter * batch size) / images
'''


# train configuration
cfg.runner = dict(type='IterBasedRunner', max_iters=3178) # Number of training iteration
cfg.checkpoint_config = dict(by_epoch=False, interval=1000)
cfg.evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)
meta = dict()
meta['config'] = cfg.pretty_text
print(f'Config:\n{cfg.pretty_text}')

##
#train
# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES
model.PALETTE = datasets[0].PALETTE
print(datasets[0].CLASSES)
print(datasets[0].PALETTE)
torch.cuda.empty_cache()

# Create work_dir and Train

##
#train
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                meta=dict(CLASSES=classes, PALETTE=palette))