##
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from PIL import Image
from mmseg.apis import set_random_seed
import natsort
import shutil
from pycocotools.coco import COCO
import os
import cv2
from mmseg.apis import set_random_seed
import torch
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import matplotlib
import math
import matplotlib.pyplot as plt
from mmcv import Config
import os.path as osp
from functools import reduce
import random
import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data import Dataset
import json
from mmseg.core import mean_iou
from mmseg.utils import get_root_logger
from skimage import measure
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from utils.inference_tool import sliding_mask, sliding_img, sliding_window
##
os.chdir('/home/shm/hjh/kepco_crack2/')
current = os.getcwd()
dst_path = '/home/shm/hjh/kepco_crack2/'
with open(current + '/data_info.json', 'r') as outfile:
    data_info = json.load(outfile)
# Sliding segmentation configuration
winH = 1024
winW = 1024
stepSize = 512
windowSize = (winH, winW)
##
# split train and validation dataset
for key in data_info :
    data = data_info[key]
    img_path = data['img_path']
    label_path = data['label_path']
    img_list = os.listdir(img_path)
    img_list = natsort.natsorted(img_list)
    mask_list = os.listdir(label_path)
    mask_list = natsort.natsorted(mask_list)
    # 20% validation
    k = round(0.2 * len(img_list))
    val_list = random.sample(img_list,k)
    num = 0
    for val in val_list :
        # move validation image
        shutil.move(img_path + val, './data/validation/raw/'+key+'/image/' + val)
        shutil.move(label_path + val.split('.')[0]+'.png', './data/validation/raw/' + key + '/label/' + val.split('.')[0]+'.png')
        num = num + 1
        print('%s : %d / %d is moved to validation' %(key, num,k))
    # 80% train
    img_list = os.listdir(img_path)
    img_list = natsort.natsorted(img_list)
    mask_list = os.listdir(label_path)
    mask_list = natsort.natsorted(mask_list)
    num = 0
    for img in img_list :
        # move train image
        shutil.copy(img_path + img, './data/train/raw/' + key + '/image/' + img)
        shutil.copy(label_path + img.split('.')[0] + '.png','./data/train/raw/' + key + '/label/' + img.split('.')[0] + '.png')
        num = num + 1
        print('%s : %d / %d is moved to train' % (key, num, len(img_list)))
##
# Segment img and mask
for key in data_info :
    # for train image
    src_path = './data/train/raw/' + key + '/image/'
    dst_path = './data/train/seg/' + key + '/image/'
    src_list = os.listdir(src_path)
    src_list = natsort.natsorted(src_list)
    sliding_img(src_path,dst_path, src_list, stepSize,windowSize)
    # for validation image
    src_path = './data/validation/raw/' + key + '/image/'
    dst_path = './data/validation/seg/' + key + '/image/'
    src_list = os.listdir(src_path)
    src_list = natsort.natsorted(src_list)
    sliding_img(src_path,dst_path, src_list, stepSize,windowSize)

    # for train mask
    src_path = './data/train/raw/' + key + '/label/'
    dst_path = './data/train/seg/' + key + '/label/'
    src_list = os.listdir(src_path)
    src_list = natsort.natsorted(src_list)
    sliding_mask(src_path,dst_path, src_list, stepSize,windowSize)
    # for validation mask
    src_path = './data/validation/raw/' + key + '/label/'
    dst_path = './data/validation/seg/' + key + '/label/'
    src_list = os.listdir(src_path)
    src_list = natsort.natsorted(src_list)
    sliding_mask(src_path,dst_path, src_list, stepSize,windowSize)
