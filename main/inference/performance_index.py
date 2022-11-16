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
from pycocotools.coco import COCO
import os
import cv2
from mmseg.apis import set_random_seed
import torch
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import matplotlib
#%matplotlib nbagg
import math
import matplotlib.pyplot as plt
#from .registry import DATASETS
from mmcv import Config
import os.path as osp
from functools import reduce
import random
import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data import Dataset
from mmseg.core import mean_iou
from mmseg.utils import get_root_logger
from skimage import measure
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import time
from skimage import measure
import json
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from utils.inference_tool import sliding_window
from utils.inference_tool import slding_inference,gt_vis
from utils.inference_tool import save_pred_gt
##
# Dataset raw
os.chdir('/home/shm/hjh/kepco_crack2/')
# load data information json
current = os.getcwd()
with open(current + '/data_info.json', 'r') as outfile:
    data_info = json.load(outfile)
##
# model initialization
cfg_path = 'configs/swin/swin_kepco3-2.py'
checkpoint_file = 'data/output/new_train/latest.pth'
cfg = Config.fromfile(cfg_path)
model_ckpt = init_segmentor(cfg, checkpoint_file, device='cuda:0')
##
#for save gt overlay and prediction
# save mask overlay and copy a image

for key in data_info :
    img_path = data_info[key]['img_path']
    label_path = data_info[key]['label_path']
    img_list = os.listdir(img_path)
    img_list = natsort.natsorted(img_list)
    label_list = os.listdir(label_path)
    label_list = natsort.natsorted(label_list)
    img_part = random.sample(img_list, k = 2)
    out_path = './data/inference/' + key + '/'
    windowSize = (1024,1024)
    stepSize = 512

    # Start inference
    index1_list,index2_list, index3_list = save_pred_gt(img_path, label_path, img_part,model_ckpt,out_path,windowSize,stepSize)
    data_info[key]['index1'] = index1_list
    data_info[key]['index2'] = index2_list
    data_info[key]['index3'] = index3_list
    print('%s is finished' %(key))

##


