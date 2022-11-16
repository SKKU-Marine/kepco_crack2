
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

def sliding_window(image, stepSize, windowSize):
# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			if x + windowSize[1] > image.shape[1] :
				x = image.shape[1] - windowSize[1]

			elif y + windowSize[0] > image.shape[0] :
				y = image.shape[0] - windowSize[0]

			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def slding_inference(img,stepSize, windowSize, model_ckpt) :
    full_pred = np.zeros((img.shape[0],img.shape[1]))
    full_overlay = img.copy()
    winH = windowSize[0]
    winW = windowSize[1]
    for (x, y, window) in sliding_window(img, stepSize=256, windowSize=windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        result = inference_segmentor(model_ckpt, window) # inference on segmented window
        full_pred[y:y + windowSize[1], x:x + windowSize[0]] = result[0] # overwrite on full mask
        overlay = window.copy()

    for i in range(full_pred.shape[1]):  # loop on x pixels
        for j in range(full_pred.shape[0]):  # loop on y pixels
            if full_pred[j, i] == 1.0:
                full_overlay[j, i, :] = np.array((0, 0,255), dtype=np.uint8)
    return full_pred, full_overlay

def gt_vis(img,mask) :
    gt_overlay = img.copy()
    pixel1 = np.array((1, 1, 1), dtype=np.uint8)
    pixel2 = np.array((2, 2, 2), dtype=np.uint8)
    for i in range(mask.shape[1]):  # loop on x pixels
        for j in range(mask.shape[0]):  # loop on y pixels
            if np.array_equal(mask[j, i, :], pixel1) :
                gt_overlay[j, i, :] = np.array((255, 0, 0), dtype=np.uint8)
            elif np.array_equal(mask[j,i,:],pixel2) :
                mask[j,i,:] = np.array((1, 1, 1), dtype=np.uint8)
                gt_overlay[j, i, :] = np.array((255, 0, 0), dtype=np.uint8)

    return gt_overlay, mask

def index_cal(gt_overlay,full_pred,mask):
    objects = measure.label(mask, return_num=True)
    num_crack = objects[1] #total number of crack object in gt
    pos_obj = 0  # total number of detected positive crack by model
    pos_pixels = [] # total detected crack pixels per each crack object
    gt_coords = [] # coordinates of crack object in gt
    gt_pixels = [] # total number of crack pixels in gt
    regions = measure.regionprops(objects[0])
    gt_index = gt_overlay.copy()
    for region in regions:
        num = 0
        coordis = region.coords  # coordinate list of a crack object
        num_coords = coordis.shape[0] / 3  # total pixels of a crack object
        gt_pixels.append(num_coords)
        for coord in coordis:
            if full_pred[coord[0], coord[1]] == 1:
                num = num + 1
        num = num / 3  # it counts also 3 channels so divide it by 3
        gt_coords.append(coordis)
        pos_pixels.append(num)
        minr, minc, maxr, maxc = region.bbox[0], region.bbox[1], region.bbox[3], region.bbox[4]
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        if num > 1:
            pos_obj = pos_obj + 1
            index = round(((num*3) / num_coords) * 100, 1)
            gt_index = cv2.rectangle(gt_index, (minc, minr), (maxc, maxr), (36, 255, 12), 4)
            s = str(index) + '%'
            gt_index =cv2.putText(gt_index, s, (minc, maxr + 50), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (36, 255, 12), 6)


        if num == 0:
            gt_index = cv2.rectangle(gt_index, (minc, minr), (maxc, maxr), (0, 255, 255), 4)
            s = str(num) + '%'
            gt_index =cv2.putText(gt_index, s, (minc + 50, maxr), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 6)

    index21_list = [] # performance index list for predicted crack at pixel level
    index31_list = [] # performance index list for whole crack at pixel level
    for i in range(len(pos_pixels)):
        acc = (pos_pixels[i] / gt_pixels[i]) * 100
        index31_list.append(acc)
        if pos_pixels[i] != 0:
            index21_list.append(acc)
    index1 = (pos_obj / num_crack) * 100
    if len(index21_list) == 0 :
        index2 = 0
    if len(index31_list) == 0 :
        index3 = 0
    else :
        index2 = sum(index21_list) / len(index21_list)
        index3 = sum(index31_list) / len(index31_list)

    return index1, index2, index3, gt_index,gt_overlay


def save_pred_gt(img_path, label_path, img_list, model_ckpt, out_path,windowSize, stepSize) :
    count = 0
    index1_list = []
    index2_list = []
    index3_list = []
    for img_name in img_list :
        file_name = img_name.split('.')[0]
        img = mmcv.imread(img_path + img_name, flag='color', channel_order='rgb', backend='cv2')
        mask = mmcv.imread(label_path + file_name + '.png', flag='color', channel_order='rgb', backend='cv2')
        mask = np.array(mask, dtype=np.uint8)
        # generate gt_overlay
        gt_overlay, mask = gt_vis(img, mask)
        # inference setting
        winH = windowSize[0]
        winW = windowSize[1]
        # result = inference_segmentor(model_ckpt, img)
        full_pred, full_overlay = slding_inference(img, stepSize, windowSize, model_ckpt)

        # calculate performance index
        index1, index2, index3, gt_index, gt_overlay = index_cal(gt_overlay, full_pred, mask)
        #gt_overlay = cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        index1_list.append(index1)
        index2_list.append(index2)
        index3_list.append(index3)
        # gt_overlay : gt overlayed on image RGB
        # full_overlay : prediction overlayed on image RGB
        # full_pred : prediction in mask grayscale
        # gt_index : gt overlayed on gt_overlay with bbox and performance index
        # save image and gt_overlayed index
        Image.fromarray(gt_index, mode = 'RGB').save(out_path+ 'gt_index/' + file_name + '.jpg')
        Image.fromarray(gt_overlay, mode = 'RGB').save(out_path+ 'gt_overlay/' + file_name + '.jpg')
        Image.fromarray(full_overlay, mode = 'RGB').save(out_path + 'predicted/' + file_name + '.jpg')
        #Image.fromarray(img, mode = 'RGB').save(out_path + 'image/' + file_name + '.jpg')
        # cv2.imwrite(out_path+ 'gt_index/' + file_name + '.jpg', gt_index)
        # cv2.imwrite(out_path+ 'gt_overlay/' + file_name + '.jpg', gt_overlay)
        # cv2.imwrite(out_path + 'predicted/' + file_name + '.jpg', full_overlay)
        #cv2.imwrite(out_path + 'image/' + file_name + '.jpg', img)
        count = count + 1
        print('%d / %d is finished'%(count, len(img_list)))
    return index1_list, index2_list, index3_list

def sliding_mask(src_path, dst_path, src_list, stepSize, windowSize) :
    num = 0
    winH = windowSize[0]
    winW = windowSize[1]
    for mask_name in src_list:
        mask = mmcv.imread(src_path + mask_name, flag='grayscale')
        count = 0
        num = num + 1
        for (x, y, window) in sliding_window(mask, stepSize=stepSize, windowSize=windowSize):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            cv2.imwrite(dst_path+mask_name.split('.')[0]+'-'+str(count)+'.png',
                mask[y:y + windowSize[1], x:x + windowSize[0]])
            count = count + 1
            print('%d sub mask saved : %d / %d' % (count, num, len(src_list)))

def sliding_img(src_path, dst_path, src_list, stepSize, windowSize) :
    num = 0
    winH = windowSize[0]
    winW = windowSize[1]
    for img_name in src_list:
        count = 0
        num = num + 1
        img = mmcv.imread(src_path + img_name)
        for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=windowSize):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            cv2.imwrite(dst_path +img_name.split('.')[0] +'-'+str(count)+'.' + img_name.split('.')[1],
                        img[y:y + windowSize[1], x:x + windowSize[0]])
            count = count + 1
            print('%d sub image saved : %d / %d' % (count, num, len(src_list)))



