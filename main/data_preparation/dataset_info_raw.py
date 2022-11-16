##
#library import
import shutil
import os
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from skimage.draw import polygon
import numpy as np
import json
import cv2
import mmcv
import natsort
##
os.chdir('/home/shm/hjh/kepco_crack2/')

# Dataset preparation
'''
- train dataset(80%)
    - aihub shm
    - aihub_welcon
    - seoul_crack
    - tancheon
    - bridge2
    

- validation dataset(20%)
    - aihub shm
    - aihub welcon
    - seoul_crack
    - tancheon
    - bridge2
    '''
# train data
aihub_shm = dict(
    img_path = './data/aihub_shm/image/',
    label_path = './data/aihub_shm/label/',
    total_img = 0,
    total_mask = 0,
    resolution = []
)

seoul_crack = dict(
    img_path = './data/seoul_crack/image/',
    label_path = './data/seoul_crack/label/',
    total_img = 0,
    total_mask = 0,
    resolution = []
)

tancheon = dict(
    img_path = './data/tancheon/image/',
    label_path = './data/tancheon/label/',
    total_img = 0,
    total_mask = 0,
    resolution = []
)

# validation data
aihub_welcon = dict(
    img_path = './data/aihub_welcon/image/',
    label_path = './data/aihub_welcon/label/',
    total_img = 0,
    resolution = []
)

bridge2 = dict(
    img_path = './data/bridge2/image/',
    label_path = './data/bridge2/label/',
    total_img = 0,
    total_mask = 0,
    resolution = []
)

data_list = [aihub_shm, aihub_welcon, seoul_crack,tancheon,bridge2]
##
for data in data_list :
    count = 0
    img_list = os.listdir(data['img_path'])
    img_list = natsort.natsorted(img_list)
    mask_list = os.listdir(data['label_path'])
    mask_list = natsort.natsorted(mask_list)
    data['total_img'] = len(img_list)
    data['total_mask'] = len(mask_list)
    data['img_name'] = img_list[0]
    data['label_name'] = mask_list[0]
    x_res = []
    y_res = []
    # rename mask file
    for mask_name in mask_list :
        os.rename(data['label_path'] + mask_name, data['label_path'] + mask_name.split('.')[0] + '.png')
    # remove img and mask which its resolution is under 1024
    for img_name in img_list :
        img = mmcv.imread(data['img_path'] + img_name)
        resolution = img.shape
        if resolution[0] < 1024 and resolution[1] < 1024 :
            data['total_img'] = data['total_img'] - 1
            data['total_mask'] = data['total_mask'] - 1
            os.remove(data['img_path'] + img_name)
            os.remove(data['label_path'] + img_name.split('.')[0] + '.png')

        else :
            x_res.append(resolution[0])
            y_res.append(resolution[1])
            count = count + 1
            print('%d / %d is finished\n' %(count, len(img_list)))
    xmin = min(x_res)
    ymin = min(y_res)
    xmax = max(x_res)
    ymax = max(y_res)
    data['resolution'].append((xmin,ymin))
    data['resolution'].append((xmax,ymax))
    print('%s is finished\n' %(str(data)))

##
data_info = dict(
    aihub_shm = aihub_shm,
    aihub_welcon = aihub_welcon,
    seoul_crack = seoul_crack,
    bridge2 = bridge2,
    tancheon = tancheon
)
current = os.getcwd()
with open(current + '/data_info.json', 'w') as outfile:
    json.dump(data_info, outfile, indent=4)

print('save completed')

##
for key in data_info :
    print(data_info[key])

##
for key in data_info :
    data = data_info[key]
    img_list = os.listdir(data['img_path'])
    img_list = natsort.natsorted(img_list)

    for img_name in img_list :
        if img_name.split('.')[1] == 'JPG':
            os.rename(data['img_path'] + img_name, data['img_path'] + img_name.split('.')[0] + '.jpg')