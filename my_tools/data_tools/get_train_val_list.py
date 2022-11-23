import glob
import os

import numpy as np
import pickle
import sys
import tqdm
import shutil
from skimage import io

pre_path = r'D:\GID\RGB_15_train'
sub_folder_list = glob.glob(pre_path +'/*')
train_frac = 0.8

train_list = []
val_list = []
for sub_folder in sub_folder_list:
    img_list = glob.glob(sub_folder+'/*')
    img_list = [x for x in img_list if 0 < io.imread(x).shape[0] < 60]
    np.random.shuffle(img_list)
    np.random.shuffle(img_list)
    np.random.shuffle(img_list)
    img_list = img_list
    # num_train_samps = int(len(img_list) * train_frac)
    num_train_samps = len(img_list)
    train_list += img_list[:num_train_samps]
    val_list += img_list[num_train_samps:]

for phase in ['train_list', 'val_list']:
    folder = pre_path + f'/../N600'
    os.makedirs(folder, exist_ok=True)
    with open(folder+f'/all_{phase}_56.txt', 'w') as f:
        for file in eval(phase):
            file_name = os.path.basename(os.path.dirname(file)) + '/' + os.path.basename(file)
            gt_label = os.path.basename(os.path.dirname(file))
            f.write(file_name+' '+gt_label+'\n')

