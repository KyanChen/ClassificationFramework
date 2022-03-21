import glob
import os

import numpy as np
import pickle
import sys
import tqdm
import shutil

pre_path = r'H:\DataSet\SceneCls\UCMerced_LandUse\UCMerced_LandUse\Images'
sub_folder_list = glob.glob(pre_path +'/*')
train_frac = 0.8

train_list = []
val_list = []
for sub_folder in sub_folder_list:
    img_list = glob.glob(sub_folder+'/*')
    np.random.shuffle(img_list)
    np.random.shuffle(img_list)
    num_train_samps = int(len(img_list) * train_frac)
    train_list += img_list[:num_train_samps]
    val_list += img_list[num_train_samps:]

for phase in ['train_list', 'val_list']:
    with open(pre_path+f'/../{phase}.txt', 'w') as f:
        for file in eval(phase):
            file_name = os.path.basename(os.path.dirname(file)) + '/' + os.path.basename(file)
            gt_label = os.path.basename(os.path.dirname(file))
            f.write(file_name+' '+gt_label+'\n')

