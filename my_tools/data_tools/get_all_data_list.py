import glob
import os

import numpy as np
import pickle
import sys
import tqdm
import shutil

pre_path = '/Users/kyanchen/Code/ClassificationFramework/data/UC/UCMerced_LandUse/Images'
sub_folder_list = glob.glob(pre_path +'/*')
all_data_list = []
for sub_folder in sub_folder_list:
    img_list = glob.glob(sub_folder+'/*')
    all_data_list += img_list

with open(pre_path+f'/../all_img_list.txt', 'w') as f:
    for file in all_data_list:
        file_name = os.path.basename(os.path.dirname(file)) + '/' + os.path.basename(file)
        gt_label = os.path.basename(os.path.dirname(file))
        f.write(file_name+' '+gt_label+'\n')

