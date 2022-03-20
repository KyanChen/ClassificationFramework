import os

import lmdb
import numpy as np
import pickle
import sys
import tqdm
import shutil

pre_path = '../../data/new_data'
phase_list = ['train', 'val']

labels = pickle.load(open(pre_path + '/all_gt_label.pkl', 'rb'))
urls = pickle.load(open(pre_path + '/all_urls.pkl', 'rb'))
embs = np.load(pre_path + '/all_emb.npy')
assert len(labels) == len(urls) == len(embs)
if os.path.exists(pre_path + f'/lmdb'):
    shutil.rmtree(pre_path + f'/lmdb')
os.makedirs(pre_path + f'/lmdb', exist_ok=True)
commit_interval = 1000
for phase in phase_list:
    keys = pickle.load(open(pre_path + f'/{phase}_list.pkl', 'rb'))
    num_keys = len(keys)
    items = ['urls', 'embs', 'labels']
    # items = ['embs']
    for item in items:
        print(eval(item)[50])
        data_size_per_item = sys.getsizeof(eval(item)[0])
        print(f'data size: {item}--{data_size_per_item}')
        data_size = data_size_per_item * num_keys

        env = lmdb.open(pre_path + f'/lmdb/{phase}_{item}.lmdb', map_size=data_size * 100)
        txn = env.begin(write=True)

        for idx, key in enumerate(tqdm.tqdm(keys)):
            key_byte = str(key).encode()
            if item == 'urls':
                data = urls[key].replace('[', '').replace(']', '').encode()
            elif item == 'embs':
                data = eval(item)[key]
            elif item == 'labels':
                data = eval(item)[key].encode()
            else:
                raise NotImplementedError
            txn.put(key_byte, data)
            if idx % commit_interval == 1:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        env.close()
        print(f'Finish writing {phase}_{item}')

