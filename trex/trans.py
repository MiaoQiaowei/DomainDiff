
import os
from tqdm import tqdm
# i_tgt = '/data/miaoqiaowei/data/imagenet-100-all/intra/train'

rate = 0.5
name = 'stable'
src = f'/data/miaoqiaowei/data/imagenet-100-all/last_chance/imagenet-100-intra-img'
tgt = f'/data/miaoqiaowei/data/imagenet-100-all/last_chance/train'
# for domain_n in os.listdir(src):
#     src_3 = os.path.join(src, domain_n)

for class_n in tqdm(os.listdir(src)):
    src_ = os.path.join(src,class_n)
    tgt_ = os.path.join(tgt,class_n)

    file_names = os.listdir(src_)
    nums = int(len(file_names)*rate)
    file_names = file_names[:nums]
    for file in file_names:
        # src_p = os.path.join(src_,file)
        src_p = os.path.join(src_,file)
        tgt_p = os.path.join(tgt_,file)
        os.system(f'cp {src_p} {tgt_p}')
        print(f'cp {src_p} {tgt_p}')
        # exit()