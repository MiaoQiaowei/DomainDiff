import os
# with open('/home/miaoqiaowei/d2d_lora/useful/imagenet-list.txt','r') as f:
#     dir_name_list =  f.readlines()
#     dir_name_list = [name.replace('\n', '') for name in dir_name_list]
#     dir_name_list = [name for name in dir_name_list if name != '']
#     # dir_cn_name_list = [name.split(' ')[2] for name in dir_name_list]
#     index = [name.split(' ')[0] for name in dir_name_list]
#     dir_name_list = [name.split(' ')[1] for name in dir_name_list]
#     # dir_cn_name_list = [name.split(',')[0] for name in dir_cn_name_list]
# # print(index)

with open('/home/miaoqiaowei/d2d_lora/useful/imagenet-100.txt','r') as f:
    dir_names =  f.readlines()
    dir_names = [name.replace('\n', '') for name in dir_names]

src = '/data/miaoqiaowei/data/imagenet-v2-100/val'

for f in os.listdir(src):
    if f not in dir_names:
        f_p = os.path.join(src, f)
        print(f'rm -rf {f_p}')
        os.system(f'rm -rf {f_p}')
    # idx = int(f)
    # old_name = os.path.join(src, index[idx])
    # new_name = os.path.join(src, dir_name_list[idx])
    # print(f'from {old_name} to {new_name}')
    # # exit()
    # os.system(f'mv {old_name} {new_name}')
    