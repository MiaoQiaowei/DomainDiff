import os

src = '~/data/imagenet-a/imagenet-a'
tg = '/home/miaoqiaowei/trex/in-a'

with open('/home/miaoqiaowei/trex/imagnet-100/imagenet-100.txt','r') as f:
    dir_names =  f.readlines()
    dir_names = [name.replace('\n', '') for name in dir_names]
    for name in dir_names:
        os.system(f'cp -r {src}/{name} {tg}')
    print('done')
    exit()