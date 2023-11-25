# get test data set
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import torch
import argparse                                                                                    
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from lora_diffusion import monkeypatch_or_replace_lora, tune_lora_scale
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('--id',type=int)
parser.add_argument('--bs',type=int)

args = parser.parse_args()

def image_grid(imgs, rows=2, cols=2):                                                                                                                                                                                                         
    w, h = imgs[0].size                                                                                                                                                                                                                       
    grid = Image.new('RGB', size=(cols*w, rows*h))                                                                                                                                                                                            
                                                                                                                                                                                                                                              
    for i, img in enumerate(imgs):                                                                                                                                                                                                            
        grid.paste(img, box=(i%cols*w, i//cols*h))                                                                                                                                                                                            
    return grid 
def get_inputs(batch_size=1):                                                                                                                                                                                                                 
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]                                                                                                                                                             
    prompts = batch_size * [prompt]                                                                                                                                                                                                             
    num_inference_steps = 20                                                                                                                                                                                                                    
    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}
def get_batch_inputs(b, e):                                                                                                                                                                                                                 
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(b,e)]                                                                                                                                                             
    prompts = (e-b) * [prompt]                                                                                                                                                                                                             
    num_inference_steps = 50                                                                                                                                                                                                                    
    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps, "guidance_scale":2}





with open('../useful/imagenet-100.txt','r') as f:
    dir_names =  f.readlines()
    dir_names = [name.replace('\n', '') for name in dir_names]

with open('../useful/imagenet-list.txt','r') as f:
    dir_name_list =  f.readlines()
    dir_name_list = [name.replace('\n', '') for name in dir_name_list]
    dir_name_list = [name for name in dir_name_list if name != '']
    # dir_cn_name_list = [name.split(' ')[2] for name in dir_name_list]
    dir_name_list = [name.split(' ')[1] for name in dir_name_list]
    # dir_cn_name_list = [name.split(',')[0] for name in dir_cn_name_list]
    

with open('../useful/imagenet-names.txt','r') as f:
    names =  f.readlines()
    names = [name.replace('\n', '') for name in names]
    names = [name.split(":")[1] for name in names]
    names = [name.split('\'')[1] for name in names]
    names = [name.split(',')[0] for name in names]


idx2name = {k:v for k,v in zip(dir_name_list,names)}
name2idex = {v.lower():k for k,v in idx2name.items() if k in dir_names}
# data_path = '/data/miaoqiaowei/data/imagenet-100-intra/train'
path = '/data/miaoqiaowei/data/imagenet-100/train'

old_names =  os.listdir(path)
new_names = [idx2name[idx]for idx in old_names if idx in idx2name.keys()]
old2new = {k:v.lower() for k,v in zip(old_names, new_names)}


src = '/data/miaoqiaowei/data/imagenet-100/train'
tgt = '/data/miaoqiaowei/data/imagenet-100-intra/tmp_new'
# file_names = os.listdir('/data/miaoqiaowei/data/imagenet-100/finetune')
# src = '/data/miaoqiaowei/data/imagenet-100/finetune'
# for f in file_names:
#     print(old2new[f])
#     os.rename(f'{src}/{f}', f'{src}/{old2new[f]}')
# # print(file_names)
# exit()
# tgt = '/data/miaoqiaowei/data/imagenet-100-intra/tmp'



# names = ['robin', 'stretcher', 'gasmask', 'honeycomb', 'wing', 'mortarboard', 'great dane', 'green mamba', 'jean', 'little blue heron', 'cauliflower', 'red fox', 'american lobster', 'rock crab', 'vacuum', 'gyromitra', 'pirate', 'meerkat', 'boathouse', 'rottweiler', 'laptop', 'pineapple', 'shih-tzu', 'cinema', 'leafhopper', 'bonnet', 'english foxhound', 'safety pin', 'tripod', 'doberman', 'hard disc', 'dutch oven', 'papillon', 'komondor', 'african hunting dog']
# names = [ 'carbonara', 'gila monster', 'saluki', 'langur', 'window screen', 'vizsla', 'theater curtain', 'standard poodle', 'moped', 'bassinet', 'sarong', 'toy terrier', 'garter snake', 'chihuahua', 'computer keyboard', 'tile roof', 'harmonica', 'borzoi', 'football helmet', 'mousetrap', 'fiddler crab', 'goose', 'chocolate sauce', 'bottlecap', 'tabby', 'lorikeet', 'boxer', 'mixing bowl', 'modem', 'throne', 'car wheel', 'cocktail shaker', 'obelisk', 'rotisserie' ]
# names = ['american staffordshire terrier', 'ambulance', 'hognose snake', 'mexican hairless', 'ski mask', 'head cabbage', 'tub', 'reel', 'kuvasz', 'walking stick', 'chime', 'pickup', 'lampshade', 'walker hound', 'milk can', 'bannister', 'gibbon', 'park bench', 'american coot', 'stinkhorn', 'pedestal', 'dung beetle', 'purse', 'garden spider', 'coyote', 'wild boar','iron', 'hare', 'rocking chair', 'chesapeake bay retriever', 'slide rule']
bs = args.bs
id = args.id
print(id)
model_id = "/home/miaoqiaowei/.cache/huggingface/models--stabilityai--stable-diffusion-2-1-base/snapshots/88bb1a46821197d1ac0cb54d1d09fb6e70b171bc"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    f"cuda:{id}"
)
T = 0.1
monkeypatch_or_replace_lora(pipe.unet, torch.load(f"/home/miaoqiaowei/d2d_lora/output/imagenet-100-fine*/lora_weight_e1999_s2000.pt"))
tune_lora_scale(pipe.unet, T)

b_n = id*25
e_n = b_n + 25
see = {}
for i,(k,v) in enumerate(old2new.items()):
    if i>=b_n and i < e_n:
        see[k]=v
# f_names = os.listdir('/data/miaoqiaowei/data/imagenet-100-intra/train/tmp')
# pipe = torch.load('/home/miaoqiaowei/d2d_lora/scripts/intra-use.pt',map_location={'cuda:3':f'cuda:{id}'})
# pipe = torch.load('scripts/intra.pt').to(f"cuda:{3}")
# see = {'n01749939': 'green mamba'}
'''
photo reel
vacuum
'''
for dir_name, class_name in tqdm(see.items()):
    prompt = f'photo {class_name}.' 
    src_path = os.path.join(src, dir_name)
    tgt_path = os.path.join(tgt, dir_name)                                                                                                                                  
    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)
    num = len(os.listdir(src_path))
    max_id = num+1 
    for i in tqdm(range(int(num/bs))):
        b = i*bs
        e = min(num, b+bs)
        images = pipe(**get_batch_inputs(b,e)).images
                                                                                                                                                                            
        for idx, img in tqdm(enumerate(images)):
            f_p = f'{tgt_path}/{b+idx}.png'
            img.save(f_p)
            flag = False
            while flag==False:
                # f_p = os.path.join(tgt_path, f)
                image = cv2.imread(f_p, 0)
                if cv2.countNonZero(image) == 0:
                    max_id+=1
                    os.remove(f_p)
                    img = pipe(**get_batch_inputs(max_id,max_id+1)).images[0]
                    img.save(f_p)
                else:
                    flag = True
                    break
            
