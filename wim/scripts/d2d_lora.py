# get test data set
from PIL import Image
from tqdm import tqdm
import os
import torch                                                                                                                                                                                                                     

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
    num_inference_steps = 20                                                                                                                                                                                                                    
    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}

# a+c, a+p, a+r, c+p, c+r, p+r
# art, clipart, product， real_world
domain = 'c+r'
device = 'cuda:3'
domain_a = 'clipart'
domain_b = 'real_world'


root = f'/data/miaoqiaowei/DG/office_home_inter/{domain}'
if not os.path.exists(root):
    os.makedirs(root)
bs = 10 
num = 100
num_inference_steps = 20 

class_names = ['Bottle', 'Fan', 'Calendar', 'Mop', 'Bike', 'Bed', 'Alarm_Clock', 'TV', 'Bucket', 'Postit_Notes', 'Batteries', 'Telephone', 'Kettle', 'Monitor', 'Shelf', 'Toys', 'Keyboard', 'Paper_Clip', 'Flowers', 'Marker', 'Computer', 'Table', 'Pencil', 'Hammer', 'Sneakers', 'Scissors', 'Eraser', 'Calculator', 'Clipboards', 'Flipflops', 'Webcam', 'Drill', 'Push_Pin', 'Folder', 'Printer', 'Desk_Lamp', 'Pen', 'Glasses', 'Couch', 'Mouse', 'Notebook', 'Speaker', 'Backpack', 'Helmet', 'Lamp_Shade', 'Knives', 'Laptop', 'Exit_Sign', 'Refrigerator', 'Screwdriver', 'Chair', 'Pan', 'Spoon', 'Ruler', 'ToothBrush', 'Sink', 'Oven', 'File_Cabinet', 'Radio', 'Curtains', 'Trash_Can', 'Mug', 'Candles', 'Fork', 'Soda']
class_names = [c.lower().replace('_', ' ') for c in class_names]

pipe = torch.load(f'{domain}.pt').to(device)

for class_name in tqdm(class_names):
    prompt = f'a {domain_a} {domain_b} {class_name}.'  
    data_path = os.path.join(root, class_name)                                                                                                                                   
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for i in tqdm(range(int(num/bs))):
        b = i*bs
        e = min(num, b+bs)

        images = pipe(**get_batch_inputs(b,e)).images
                                                                                                                                                                            
        for idx, img in tqdm(enumerate(images)):
            img.save(f'{data_path}/{b+idx}.png')
