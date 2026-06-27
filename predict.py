import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import random
import time
from PIL import Image
import argparse
import os.path as osp
import yaml
import os.path as osp
import os
from utils import AverageMeter,normalize_img,get_obj_from_string,log_config,UIE_train,UIE_val,get_file_lst
import importlib
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
import clip
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info

try:
    from thop import profile
    THOP_AVAILABLE = True
except:
    THOP_AVAILABLE = False

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default=0,help="id of cuda device,default:0")
parser.add_argument('--exp', type=str, default=0,help="path experiment config")
parser.add_argument('--debug', type=bool, default=False,help="debug or not")
parser.add_argument('--ckpt', type=str, default="best",help="what checkpoint to use, defaulte best loss")
parser.add_argument('--data', type=str, default="ts",help="which dataset to evaluate")
parser.add_argument('--wild', type=str, default="",help="path to wild dataset")
parser.add_argument('--resize', type=int, default=0,help="resize input images")
args = parser.parse_args()

with open(args.exp,'r') as f:
    config = yaml.load(f,Loader=yaml.FullLoader)


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

device = torch.device(f'cuda:{args.cuda_id}')
config['model_name'] = config['model_file'].split('.')[1]
weight_decay = config['weight_decay']

if args.debug:
    
    log_path = f"./debug/{config['name']}_{config['model_name']}_lr{config['lr']}_wd{config['weight_decay']}_log/"
    snapshot_path = log_path + 'weights/'

else:
    log_path = f"./exp_results/{config['name']}_{config['model_name']}_lr{config['lr']}_wd{config['weight_decay']}_epoch{config['num_epochs']}_log/"# to save loss and dice score etc.
    snapshot_path = log_path + 'weights/'


if args.data == 'ts':
    val_vis = f"./{log_path}predict/L622_{args.ckpt}/imgs"
if args.data == 'u80':
    val_vis = f"./{log_path}predict/U80_{args.ckpt}/imgs"
if args.data == 's110':
    val_vis = f"./{log_path}predict/S110_{args.ckpt}/imgs"
if args.data == 'c60':
    val_vis = f"./{log_path}predict/C60_{args.ckpt}/imgs"
if args.data == 'r53':
    val_vis = f"./{log_path}predict/R53_{args.ckpt}/imgs"


if len(args.wild) != 0:
    if len(args.wild.split('/')[-1]) != 0:
        wild_name = args.wild.split('/')[-1]
    else:
        wild_name = args.wild.split('/')[-2]
    
    val_vis = f"./{log_path}predict/{wild_name}vis_{args.ckpt}"
if not osp.exists(val_vis):
    os.makedirs(val_vis)

def data_aug(image,label,re_size=None):
    if re_size:
        image = TF.resize(image,(re_size,re_size))
        label = TF.resize(label,(re_size,re_size))
    return image,label


def val_data_aug(image,label,re_size=None):
    if re_size:
        image = TF.resize(image,(re_size,re_size))
        label = TF.resize(label,(re_size,re_size))
    return image,label

def get_file_lst_c60(file_path):
    
    file_list = os.listdir(file_path)
    num_train = len(file_list)
    path_lst = []
    for f in file_list:
        f_path = os.path.join(file_path,f)
        path_lst.append((f_path,))
    train_lst = path_lst
    return train_lst


class UIE_c60(Dataset):
    def __init__(self,train_lst,image_size):
        self.path_lst =train_lst
        self.image_size = image_size
        print(f"got {len(self.path_lst)} images,{len(self.path_lst)} references")
    def __getitem__(self,index):
        name = self.path_lst[index][0].split('/')[-1]
        img = Image.open(self.path_lst[index][0]).convert('RGB')
        prompt = 'An underwater image'
        image = TF.to_tensor(img)      
        image,ref = val_data_aug(image,image,self.image_size)
        
        return {
            'image': image,
            'name':name,
            'prompt': prompt
        }
    def __len__(self):
        return len(self.path_lst)

# 测试集路径
if args.data == 'ts':
    val_lst = get_file_lst_c60("-")
    val_set = UIE_c60(val_lst,image_size=None)

if args.data == 'u80':
    val_lst = get_file_lst_c60("-")
    val_set = UIE_c60(val_lst,image_size=None)
    
if args.data == 's110':
    val_lst = get_file_lst_c60("-")
    val_set = UIE_c60(val_lst,image_size=None)   
     
if args.data == 'c60':
    val_lst = get_file_lst_c60("-")
    val_set = UIE_c60(val_lst,image_size=None)
    
if args.data == 'r53':
    val_lst = get_file_lst_c60("-")
    val_set = UIE_c60(val_lst,image_size=None)
    

if len(args.wild) != 0:
    val_lst = get_file_lst_c60(args.wild)
    val_set = UIE_c60(val_lst,image_size=None)

val_loader = DataLoader(val_set,batch_size = 1,shuffle=False,num_workers=8,pin_memory=False)

in_channels = 3
num_class = 3


print(f"Start loading CLIP")  
# CLIP权重     
clip_model, _ = clip.load("-", device=device, jit=False)  
clip_model.eval()   
print(f"Finish loading CLIP")

net = get_obj_from_string(config['model_file'])(in_channels, clip_model=clip_model).to(device)

mse_loss_lst = []
val_mse_loss_lst = []


def predict(net,args):
    if args.ckpt == 'best':
        ckpt = torch.load(f'{snapshot_path}best_val.pth',map_location=device)
        print(f'load bset val loss checkpoint...')
    if args.ckpt == 'best2':
        ckpt = torch.load(f'{snapshot_path}best_val2.pth',map_location=device)
        print(f'load bset val loss checkpoint...')
    if args.ckpt == 'musiq':
        ckpt = torch.load(f'{snapshot_path}best_musiq.pth',map_location=device)
        print(f'load bset musiq checkpoint...')
    if args.ckpt == 'last':
        ckpt = torch.load(f'{snapshot_path}latest.pth',map_location=device)
        print(f'load latest checkpoint...')
    elif args.ckpt == 'p':
        ckpt = torch.load(f'{snapshot_path}best_psnr_val.pth',map_location=device)
        print(f'load bset val psnr checkpoint...')
    elif args.ckpt == 's':
        ckpt = torch.load(f'{snapshot_path}best_ssim_val.pth',map_location=device)
        print(f'load bset val ssim checkpoint...')
    net.load_state_dict(ckpt['net'])
    
      
    print(f'Loading Done!')  
    net.eval()
    print(f'start inference...')
    with torch.no_grad():

        for data in tqdm(val_loader):
            name = data['name']
            
            if args.resize == 0:      
                image = data['image'].to(device)
            else:
                size = args.resize
                image = data['image']
                b,c,h,w = image.shape
                image = F.interpolate(image,size=(size,size),mode='bilinear').to(device)
            prompt = data['prompt']
            text_tokens = clip.tokenize(prompt).to(device)  
            pred_ = net(image, text_tokens)
            if isinstance(pred_,tuple):
                pred = pred_[0]
            else:
                pred = pred_
            if args.resize != 0:
                pred = F.interpolate(pred,size=(h,w),mode='bilinear')
            torchvision.utils.save_image(pred,f'{val_vis}/{name[0][:-4]}.png')
predict(net,args)