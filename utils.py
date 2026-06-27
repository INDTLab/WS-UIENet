import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
import importlib
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG19_Weights
import skimage
from skimage.metrics import structural_similarity as ssim
import clip 

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, img1, img2):
        img1_np = img1.detach().permute(0, 2, 3, 1).cpu().numpy()
        img2_np = img2.detach().permute(0, 2, 3, 1).cpu().numpy()
        
        ssim_value = 0
        for i in range(img1_np.shape[0]):
            data_range = img2_np[i].max() - img2_np[i].min()
            ssim_value += ssim(img1_np[i], img2_np[i], multichannel=True, channel_axis=-1, data_range=data_range)
        
        return 1 - (torch.tensor(ssim_value / img1_np.shape[0]))

class percep_loss(nn.Module):
    def __init__(self, device, weight=1):
        super().__init__()
        weights = VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=weights)
        features = vgg.features[:-1]
        self.features = features.to(device)
        for param in self.features.parameters():
            param.requires_grad = False
        self.weight = weight

    def forward(self, x, gt):
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        gt = TF.normalize(gt, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        x_feat = self.features(x)
        gt_feat = self.features(gt)
        loss = self.weight * F.mse_loss(x_feat, gt_feat)
        return loss

def data_aug(image, re_size=None):
    if re_size:
        image = TF.resize(image, (re_size, re_size))
    return image


def find_prompt(csv_file, filename):
    df = pd.read_csv(csv_file)

    prompt = df[df['Input'] == filename]['Prompt'].values

    if len(prompt) > 0:
        return prompt[0]
    else:
        return "未找到该文件名对应的描述提示"


def get_file_lst(file_path, gt_path, pr_path):
    file_list = os.listdir(file_path)
    num_train = len(file_list)
    path_lst = []
    for f in file_list:
        f_path = os.path.join(file_path, f)
        gt_f_path = os.path.join(gt_path, f)

        path_lst.append((f_path, gt_f_path, pr_path))
    train_lst = path_lst
    return train_lst


class UIE_train(Dataset):
    def __init__(self, train_lst, image_size):
        self.path_lst = train_lst
        self.image_size = image_size
        print(f"got {len(self.path_lst)} images,{len(self.path_lst)} references")

    def __getitem__(self, index):
        name = self.path_lst[index][0].split('/')[-1]
        img = Image.open(self.path_lst[index][0])
        ref = Image.open(self.path_lst[index][1])
        prompt = find_prompt(self.path_lst[index][2], name)
        if len(self.path_lst[index]) == 3:
            tx = img
        else:
            tx = Image.open(self.path_lst[index][-2])

        image = TF.to_tensor(img)  
        ref = TF.to_tensor(ref)  
        tx = TF.to_tensor(tx)

        image, ref = data_aug(image, self.image_size), data_aug(ref, self.image_size)
        tx = data_aug(tx, self.image_size)

        return {
            'image': image,
            'ref': ref,
            'tx': tx,
            'name': name,
            'prompt': prompt
        }

    def __len__(self):
        return len(self.path_lst)


class UIE_val(Dataset):
    def __init__(self, val_lst, image_size):
        self.path_lst = val_lst
        self.image_size = image_size
        print(f"got {len(self.path_lst)} images,{len(self.path_lst)} references")

    def __getitem__(self, index):
        name = self.path_lst[index][0].split('/')[-1]
        img = Image.open(self.path_lst[index][0])
        ref = Image.open(self.path_lst[index][1])
        prompt = 'An underwater image'

        image = TF.to_tensor(img)  
        ref = TF.to_tensor(ref)  

        image, ref = data_aug(image, self.image_size), data_aug(ref, self.image_size)

        return {
            'image': image,  
            'ref': ref,
            'name': name,
            'prompt': prompt
        }

    def __len__(self):
        return len(self.path_lst)


def get_obj_from_string(string):
    module, clas = string.rsplit('.', 1)
    return getattr(importlib.import_module(module, package=None), clas)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def log_config(f, config):
    for k, v in config.items():
        f.write(f'{k}: {v}\n')

def normalize_img(img):
    if torch.max(img) > 1 or torch.min(img) < 0:
        b, c, h, w = img.shape
        temp_img = img.view(b, c, h * w)
        im_max = torch.max(temp_img, dim=2)[0].view(b, c, 1)
        im_min = torch.min(temp_img, dim=2)[0].view(b, c, 1)
        temp_img = (temp_img - im_min) / (im_max - im_min + 1e-7)

        img = temp_img.view(b, c, h, w)

    return img

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
        image = val_data_aug(image,self.image_size)
        
        return {
            'image': image,
            'name':name,
            'prompt': prompt
        }
    def __len__(self):
        return len(self.path_lst)
        
def get_file_lst_c60(file_path):
    file_list = os.listdir(file_path)
    num_train = len(file_list)
    path_lst = []
    for f in file_list:
        f_path = os.path.join(file_path,f)
        path_lst.append((f_path,))
    train_lst = path_lst
    return train_lst
    
def get_file_lst_val(file_path, gt_path, pr_path):
    file_list = os.listdir(file_path)
    num_train = len(file_list)
    path_lst = []
    for f in file_list:
        f_path = os.path.join(file_path, f)
        gt_f_path = os.path.join(gt_path, f)
        path_lst.append((f_path, gt_f_path, pr_path))
    train_lst = path_lst
    return train_lst


def val_data_aug(image, re_size=None):
    if re_size:
        image = TF.resize(image, (re_size, re_size))
    return image