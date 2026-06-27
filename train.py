from logging import logProcesses
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
import torch.distributed as dist  
from torch.nn.parallel import DistributedDataParallel as DDP  
from torch.utils.data import DataLoader, DistributedSampler  
import os
import torch.cuda.amp as amp
import numpy as np
import random
import time
from PIL import Image
import argparse
import os.path as osp
import yaml
from utils import AverageMeter, get_obj_from_string, UIE_train, UIE_val, get_file_lst
from utils import normalize_img, log_config, get_file_lst_c60, UIE_c60, SSIMLoss, percep_loss, CLIPSimilarityAlignmentLoss
import importlib
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid
import clip
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default=0, help="path experiment config")
parser.add_argument('--debug', type=bool, default=False, help="debug or not")
parser.add_argument('--resize_val', type=bool, default=True, help="resize validation or not")
args = parser.parse_args()


dist.init_process_group(backend='nccl', init_method='env://')
local_rank = int(os.environ["LOCAL_RANK"])  
torch.cuda.set_device(local_rank)
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

rank = dist.get_rank()
world_size = dist.get_world_size()


with open(args.exp, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

config['model_name'] = config['model_file'].split('.')[1]
weight_decay = config['weight_decay']


if args.debug:
    log_path = f"./debug/{config['name']}_{config['model_name']}_lr{config['lr']}_wd{config['weight_decay']}_epoch{config['num_epochs']}_log/"
    snapshot_path = osp.join(log_path, 'weights')
else:
    log_path = f"./exp_results/{config['name']}_{config['model_name']}_lr{config['lr']}_wd{config['weight_decay']}_epoch{config['num_epochs']}_log/"
    snapshot_path = osp.join(log_path, 'weights')

if rank == 0:
    os.makedirs(snapshot_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    f_log = open(osp.join(log_path, 'save_log.txt'), 'w')
    f_log.write(f'experiment config file: {args.exp}\n')
    for k, v in config.items():
        f_log.write(f'{k}: {v}\n')
    f_log.flush()

# 数据集路径
if 'LUIQD' in config['name']:
    tr_input_path = "-"
    tr_gt_path = "-"
    val_input_path = "-"
    val_gt_path = "-"
    ts_input_path = "-"
    ts_gt_path = "-"
    prompt_path = "-"

if rank == 0:
    print(f"[Rank 0] Loading train dataset from input path:{tr_input_path} --- gt path:{tr_gt_path}")
train_lst = get_file_lst(tr_input_path, tr_gt_path, prompt_path)
train_set = UIE_train(train_lst, image_size=config['img_size'])
train_bs = config['batch_size']
train_sampler = DistributedSampler(train_set, shuffle=True)
train_loader = DataLoader(train_set, batch_size=train_bs, sampler=train_sampler, num_workers=8, pin_memory=True)

if rank == 0:
    print(f"[Rank 0] Loading val dataset from val input:{val_input_path}--- val gt:{val_gt_path}")
val_lst = get_file_lst(val_input_path, val_gt_path, prompt_path)
if args.resize_val:
    val_set = UIE_val(val_lst, image_size=config['img_size'])
else:
    val_set = UIE_val(val_lst, image_size=None)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

in_channels = 3
num_class = 3
num_epochs = config['num_epochs']
lr = config['lr']
scaler = amp.GradScaler(enabled=False)

if rank == 0:
    print(f"Start loading CLIP")      
# CLIP权重路径 
clip_model, _ = clip.load("-", device=device, jit=False)  
clip_model.eval()   
if rank == 0:
    print(f"Finish loading CLIP")


net = get_obj_from_string(config['model_file'])(in_channels, clip_model=clip_model).to(device)
net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()
criterion_ssim = SSIMLoss()
criterion_per = percep_loss(device=device)

scaler = torch.cuda.amp.GradScaler(enabled=False)

writer = None
if rank == 0:
    writer = SummaryWriter(osp.join(log_path, 'tensorboard'))

def train(net, num_epochs):
    num_iter = 0
    best_val_loss = 1000
    best_val_loss2 = 1000
    best_psnr = -100
    best_ssim = -100

    for epoch in range(num_epochs):
        net.train()
        train_sampler.set_epoch(epoch)

        total_num = 0
        val_total_num = 0
        total_mse_loss = 0
        val_total_mse_loss = 0
        tr_loss_meter = AverageMeter()

        tic = time.time()  
        for data in tqdm(train_loader, disable=(rank != 0)):  
            optimizer.zero_grad()
            image = data['image'].to(device)
            label = data['ref'].to(device)
            prompt = data['prompt']
            name = data['name']
            text_tokens = clip.tokenize(prompt).to(device)  
            pred_ = net(image, text_tokens)
            
            if isinstance(pred_, tuple):
                pred = pred_[-1]
            else:
                pred = pred_                                    
            loss_mse = criterion(pred, label)  
            loss_ssim = criterion_ssim(pred, label)
            loss_per = criterion_per(pred, label)
            alpha = 0.1 
            beta = 0.0001
            loss = loss_mse+loss_ssim+(alpha*loss_per) 
            tr_loss_meter.update(loss.item(), len(label))
            loss.backward(retain_graph=True)
            optimizer.step()
        toc = time.time()  
        
        if rank == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] loss={tr_loss_meter.avg:.5f}, time={toc - tic:.2f}")
            if (epoch + 1) % 10 == 0:
                if writer is not None:
                    writer.add_scalar('Loss/train', tr_loss_meter.avg, epoch)                

        net.eval()
        val_loss_meter = AverageMeter()
        val_psnr_meter = AverageMeter()
        val_ssim_meter = AverageMeter()
        
        tic = time.time()  
        with torch.no_grad():
            for data in tqdm(val_loader, disable=(rank != 0)):
                image = data['image'].to(device)
                label = data['ref'].to(device)
                prompt = data['prompt']
                name = data['name']
                text_tokens = clip.tokenize(prompt).to(device)  
                pred_ = net(image, text_tokens)
                
                if isinstance(pred_, tuple):
                    pred = pred_[-1]
                else:
                    pred = pred_
                                    
                loss_mse = criterion(pred, label)  
                loss_ssim = criterion_ssim(pred, label)
                loss_per = criterion_per(pred, label)

                alpha = 0.1 
                beta = 0.0001
                loss = loss_mse+loss_ssim+(alpha*loss_per) 
                val_loss_meter.update(loss.item())
        toc = time.time()  
        
        if rank == 0:
            print(f"Val: epoch={epoch+1}, loss={val_loss_meter.avg:.5f}, time={toc - tic:.2f}")
            if (epoch + 1) % 10 == 0:
                if writer is not None:
                    writer.add_scalar('Loss/val', val_loss_meter.avg, epoch)
            val_loss = val_loss_meter.avg

            if (epoch + 1) <= 1000:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    state_dict = {
                        'net': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch
                    }
                    torch.save(state_dict, osp.join(snapshot_path, 'best_val.pth'))
                    print(f"save best val model on epoch={epoch+1}, val_loss={best_val_loss:.5f}")
                    f_log.write(f"Best model at epoch={epoch+1}, val_loss={best_val_loss:.5f}\n")
                    f_log.flush()
            else:
                if val_loss < best_val_loss2:
                    best_val_loss2 = val_loss
                    state_dict = {
                        'net': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch
                    }
                    torch.save(state_dict, osp.join(snapshot_path, 'best_val2.pth'))
                    print(f"save best val2 model on epoch={epoch+1}, val_loss={best_val_loss2:.5f}")
                    f_log.write(f"Best2 model at epoch={epoch+1}, val_loss={best_val_loss2:.5f}\n")
                    f_log.flush()

            latest_dict = {
                'net': net.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(latest_dict, osp.join(snapshot_path, 'latest.pth'))
    if rank == 0:
        f_log.close()
        writer.close()


train(net, num_epochs)
dist.barrier()
dist.destroy_process_group()