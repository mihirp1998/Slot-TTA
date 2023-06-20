import pickle
import socket
from PIL import Image
import glob
import torchvision
import numpy as np
import ipdb
import torch.nn.functional as F
st = ipdb.set_trace
import torch
import time

class ClevrDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        root_file = opt.root_folder
        self.all_files = glob.glob(f'{root_file}/clevr_train/*')

        self.resize = torchvision.transforms.Resize((opt.image_height,opt.image_width))
        self.resize_mask = torchvision.transforms.Resize((opt.image_height,opt.image_width),torchvision.transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        if self.opt.overfit:
            idx = 0

        file_val = self.all_files[idx]
        pickled_file = pickle.load(open(file_val,'rb'))
        
        rgb_val = torch.from_numpy(pickled_file['image']).squeeze().float()
        images = rgb_val / 256.0 
        # Normalize to [0, 1] range.
        gt_mask_val = torch.from_numpy(np.argmax(pickled_file['mask'].squeeze(),0))
        
        images = images.permute(2,0,1).unsqueeze(0)
        images = self.resize(images)
        gt_mask_val = self.resize_mask(gt_mask_val.unsqueeze(0))
        
        
        images = images.squeeze()
        gt_mask_val = gt_mask_val.squeeze()
        max_objs = gt_mask_val.max()
        
        gt_indices = torch.zeros(self.opt.num_slots)
        gt_indices[:max_objs] = 1.0
        gt_mask_val = F.one_hot(gt_mask_val, self.opt.num_slots).permute(2,0,1)

        return images, gt_mask_val , gt_indices

class ClevrTexDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        root_file = opt.root_folder
        self.all_files = glob.glob(f'{root_file}/clevr_tex/*')
        self.resize = torchvision.transforms.Resize((opt.image_height,opt.image_width))
        self.resize_mask = torchvision.transforms.Resize((opt.image_height,opt.image_width),torchvision.transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        if self.opt.overfit:
            idx = 0
        
        if self.opt.specific_example != 'None':
            idx = int(self.opt.specific_example)
        
        file_val = self.all_files[idx]
        pickled_file = pickle.load(open(file_val,'rb'))
        
        
        rgb_val = torch.from_numpy(pickled_file['image']).squeeze().float()
        images = rgb_val / 256.0 
        # Normalize to [0, 1] range.
        gt_mask_val = torch.from_numpy(np.argmax(pickled_file['mask'].squeeze(),0))
        
        images = images.permute(2,0,1).unsqueeze(0)
        images = self.resize(images)
        
        gt_mask_val = self.resize_mask(gt_mask_val.unsqueeze(0))
        
        
        images = images.squeeze()
        gt_mask_val = gt_mask_val.squeeze()
        max_objs = gt_mask_val.max()

        gt_indices = torch.zeros(self.opt.num_slots)
        gt_indices[:max_objs] = 1.0
        gt_mask_val = F.one_hot(gt_mask_val, self.opt.num_slots).permute(2,0,1)

        return images, gt_mask_val , gt_indices



def get_dataloader(opt, dataset):
    # Improve reproducibility in dataloader.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=opt.num_workers,
        persistent_workers=opt.persistent_worker,
    )
    
    iterator = iter(data_loader)
    return data_loader, iterator


def get_data(opt):
    if opt.dataset_name == "clevr_tex":
        dataset = ClevrTexDataset(opt)         
    else:
        dataset = ClevrDataset(opt)
    loader, iterator = get_dataloader(opt, dataset)
    return loader, iterator

def get_data_tta(opt):
    if opt.dataset_name == "clevr_tex":
        dataset = ClevrTexDataset(opt)
    else:
        dataset = ClevrDataset(opt)
    return dataset

def get_input(opt, iterator, train_loader):
    time_init = time.time()
    try:
        input = next(iterator)
    except StopIteration:
        iterator = iter(train_loader)
        input = next(iterator)
        print("stop loading")
    time_init = time.time()
    image, gt_mask, gt_indices = input 
    
    
    image = image.to(opt.device)
    gt_mask = gt_mask.to(opt.device)
    gt_indices = gt_indices.to(opt.device)
    
    feed_dict = {}
    feed_dict["image"] = image
    feed_dict["gt_mask"] = gt_mask
    feed_dict["gt_indices"] = gt_indices
    
    return  feed_dict
