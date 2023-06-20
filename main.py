import hydra
from tqdm import tqdm
import time
import ipdb 
import matplotlib.pyplot as plt
st = ipdb.set_trace
import torch
import model_utils
import wandb
import random
import utils
import os
import numpy as np
from dataset import ClevrDataset
import dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from omegaconf import open_dict
from omegaconf import OmegaConf

def parse_args(opt):
    with open_dict(opt):
        opt.log_dir = os.getcwd()
        print(f"Logging files in {opt.log_dir}")
        opt.device = "cuda:0" if opt.use_cuda else "cpu"
        opt.cwd = get_original_cwd()

    if not opt.use_random_seed:
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)
    
    if opt.deep_tta_vis:
        opt.log_freq = 1
    
    print(OmegaConf.to_yaml(opt))
    return opt


def do_tta(opt, model, optimizer, tta_dataset):
    model.eval()
    step =0
    before_tta_acc = []
    after_tta_acc = []
    run_name = wandb.run.name
    
        
    


    before_tta_acc_fg = []
    after_tta_acc_fg = []
    for index_val in tqdm(range(0,len(tta_dataset))):
        all_losses = []
        all_accs = []
        
        if opt.deep_tta_vis:
            folder_name = f"tta_dump/{run_name}/{index_val}"
            gt_rgb_folder_name = f"tta_dump/{run_name}/{index_val}/gt_rgb"
            pred_mask_folder_name = f"tta_dump/{run_name}/{index_val}/pred_mask"
            pred_rgb_folder_name = f"tta_dump/{run_name}/{index_val}/pred_rgb"
            os.makedirs(folder_name, exist_ok=True)
            os.makedirs(gt_rgb_folder_name, exist_ok=True)
            os.makedirs(pred_mask_folder_name, exist_ok=True)
            os.makedirs(pred_rgb_folder_name, exist_ok=True)

        for tta_step in tqdm(range(opt.tta_steps)):
            images, gt_mask_val , gt_indices = tta_dataset[index_val]
            images, gt_mask_val , gt_indices = (images.unsqueeze(0).to(opt.device),gt_mask_val.unsqueeze(0).to(opt.device),gt_indices.unsqueeze(0).to(opt.device))    
            feed_dict = {}
            feed_dict["image"] = images
            feed_dict["gt_mask"] = gt_mask_val
            feed_dict["gt_indices"] = gt_indices

            if tta_step ==0:
                with torch.no_grad():
                    model.eval()
                    loss, vis_dict = model(feed_dict, step)
                    before_tta_acc.append(vis_dict["ari_score"])
                    before_tta_acc_fg.append(vis_dict["fg_ari_score"])            
            
            learning_rate = optimizer.param_groups[0]['lr']
            feed_dict["learning_rate"] = learning_rate
            
            loss, vis_dict = model(feed_dict, step)

            if opt.deep_tta_vis:
                vis_dict['gt_rgb'].image.save(f"{gt_rgb_folder_name}/{tta_step:04d}.png")
                vis_dict['pred_mask'].image.save(f"{pred_mask_folder_name}/{tta_step:04d}.png")
                vis_dict['pred_rgb'].image.save(f"{pred_rgb_folder_name}/{tta_step:04d}.png")
            
            all_losses.append(vis_dict['reconstruction_loss'])
            all_accs.append(vis_dict['ari_score'])



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            vis_dict["learning_rate"] = learning_rate
            
            step += 1

            if tta_step == opt.tta_steps-1:
                with torch.no_grad():
                    model.eval()
                    loss, vis_dict = model(feed_dict, step)
                    after_tta_acc.append(vis_dict["ari_score"])
                    after_tta_acc_fg.append(vis_dict["fg_ari_score"])                
                    vis_dict["before_tta_mean_acc"] = np.array(before_tta_acc).mean()
                    vis_dict["before_tta_mean_acc_fg"] = np.array(before_tta_acc_fg).mean()                                
                    vis_dict["after_tta_mean_acc"] = np.array(after_tta_acc).mean()
                    vis_dict["after_tta_mean_acc_fg"] = np.array(after_tta_acc_fg).mean()                                
                    wandb.log(vis_dict, step=index_val)
                
                if opt.deep_tta_vis:
                    min_loss = float(min(all_losses)) - 0.0025
                    max_loss = float(max(all_losses)) + 0.0025

                    min_acc = float(min(all_accs)) - 0.05
                    max_acc = float(max(all_accs)) + 0.05

                    all_losses = [float(i) for i in all_losses]
                    all_accs = [float(i) for i in all_accs]
                    all_steps = list(range(len(all_losses)))
                    
                    
                    for tta_step in tqdm(range(len(all_losses))):
                        # log reconstruction loss
                        all_custom_losses = all_losses[:tta_step]
                        all_custom_accs = all_accs[:tta_step]
                        all_custom_steps = all_steps[:tta_step]

                        plt.figure(dpi=250, figsize=(4.4, 3.8))
                        plt.xlim(-5, 150)
                        plt.ylim(min_loss,max_loss)
                        plt.plot(all_custom_steps, all_custom_losses,  markersize=1,color='red')
                        plt.xlabel('Test-time Adaptation Steps')
                        plt.ylabel('Reconstruction Loss')
                        os.makedirs(f'{folder_name}/loss_tta/', exist_ok=True)

                        plt.tight_layout(pad=0)
                        plt.savefig(f'{folder_name}/loss_tta/{tta_step:05d}_loss_tta.png', bbox_inches='tight')
                        
                        # log segmentation accuracy
                        plt.figure(dpi=250, figsize=(4.4, 3.8))
                        plt.xlim(-5, 150)
                        plt.ylim(min_acc, max_acc)
                        plt.plot(all_custom_steps, all_custom_accs,  markersize=1)
                        plt.xlabel('Test-time Adaptation Steps')
                        plt.ylabel('Segmentation Accuracy')
                        os.makedirs(f'{folder_name}/seg_tta/', exist_ok=True)
                        plt.tight_layout(pad=0)
                        plt.savefig(f'{folder_name}/seg_tta/{tta_step:05d}_seg_tta.png', bbox_inches='tight')

                all_losses = []
                all_accs = []
                
        if opt.specific_example != "None":
            break

        model, optimizer = model_utils.get_model_and_optimizer(opt)
    

def train(opt, model, optimizer, train_iterator, train_loader, checkpoint):
    start_time = time.time()

    for step in tqdm(range(opt.training_steps + 1)):
        vis_dict = {}
        time_init = time.time()
        
        feed_dict = dataset.get_input(opt, train_iterator, train_loader)

        optimizer, learning_rate = utils.update_learning_rate(optimizer, opt, step)            
        
            
        feed_dict["learning_rate"] = learning_rate
        loss, vis_dict = model(feed_dict, step)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if step % opt.log_freq == 0:
            vis_dict["learning_rate"] = learning_rate
            wandb.log(vis_dict, step=step)
        
        if step != 0 and step % opt.save_freq ==0:
            checkpoint.save_checkpoint(model, optimizer, step)
    checkpoint.save_checkpoint(model, optimizer, train_step)
    return optimizer, step




@hydra.main(config_path="config", config_name="config")
def my_main(opt: DictConfig) -> None:
    opt = parse_args(opt)


    wandb.login()
    wandb.init(project="slot-tta", config=opt)    

    run_name = wandb.run.name
    
    opt.log_dir = f"{opt.cwd}/checkpoint/{run_name}"
    os.makedirs(opt.log_dir, exist_ok=True)        

    model, optimizer = model_utils.get_model_and_optimizer(opt)
    
    checkpointer = model_utils.ModelCheckpoint(opt.log_dir, keep=10)
    
    if opt.do_tta:
        tta_dataset = dataset.get_data_tta(opt)
        do_tta(opt, model, optimizer, tta_dataset)
    else:
        train_loader, train_iterator = dataset.get_data(opt)
        train(opt, model, optimizer, train_iterator, train_loader, checkpointer)

if __name__ == "__main__":
    my_main()
