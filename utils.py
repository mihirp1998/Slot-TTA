import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import ipdb 
st = ipdb.set_trace


def get_learning_rate(opt, step):
    if opt.learning_rate_schedule == 0:
        return opt.learning_rate
    elif opt.learning_rate_schedule == 1:
        return get_linear_warmup_lr(opt, step)
    else:
        raise NotImplementedError


def get_linear_warmup_lr(opt, step):
    if step < opt.warmup_steps:
        return opt.learning_rate * step / opt.warmup_steps
    else:
        return opt.learning_rate

def update_learning_rate(optimizer, opt, step):
    lr = get_learning_rate(opt, step)
    optimizer.param_groups[0]["lr"] = lr
    return optimizer, lr


def summ_instance_masks(masks,  pred=False):
    masks = masks.squeeze(1)
    if pred:
        old_shape = masks.shape
        num_slots = masks.shape[0]
        masks = torch.argmax(masks.reshape(masks.shape[0],-1).transpose(1,0),axis=-1)
        masks = F.one_hot(masks,num_slots).float().transpose(1,0).reshape(old_shape)

    num_slots_c = torch.sum(masks.sum([1,2])>0.0)

    farthest_colors = plt.get_cmap("rainbow")([np.linspace(0, 1, num_slots_c)])[:,:,:3][0]
    rgb_canvas = torch.ones([3,masks.shape[-2],masks.shape[-1]])
    start_idx = 0
    for index, mask in enumerate(masks):
        if torch.sum(mask) > 0:
            chosen_color = farthest_colors[start_idx].reshape([3,1])
            start_idx += 1
            indicies = torch.where(mask == 1.0)
            rgb_canvas[:,indicies[0],indicies[1]] = torch.from_numpy(chosen_color).float()
    rgb_canvas = rgb_canvas - 0.5
    rgb_canvas = rgb_canvas.unsqueeze(0)
    return rgb_canvas
