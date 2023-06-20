import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import numpy as np
import wandb
import segmentation_metric
from hungarian_match import HungarianMatcher
import time
import utils
import ipdb
st = ipdb.set_trace


def build_grid_encoder(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).cuda()



class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128, pos_dims=0):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))

        self.feat_dim = dim
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)


        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        hidden_dim = max(dim, hidden_dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)

        self.norm_input  = nn.LayerNorm(dim)




    def forward(self, inputs):
        b, n, d = inputs.shape        
        
        slots = self.slots_mu.repeat([b,1,1])


        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        all_attn_slot = []
        all_attn = []


        for iter_num in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale

            attn = dots.softmax(dim=1) + self.eps
            attn_slot = attn
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            all_attn.append(attn)

            all_attn_slot.append(attn_slot)

            updates = torch.einsum('bjd,bij->bid', v, attn)                
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, all_attn, all_attn_slot


"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid_encoder(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim, in_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)              
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)            
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        return x


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super(ResnetBlockFC, self).__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx




class ImplicitMLP2DDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    def __init__(self, dim=2, c_dim=64,
                 hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=1, grid_there = False, resolution=None):
        super(ImplicitMLP2DDecoder, self).__init__()
        print('Implicit Local Decoder...')
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.xyz_grid = self.build_grid2D_imp(resolution)
        self.xyz_grid = self.xyz_grid*(resolution[0]-1 )
        self.fc_p = nn.Linear(dim, hidden_size)
        self.resolution = resolution

        self.fc_c = nn.ModuleList([
            nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
        ])

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])


        self.fc_out = nn.Linear(hidden_size, out_dim)

        self.out_dim = out_dim
        
        self.actvn = F.relu

        self.padding = padding



    def build_grid2D_imp(self,resolution):
        ranges = [np.linspace(0., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(grid).cuda()


    def forward(self, featmap):
        B = featmap.shape[0]

        pcl_mem = self.xyz_grid
        pcl_mem_ = pcl_mem.reshape([1,-1,2]).repeat([B,1,1])
        pcl_norm = (pcl_mem_/self.resolution[0]) -0.5
            
        net = self.fc_p(pcl_norm)
        c = featmap.unsqueeze(dim=1).repeat(1,pcl_norm.shape[1],1)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net)).permute(0,2,1)
        out = out.reshape(B, self.out_dim, self.resolution[0], self.resolution[1])        
        return out



class OccLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, gt_vox_grid, p_vox_grids, steps=0, fix_pos_weight=0.0):
        pos_examples = torch.sum(gt_vox_grid)
        neg_examples = gt_vox_grid.numel() - pos_examples
        pos_weight = (neg_examples+ 1)/(pos_examples+ 1)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.detach())
        prob_loss = criterion(p_vox_grids,gt_vox_grid)

        return prob_loss

def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    assert(B==B_)
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B*S]+otherdims)
    return tensor

def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert(BS%B==0)
    otherdims = shapelist[1:]
    S = int(BS/B)
    tensor = torch.reshape(tensor, [B,S]+otherdims)
    return tensor




class ModelIter(nn.Module):
    def __init__(self, opt):
        super(ModelIter, self).__init__()
        self.device = "cuda"
        self.opt = opt
        feat_dim = opt.feat_dim
        input_dim = opt.input_dim

        resolution = [opt.image_height,opt.image_width]
        self.encoder_cnn = Encoder(resolution, feat_dim, input_dim)

        slot_featdim = opt.feat_dim
        num_slots = opt.num_slots

        num_iterations = opt.num_iterations

        self.do_tta = opt.do_tta
        self.num_slots = num_slots

        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=slot_featdim,
            iters = num_iterations,
            eps = 1e-8, 
            hidden_dim = slot_featdim)
        
        decoder_dim = opt.feat_dim

        self.decoder_cnn = ImplicitMLP2DDecoder(c_dim=decoder_dim, n_blocks=4,hidden_size=decoder_dim, out_dim=opt.decoder_num_blocks,resolution=resolution).cuda()

        self.mse_loss = nn.MSELoss()

        self.occ_loss = OccLoss()

        self.hungarianMatcher = HungarianMatcher()

    def forward(self, feed, step):
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: pack_seqdim(x, B)
        __u = lambda x: unpack_seqdim(x, B)
        

        rgb_image =  feed['image']
        seg_image = feed['gt_mask']
        gt_indices = feed['gt_indices']

        vis_dict = {}

        B = rgb_image.shape[0]

        total_loss = torch.tensor(0.0).cuda()

        if step % self.opt.log_freq == 0:
            vis_dict["gt_rgb"] =  wandb.Image(rgb_image[:1] +0.5, caption="input RGB image")
            gt_mask_vis = utils.summ_instance_masks(seg_image[0].squeeze())
            vis_dict["gt_mask"] = wandb.Image(gt_mask_vis[:1] + 0.5, caption="input GT mask")

        input_feats  = self.encoder_cnn(rgb_image)
        input_feats_ = input_feats.flatten(1,2)

        slots, all_attn, all_attn_slot = self.slot_attention(input_feats_)
        slots_ = __p(slots)

        rgb_mask_ = self.decoder_cnn(slots_)
        rgb_mask = __u(rgb_mask_).permute(0,1,3,4,2)


        masks = rgb_mask[:,:,...,3:]
        recons = rgb_mask[:,:,...,:3]
        pred_masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * pred_masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)                        

        rgb_loss = self.mse_loss(recon_combined, rgb_image)
        rgb_loss = rgb_loss * self.opt.rgb_loss_coeff
        vis_dict["reconstruction_loss"] = rgb_loss
    
        total_loss = total_loss + rgb_loss

        # log mask and rgb
        pred_mask_vis = utils.summ_instance_masks(pred_masks[0].squeeze(),pred=True)

        if step % self.opt.log_freq == 0:
            vis_dict["pred_rgb"] = wandb.Image(
                recon_combined[:1] + 0.5, caption="pred RGB image")
            vis_dict["pred_mask"] = wandb.Image(
                pred_mask_vis[:1] + 0.5, caption="pred mask")


        # segmentation loss
        gt_mask = seg_image.unsqueeze(2)
        pred_masks = pred_masks.squeeze(-1).unsqueeze(2)

        pred_masks = pred_masks.flatten(2,4)
        pred_height = int(pred_masks.shape[-1]**0.5)
        
        gt_mask = gt_mask.reshape(pred_masks.shape)
        gt_mask = gt_mask*gt_indices.unsqueeze(-1)

        gt_mask_neg = (1.0-gt_mask)*gt_indices.unsqueeze(-1)
        num_pos_classes = torch.sum(gt_mask)
        num_neg_classes = torch.sum(gt_mask_neg)

        pos_weight = num_neg_classes/(num_pos_classes+1e-6)
        gt_mask_w = gt_mask*pos_weight
        total_w = gt_mask_w + gt_mask_neg
        
        new_indices = self.hungarianMatcher(gt_mask.squeeze(2),pred_masks, use_mm=True)

        pred_mask_indices = torch.stack(new_indices,0)[:,1].flatten()
        gt_mask_indices = torch.stack(new_indices,0)[:,0].flatten()
        batch_indices = torch.arange(B).unsqueeze(1).repeat(1, self.num_slots).flatten()
        
        
        gt_mask_ra = gt_mask[batch_indices, gt_mask_indices].reshape([B, self.num_slots,-1])
        total_w_ra = total_w[batch_indices, gt_mask_indices].reshape([B, self.num_slots,-1])
        pred_masks_ra = pred_masks[batch_indices, pred_mask_indices].reshape([B, self.num_slots,-1])
        criterion_occ = nn.BCELoss(reduction='none')

        mask_occ_loss = criterion_occ(pred_masks_ra, gt_mask_ra)
        mask_occ_loss = mask_occ_loss*total_w_ra
        mask_occ_loss = torch.sum(mask_occ_loss)/(torch.sum(total_w_ra) +1e-6)     
        
        if not self.do_tta:
            mask_occ_loss = mask_occ_loss * self.opt.mask_loss_coeff
            vis_dict["segmentation_loss"] = mask_occ_loss
            total_loss = total_loss + mask_occ_loss

        
        # ari segmentation metrics
        gt_mask_ra_ari = gt_mask_ra.reshape(gt_mask_ra.shape[0],gt_mask_ra.shape[1],-1).permute(0,2,1)
        pred_masks_ra_ari = pred_masks_ra.reshape(pred_masks_ra.shape[0],pred_masks_ra.shape[1],-1).permute(0,2,1)

        fg_gt_mask_ra_ari = gt_mask_ra_ari[:,:,1:]
        fg_seg_scores = segmentation_metric.adjusted_rand_index(fg_gt_mask_ra_ari, pred_masks_ra_ari)
        fg_seg_scores = torch.tensor([score for score in  fg_seg_scores if score.isfinite()]).mean()
        vis_dict["fg_ari_score"] = fg_seg_scores
        

        seg_scores = segmentation_metric.adjusted_rand_index(gt_mask_ra_ari, pred_masks_ra_ari)
        seg_scores = torch.tensor([score for score in  seg_scores if score.isfinite()]).mean()
        vis_dict["ari_score"] = seg_scores
                    
        return total_loss, vis_dict
