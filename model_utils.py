import models
import os
import torch 
import ipdb
st = ipdb.set_trace

def get_model_and_optimizer(opt):
    model = models.ModelIter(opt)
    model = model.to(opt.device)
    
    if opt.tta_optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9)
    
    if opt.load_folder != "None":
        print("Loading model from", opt.load_folder)
        # st()
        load_path = opt.cwd + "/" + opt.load_folder
        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict["model"])    
        if opt.tta_optimizer == "adam":
            optimizer.load_state_dict(state_dict["optimizer"])

    return model, optimizer



class ModelCheckpoint:
    def __init__(self, save_path, keep=3):
        self.save_path = save_path
        self.keep = keep
        self.checkpoints = []

    def save_checkpoint(self, model, optimizer, train_step):
        checkpoint = {
            'epoch': train_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        path = os.path.join(self.save_path, f'checkpoint_{train_step}.pt')

        # Save the checkpoint.
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

        # Add the checkpoint.
        self.checkpoints.append(path)

        # If there are more than `keep` checkpoints, remove the oldest one.
        if len(self.checkpoints) > self.keep:
            oldest_checkpoint = self.checkpoints.pop(0)
            os.remove(oldest_checkpoint)
            print(f"Checkpoint deleted: {oldest_checkpoint}")
