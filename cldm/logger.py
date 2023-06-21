import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, name, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.name=name

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", self.name)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale and k!='control':
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "gs-{:06}_e-{:04}_{}.png".format(global_step, current_epoch, k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")





############ Used only by trainer.validate() in Cityscapes_test.py



class TestImageLogger(Callback):
    def __init__(self, name, batch_frequency=2000, max_images=1, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, resume_jsonidx = 0):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.name=name
        self.resume_jsonidx = resume_jsonidx

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        for k in images:
            if self.rescale:
                if k!='control':
                    images[k] = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            images[k] = images[k].squeeze()
            images[k] = np.rollaxis(images[k].numpy(), 0,3)
            if k=='control': print(f'in save function: {np.mean(images[k])}')
            images[k] = (images[k] * 255).astype(np.uint8)
        
        REF = images['control']
        GT = images['reconstruction']
        HR = images['samples_cfg_scale_9.00']

        root = os.path.join(save_dir, "results", self.name)
        for folder_name in ['REF','GT','HR']:
            if not os.path.exists(os.path.join(root, folder_name)): 
                os.makedirs(os.path.join(root, folder_name), exist_ok=True)

        path_REF=os.path.join(root, 'REF', str(batch_idx+self.resume_jsonidx+1)+'.png')
        path_GT=os.path.join(root, 'GT', str(batch_idx+self.resume_jsonidx+1)+'.png')
        path_HR=os.path.join(root, 'HR', str(batch_idx+self.resume_jsonidx+1)+'.png')
        Image.fromarray(REF).convert('L').save(path_REF)    #save label map as grayscale, not RGB
        Image.fromarray(GT).save(path_GT)
        Image.fromarray(HR).save(path_HR)
        if not os.path.exists(os.path.join(root, 'example_prompt')): #only save one prompt image
            Image.fromarray(images['conditioning']).save(os.path.join(root, 'example_prompt.png'))

        # with open(os.path.join(root,'prompts.txt'), 'w') as fp:
        #     fp.write(images['conditioning']+'\n')

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx)
