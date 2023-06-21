from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from Cityscapes_dataset_customizable import CityscapesDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint 

############################# Parameters you can change #########################

name = 'test_run_5_lr_4e-5' # name of the folder in which training samples will be stored (within '/log' directory)

crop=False # whether or not to use crop-based training
sd_locked = True # if False, finetune stable diffusion weights. Refer to tutorial on ControlNet github for details
max_epochs=800 # number of epochs to train for

# If training from scratch, use './models/control_sd15_ini.ckpt'
resume_path = '/home/long/ControlNet-for-BEP/log/test_run_2_lr_4e-5/epoch=19-step=9919.ckpt'
# Original batch size = 6
batch_size =  6 # might need to be adjusted according to GPU capabilities
logger_freq = 400 # how frequently (in iterations) to store training samples while training (in image_log directory)
learning_rate = 4e-5
only_mid_control = False # if true, it makes training faster, but less good. Refer to tutorial on ControlNet github for details

# By default, checkpoints are saved in 'log' directory
################################################################################


cs_in_prompt = not sd_locked #For crop-based training, should the keyword 'A CityScapes image of' be included in the prompt
parts = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
train_dataset = CityscapesDataset('train_prompts.json',crop=crop, parts=parts, cs_in_prompt=cs_in_prompt)
print("train_prompts.json loaded!")
train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
val_dataset = CityscapesDataset('val_prompts.json', crop=crop, parts=parts)
print("val_prompts.json loaded!")
val_dataloader = DataLoader(val_dataset, num_workers=2, batch_size=batch_size, shuffle=False)
logger = ImageLogger(name, batch_frequency=logger_freq)
val_checkpoint_callback = ModelCheckpoint(dirpath="./log/"+name, save_top_k=1, monitor="val/loss")
train_checkpoint_callback = ModelCheckpoint(dirpath="./log/"+name, save_top_k=0, monitor="train/loss")
trainer = pl.Trainer(accelerator='gpu', devices=1, strategy='ddp', precision=32, callbacks=[logger, val_checkpoint_callback, train_checkpoint_callback], max_epochs=max_epochs)


# Train!
print("STARTING TRAINING")
trainer.fit(model, train_dataloader, val_dataloader)

