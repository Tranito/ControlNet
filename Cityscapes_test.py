from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from Cityscapes_dataset_customizable import CityscapesDataset
# # from cityscapesdatasetcreator_gtadatasetloader_test import GTADataset
from GTA_dataset import GTADataset
from cldm.logger import TestImageLogger
from cldm.model import create_model, load_state_dict

############################# Parameters you can change #########################

# Default location for saved images: "./results"
name = 'inference_on_val_crops_lr_4e-5/GTA_SD_LOCKED_STREET_VIEW' # name of the directory to which images will be saved
train=False # if true, run inference on the training set, if false, use validation set
crop=True # if true, run inference on crops of input images

# Checkpoint: Load the checkpoint you want to use for inference
checkpoint = '/home/long/ControlNet-for-BEP/checkpoints_Dan/SD_Locked_epoch=781-step=19549.ckpt'
################################################################################

batch_size = 1
logger_freq = 1
sd_locked = False
only_mid_control = False
parts=False # if true, use part-aware cityscapes (you can ignore this)


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15_test.yaml').cpu()
model.load_state_dict(load_state_dict(checkpoint, location='cpu'))
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
prompt_name='train_prompts.json' if train else 'val_prompts_city6k.json'
#Index 8793 is the 8794rd JSON object in the val_promps2.json file
#The index of the generated images of ControlNet corresponds to the index of the corresponding path to the image in the JSON object in the JSON file val_prompts2.json
val_dataset = GTADataset(prompt_name, crop=crop, parts=parts, resume = 0)
val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=1, shuffle=False)
logger = TestImageLogger(name, batch_frequency=logger_freq, resume_jsonidx = 0)
trainer = pl.Trainer(accelerator='gpu', devices=1, strategy='ddp', precision=32, callbacks=[logger])


# Train!
trainer.validate(model, val_dataloader)
