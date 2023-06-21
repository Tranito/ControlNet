import json
import cv2
import numpy as np
import torchvision.transforms as tf
import os
from PIL import Image

from torch.utils.data import Dataset
# resume = 0

class GTADataset(Dataset):
    def __init__(self, 
                 prompt_name='train_prompts.json', 
                 crop=False, 
                 parts=False, #part-aware cityscapes?
                 prompt_model=False, #True if we want clip (fast) interrogator to generate prompts
                 num_samples=-1, #how long dataset should be: -1 = whole dataset
                 cs_in_prompt=False,
                 ): # for unlocking SD, add 'a cityscapes image containing ...' to instance prompt
        self.data = []

        self.mode=prompt_name.split('_')[1] if len(prompt_name.split('_'))>2  else ''
        prompt_path='./testing/gta_instance_prompts/'+prompt_name
    
        with open(prompt_path, 'rt') as f:
            for line in f:
                #Convert the JSON objects that have keys and the corresponding values to a Python dictionary that have the identical keys and the corresponding values as well
                self.data.append(json.loads(line))

        if num_samples!=-1: self.data = self.data[0:num_samples]    

        self.parts=parts
        self.crop=crop
        self.prompt_model=prompt_model
        self.cs_in_prompt=cs_in_prompt

        assert self.prompt_model == False, 'this implementation of ControlNet does not support generating prompts with Interrogator'

        #From the GTA dataset only the trainid labelmaps are provided to the denoising diffusion probabilistic model instead of both the images and the corresponding trainid labelmaps from which the trainid labelsmaps are provided in the full dimension to the denoising diffusion probabilistic model
        


    def __len__(self):
        return len(self.data)

    def crop_params(self, short_side): #output size is 256x512
        top=np.random.randint(0, short_side-256) if short_side!=256 else 0
        left=np.random.randint(0, 2*(short_side-256)) if short_side!=256 else 0
        height=256
        width=512
        return top,left,height,width

    def __getitem__(self, idx):
        item = self.data[idx]

        label_filename = item['source']
        img_filename = item['target']
        prompt = item['prompt'] 
        
        label = cv2.imread(label_filename)
        img = cv2.imread(img_filename)

        # Do not forget that OpenCV read images in BGR order.
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # NO IMAGES ARE OBTAINED FROM THE GTA DATASET
        #resize images
        if self.crop:
            # resize to random dimension between 256x512 and 1024x2048
            scaling = np.random.uniform(1,4) # random float between 1 and 4
            short_side=round(scaling*256)
            label = cv2.resize(label, (2*short_side,short_side), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, (2*short_side,short_side), interpolation=cv2.INTER_LINEAR)
            
            # find cropping parameters that are consistent across img and label
            top,left,height,width = self.crop_params(short_side)

            #crop images:
            label = label[top:top+height, left:left+width]
            img = img[top:top+height, left:left+width]

        else:
            label = cv2.resize(label, (512,256), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, (512,256), interpolation=cv2.INTER_LINEAR)

        # Normalize label images to [0, 1].
        label = label.astype(np.float32) / 255.0

        # Normalize img images to [-1, 1].
        img = (img.astype(np.float32) / 127.5) - 1.0

        assert img.shape[0]==256 and img.shape[1]==512, 'wrong dimensions of cropped image'

        return dict(jpg = img, txt=prompt, hint=label)

