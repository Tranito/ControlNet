import json
import cv2
import numpy as np
import torchvision.transforms as tf
import os
from PIL import Image

from torch.utils.data import Dataset


class CityscapesDataset(Dataset):
    def __init__(self, 
                 prompt_name='train_prompts.json', 
                 crop=False, 
                 parts=False, #part-aware cityscapes?
                 prompt_model=False, #True if we want clip (fast) interrogator to generate prompts
                 num_samples=-1, #how long dataset should be: -1 = whole dataset
                 cs_in_prompt=False, # for unlocking SD, add 'a cityscapes image containing ...' to instance prompt
                 save_trainId_crops_path='' #if we have part-aware images, we need to save the crops of the labelIds at specified path
                 ):
        self.data = []

        self.mode=prompt_name.split('_')[1] if len(prompt_name.split('_'))>2  else ''
        prompt_path='./training/cityscapes_instance_prompts/'+prompt_name
    
        with open(prompt_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        if num_samples!=-1: self.data = self.data[0:num_samples]

        self.parts=parts
        self.crop=crop
        self.prompt_model=prompt_model
        self.cs_in_prompt=cs_in_prompt
        self.save_traidId_crops_path=save_trainId_crops_path

        assert self.prompt_model == False, 'this implementation of ControlNet does not support generating prompts with Interrogator'

        if crop:
            if 'train' in prompt_name:
                self.panoptic_root='/home/long/mmsegmentation/data/cityscapes/gtFine/cityscapes_panoptic_train'
            else:
                self.panoptic_root='/home/long/mmsegmentation/data/cityscapes/gtFine/cityscapes_panoptic_val'
            panoptic_path=self.panoptic_root+'.json'
            fp = open(panoptic_path)
            panopdict=json.load(fp)
            self.panopdict=np.array(panopdict['annotations']) if num_samples==-1 else np.array(panopdict['annotations'][0:num_samples])
            
            self.imgId_list=np.empty(len(self.data),dtype='<U32')
            for idx,item in enumerate(self.panopdict):
                self.imgId_list[idx]=item['image_id']

            self.categories = np.array(['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'])
            self.plural_categories = np.array(['people', 'riders', 'cars', 'trucks', 'busses', 'trains', 'motorcycles', 'bicycles'])
            self.instance_ids = np.array([24, 25, 26, 27, 28, 31, 32, 33])


    def __len__(self):
        return len(self.data)
    

    def get_prompt_crop(self, panoptic, panoptic_suffix):
        img_id=panoptic_suffix.split('_gtFine')[0]
        panoptic_ids=np.unique(self.rgb2id(panoptic))
        
        seg_info=self.panopdict[self.imgId_list==img_id][0]['segments_info']

        #count instance for each 'thing' class
        instance_count=np.zeros(len(self.categories),dtype=np.uint8) #initialize to 0 instances
        for i in seg_info:
            if (panoptic_ids==i['id']).any():
                instance_count[self.instance_ids==i['category_id']]+=1

        prompt =  self.instances2prompt(instance_count)

        return prompt


    def instances2prompt(self, instance_count):
        string=''
        for i,count in enumerate(instance_count):
            if count!=0:
                if count == 1:
                    string+=str(1)+' ' + self.categories[i] + ', '
                elif count>1:
                    string+=str(count)+' '+ self.plural_categories[i] + ', '
        prompt=string
        if self.cs_in_prompt:
            prompt ='A CityScapes image' if instance_count.sum() == 0 else 'A CityScapes image containing ' + prompt[:-2]
        return prompt
    
    def add2prompt(self, prompt):
        if self.mode == 'dark':
            prompt+=' nighttime, dark, dark skies, streetlights, low visibility'
        elif self.mode == 'foggy':
            prompt+=' fog, foggy, bad vision, low visibility, misty'
        elif self.mode == 'quality':
            prompt+=' high-resolution, high quality, DSLR camera, perfect lighting conditions, photographic, city'
        elif self.mode == 'rainy':
            prompt+=' heavy rainfall, rainy, grey clouds, gray skies, wet road'
        elif self.mode == 'sunny':
            prompt+=' bright, sunny, cleear visibility, blue skies'
        return prompt

        

    def rgb2id(self, color):
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])
    

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
        prompt = item['prompt'] if not self.prompt_model else self.clip.interrogate_fast(Image.open(img_filename).convert('RGB'))
        panoptic_suffix = label_filename.split('/')[-1].replace('_labelTrainIds','_panoptic')

        if self.parts:
            trainIds_path = label_filename
            label_filename = label_filename.replace('gtFine', 'gtFinePanopticParts', 1).replace('_gtFine_labelTrainIds.png', '_labelIds.png')
        print("Label file name : " + label_filename)
        label = cv2.imread(label_filename)
        img = cv2.imread(img_filename)

        # Do not forget that OpenCV read images in BGR order.
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

            if not self.prompt_model:
                # find path to panoptic image and load image
                panoptic_path = os.path.join(self.panoptic_root, panoptic_suffix)
                panoptic = cv2.cvtColor(cv2.imread(panoptic_path), cv2.COLOR_BGR2RGB)
                panoptic = cv2.resize(panoptic, (2*short_side,short_side), interpolation=cv2.INTER_NEAREST)
                panoptic = panoptic[top:top+height, left:left+width] #crop image

                prompt=self.get_prompt_crop(panoptic, panoptic_suffix) #get instance-based prompt
                if self.mode!='':
                    prompt = self.add2prompt(prompt)

        else:
            label = cv2.resize(label, (512,256), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, (512,256), interpolation=cv2.INTER_LINEAR)

        # Normalize label images to [0, 1].
        label = label.astype(np.float32) / 255.0

        # Normalize img images to [-1, 1].
        img = (img.astype(np.float32) / 127.5) - 1.0

        assert img.shape[0]==256 and img.shape[1]==512, 'wrong dimensions of cropped image'


        #if we have part-aware crops, we need to save the labelTrainId images
        if self.save_traidId_crops_path != '' and self.parts:
            trainlabel = cv2.imread(trainIds_path)
            trainlabel = cv2.cvtColor(trainlabel, cv2.COLOR_BGR2GRAY)
            if self.crop:
                trainlabel = cv2.resize(trainlabel, (2*short_side,short_side), interpolation=cv2.INTER_NEAREST)
                trainlabel = trainlabel[top:top+height, left:left+width]#crop image
                os.makedirs(self.save_traidId_crops_path, exist_ok=True)
                success = cv2.imwrite(self.save_traidId_crops_path+str(idx+1)+'.png', trainlabel)
                assert success == True, 'trainlabelmap saving unsuccesful'
 
        return dict(jpg=img, txt=prompt, hint=label)

