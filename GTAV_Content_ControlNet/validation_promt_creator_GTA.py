import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm


path_dir_data = "/home/long/mmsegmentation/data/GTA"

path_dir_images = "img_dir"

path_dir_trainids = "ann_dir"

dir_procedure = ["test","train","val"]

val_prompts_GTA = {"source":None, "target": None, "prompt":"A realistic street view image"}

for directory in dir_procedure:

    path_dir_trainid_files = os.path.join(path_dir_data,path_dir_trainids,directory) + "_grey"

    print(path_dir_trainid_files)

    for i, trainid_file_name in tqdm(enumerate( os.listdir(path_dir_trainid_files) )):

        with open("testing/gta_instance_prompts/val_prompts4.json","a") as file:
            #The images and the trainingid labelmaps have identical IDs and file names
            path_to_image_from_dir = os.path.join(path_dir_data, path_dir_images, directory, trainid_file_name)

            path_trainid_file = os.path.join(path_dir_trainid_files,trainid_file_name)
            # print(path_trainid_file)
            val_prompts_GTA["source"] = path_trainid_file
            val_prompts_GTA["target"] = path_to_image_from_dir
            # print(val_prompts_GTA)

            json.dump(val_prompts_GTA, file)
            file.write('\n')


            
