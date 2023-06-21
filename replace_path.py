import os
import json

# _train_prompts_path = "/home/long/ControlNet-for-BEP/training/cityscapes_instance_prompts/train_prompts1.json"

list_of_json_obj = dict()

with open("/home/long/ControlNet-for-BEP/training/cityscapes_instance_prompts/train_prompts1.json","r") as f:
    for jsonObj in f:
        json_to_dict = json.load(jsonObj)
        list_of_json_obj.append(json_to_dict)
        f.close()
        print(f)

