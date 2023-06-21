import os
import json
import cv2

# with open("training/cityscapes_instance_prompts/val_prompts.json", 'rt') as f:
#     for line in f:
#             python_dict = json.loads(line)
#             path_to_label_file = python_dict['source']
#             print(path_to_label_file)
#             label = cv2.imread(path_to_label_file)
#             print(label)
#             label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)


source = "/home/long/mmsegmentation/data/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png"

labelmap_content = cv2.imread(source)

print(labelmap_content)