import os
import random

random.seed(777)

dir1 = 'results/inference_on_val_test_run_3_lr_4e-5/GTA_2/HR'
dir2 = 'results/inference_on_val_test_run_3_lr_4e-5/GTA1/HR'

dir1_imgs = os.listdir(dir1)
dir2_imgs = os.listdir(dir2)

dir1_imgs = [f"{dir1}/{x}" for x in dir1_imgs]
dir2_imgs = [f"{dir2}/{x}" for x in dir2_imgs]

imgs = dir1_imgs + dir2_imgs

imgs = random.shuffle(imgs)
random_sample = random.sample(imgs, k=500)

with open('random_sample_files.txt', 'w+') as f:
    for img_path in random_sample:
        f.writelines(img_path + "\n")