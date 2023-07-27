# -*- coding: utf-8 -*-
"""
@author: wentingchen
Code for generate shuffled image for datasets.
Dataset:
AV-DRIVE - only shuffle the trainset
HRF - shuffle all the ground-truth
"""
import numpy as np
import cv2
import os
from genFakeSample import generate, filterGreenWhite
from tqdm import tqdm


dataset = 'all_combine'

shuffle_rate_range=[0.1,0.35]
dataset_root_list = [f'../data/{dataset}/training/av',f'../data/{dataset}/test/av']
save_root = '../data/shuffled_dataset'
save_root_list = [os.path.join(save_root, f'{dataset}', 'training'), os.path.join(save_root, f'{dataset}', 'test')]


print(f"Generating shuffled data for {dataset} dataset with shuffle rate rangine from " + str(shuffle_rate_range[0]) + " to " + str(shuffle_rate_range[1]))
print("The shuffled data is saved to :" + save_root)

# check
assert len(save_root_list) == len(dataset_root_list) , "The length of saveRoot is not the same as the srcRoot!"

# make directory
if not os.path.exists(save_root):
    os.mkdir(save_root)
    print("Making directory: ", save_root)
for path in save_root_list:
    if not os.path.exists(path):
        os.makedirs(path)
        print("Making directory: ", path)


for j in range(len(save_root_list)):
    input_av_root = dataset_root_list[j]
    output_av_root = save_root_list[j]
    print("Input data root :", input_av_root)
    print("Output data root :", output_av_root)
    
    imglist = os.listdir(input_av_root)
    num = len(imglist)

    for i in tqdm(range(num)):
        shuffle_ratio = np.random.uniform(shuffle_rate_range[0], shuffle_rate_range[1], 1)
        # read image
        AVSeg = cv2.imread(os.path.join(input_av_root, imglist[i]))
        AVSeg = cv2.cvtColor(AVSeg, cv2.COLOR_BGR2RGB)

        AVSeg = filterGreenWhite(AVSeg)
        fakeAVSeg = generate(AVSeg, shuffle_ratio=shuffle_ratio)
        fakeAVSeg = cv2.cvtColor(fakeAVSeg, cv2.COLOR_RGB2BGR)
        fakeAVSeg = np.uint8(fakeAVSeg)
        # save fake AVSeg image
        cv2.imwrite(os.path.join(output_av_root, imglist[i]), fakeAVSeg)

print("---------------------------------------")
    
