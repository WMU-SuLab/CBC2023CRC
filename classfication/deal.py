import cv2
import pandas as pd
ukb = pd.read_excel('D:/dataset-cv/UKB/3.xlsx')
for i in range(len(ukb)):
    path = 'D:/dataset-cv/UKB/UKB/fundus_image_with_refraction/' + ukb['name'][i]
    image_name = ukb['name'][i].split('.')[0]
    ori = cv2.imread(path)
    # aug = manual_label_make(ori)e
    # print(aug.shape)
    path_ori = f'dataset/train/3/{image_name}.png'
    path_aug = f'D:/dataset-cv/UKB/aug/{image_name}_aug.png'
    cv2.imwrite(path_ori, ori)
