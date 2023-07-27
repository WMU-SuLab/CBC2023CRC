# -*- coding: utf-8 -*-
"""
@author: DW
Code for generate augmentation for batch image.
"""
import math

import numpy as np
import cv2
import os


def patchJudge(label,y,x, patch_h, patch_w,type=1,limit=100):
    '''

    :param label: label

    :param y: y coordinate of the patch
    :param x: x coordinate of the patch
    :param patch_h: height of the patch
    :param patch_w: width of the patch
    :param type: 'only_a', 'only_v', 'only_av', 'a_or_v'
    :param limit: the limit of the number of pixels in the patch
    0 represents background
    1 represents foreground

    '''
    # print(type)
    # print(patch_w)
    #print(limit)
    # print(f'av:{np.sum(av[y:y + patch_h, x:x + patch_w])}')
    # print(f'a:{np.sum(a[y:y + patch_h, x:x+patch_w])}')
    # print(f'v:{np.sum(v[y:y + patch_h, x:x + patch_w])}')

    if type==0:
        return np.sum(label[y:y+patch_h,x:x+patch_w])==0

    if type==1:
        return np.sum(label[y:y + patch_h, x:x + patch_w]) >= limit


def patch_select(ind,patch_size ,Label, type=0,limit=100):
    '''
    :param patch_size:
    :param Label: HW
    :param type: 'b' or 'f'
    0 represents background
    1 represents foreground

    '''
    H,W = Label.shape
    patch_size = patch_size
    y = np.random.randint(0, Label.shape[0] - patch_size + 1)
    x = np.random.randint(0, Label.shape[1] - patch_size + 1)
    cn = 0

    while (not patchJudge(Label, y, x, patch_size, patch_size, type=type,limit=limit)) and cn!=1000:
        y = np.random.randint(0, Label.shape[0] - patch_size + 1)
        x = np.random.randint(0, Label.shape[1] - patch_size + 1)
        cn = cn+1
    #print("x,y:",x,y)
    if cn==1000:
        limit=-1
    return y, x,limit

print("---------------------------------------")


if __name__ == '__main__':
    import cv2
    from PIL import Image
    from tqdm import tqdm
    np.random.seed(4)
    #a = cv2.imread(r'E:\eye_paper\AUV-GAN\data\ukbb\test\av\test_02.png')
    dataset_mean_a = 0.0
    dataset_mean_v = 0.0
    dataset_mean_av = 0.0
    dataset_mean_a_list = []
    dataset_mean_v_list = []
    dataset_mean_av_list = []
    bf_type = 1
    #dataset = cfg.dataset

    dataset_image_dict = {'CVC-ClinicDB_CVC-612':'Original_png','CVC-ColonDB':'original', 'ETIS-LaribPolypDB':'images','Kvasir-SEG':'images','sessile-main-Kvasir-SEG':'images'}

    dataset_label_dict = {'CVC-ClinicDB_CVC-612':'Ground Truth','CVC-ColonDB':'label','ETIS-LaribPolypDB':'masks','Kvasir-SEG':'masks','sessile-main-Kvasir-SEG':'masks'}

    # CVC-ClinicDB_CVC-612 CVC-ColonDB ETIS-LaribPolypDB Kvasir-SEG sessile-main-Kvasir-SEG

    dataset_name_list = ['CVC-ClinicDB_CVC-612', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir-SEG', 'sessile-main-Kvasir-SEG']
    dataset_name_index = 4
    dataset_name = dataset_name_list[dataset_name_index]

    ground_dir_name = dataset_label_dict[dataset_name]
    image_dir_name = dataset_image_dict[dataset_name]
    if not os.path.exists(os.path.join(dataset_name,'training','label00')):
        os.makedirs(os.path.join(dataset_name,'training','label00'))
        if not os.path.exists(os.path.join(dataset_name, 'training', 'image00')):
            os.makedirs(os.path.join(dataset_name, 'training', 'image00'))

    if not os.path.exists(os.path.join(dataset_name,'training','label11')):
        os.makedirs(os.path.join(dataset_name,'training','label11'))
        if not os.path.exists(os.path.join(dataset_name, 'training', 'image11')):
            os.makedirs(os.path.join(dataset_name, 'training', 'image11'))
    #dataset_name = 'hrf'

    for ind,i in enumerate(tqdm(sorted(os.listdir(os.path.join(dataset_name,ground_dir_name))))):
        prefix_i = i.split('.')[0]
        # if ind!=80:
        #     continue

        # 判断文件夹是否存在包含前缀加 tif jpg png其中一个的文件
        if os.path.exists(os.path.join(dataset_name,image_dir_name,prefix_i+'.tif')) :
            imagepath = os.path.join(dataset_name,image_dir_name,prefix_i+'.tif')
        elif os.path.exists(os.path.join(dataset_name,image_dir_name,prefix_i+'.jpg')):
            imagepath = os.path.join(dataset_name,image_dir_name,prefix_i+'.jpg')
        elif os.path.exists(os.path.join(dataset_name,image_dir_name,prefix_i+'.png')):
            imagepath = os.path.join(dataset_name,image_dir_name,prefix_i+'.png')
        elif os.path.exists(os.path.join(dataset_name,image_dir_name,prefix_i+'.tiff')):
            imagepath = os.path.join(dataset_name,image_dir_name,prefix_i+'.tiff')

        if dataset_name=='CVC-ColonDB':
            if os.path.exists(os.path.join(dataset_name, image_dir_name, prefix_i.split('p')[-1] + '.tif')):
                imagepath = os.path.join(dataset_name, image_dir_name, prefix_i.split('p')[-1] + '.tif')
            else:
                imagepath = os.path.join(dataset_name,image_dir_name,prefix_i.split('p')[-1]+'.tiff')
        path = os.path.join(dataset_name,ground_dir_name,i)
        #imagepath = os.path.join(r'E:\eye_paper\AUV-GAN\data',dataset_name,'training','images',i)

        #imagepath_ori = os.path.join(r'E:\eye_paper\AUV-GAN\data', dataset_name, 'training', 'images_ori', i)
        label_ = cv2.imread(path,0)


        a_image = cv2.imread(imagepath)
        # print(type(a_image))
        if isinstance(a_image,type(None)):
            print(f'======================={prefix_i}========================')
            continue
        #Image.fromarray(a_image).show()

        a_image_bak = a_image.copy()
        a_image = cv2.cvtColor(a_image, cv2.COLOR_BGR2RGB)

        Label = np.zeros((a_image.shape[0], a_image.shape[1]), dtype=np.uint8)

        Label[label_[:,:]> 128] = 1

        f_label_s = np.count_nonzero(Label)
        f_label_rate = f_label_s/(label_.shape[0]*label_.shape[1])

        b_lable_s = label_.shape[0]*label_.shape[1]-f_label_s
        b_label_rate = 1-f_label_rate
        thresh_rate = 0.3
        print(f_label_rate)
        if bf_type==1:
            if f_label_rate>=thresh_rate:

                #Image.fromarray(Label*255).show()
                Image.fromarray(Label*255).save(
                    fr'{dataset_name}/training/label11/{dataset_name}_f_{ind}.png')

                Image.fromarray(a_image).save(
                    fr'{dataset_name}/training/image11/{dataset_name}_f_{ind}.png')
            else:

                patch_size = max(64, int(np.ceil(math.pow(f_label_s/thresh_rate,0.5))))
                patch_size = min(Label.shape[0],patch_size,Label.shape[1])

                y, x, limit = patch_select(ind, patch_size, Label, type=bf_type,limit=f_label_s*0.7)

                if limit==-1:
                    continue

                patch_select_image = a_image[y:y + patch_size, x:x + patch_size, :]
                patch_select_label = Label[y:y + patch_size, x:x + patch_size]

                #Image.fromarray(patch_select_label*255).show()
                Image.fromarray(patch_select_label*255).save(
                    fr'{dataset_name}/training/label11/{dataset_name}_f_{ind}.png')

                Image.fromarray(patch_select_image).save(
                    fr'{dataset_name}/training/image11/{dataset_name}_f_{ind}.png')

        else:

            if f_label_rate<0.01:
                patch_size = 128
            elif f_label_rate>thresh_rate:
                patch_size=64
            else:
                patch_size=96

            y, x, limit = patch_select(ind, patch_size, Label, type=bf_type, limit=0)

            patch_select_image = a_image[y:y + patch_size, x:x + patch_size, :]
            patch_select_label = Label[y:y + patch_size, x:x + patch_size]

            #Image.fromarray(patch_select_label * 255).show()
            #Image.fromarray(patch_select_image).show()
            Image.fromarray(patch_select_label).save(
                fr'{dataset_name}/training/label00/{dataset_name}_b_{ind}.png')

            Image.fromarray(patch_select_image).save(
                fr'{dataset_name}/training/image00/{dataset_name}_b_{ind}.png')



