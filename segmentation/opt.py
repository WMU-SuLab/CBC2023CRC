# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.autograd as autograd
import os
import cv2

from Tools.data_augmentation import data_aug5, data_aug7, data_aug9

from lib.Utils import *
from models.network import GNet

from DRIVE_Evalution import Evalution_drive


def get_patch_trad_5(batch_size, patch_size, train_data, train_label_data):
    data = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)

    for j in range(batch_size):

        z = np.random.randint(0, train_data.shape[0])
        choice = np.random.randint(0, 6)

        data_mat = train_data[z,:,:,:]
        label_mat = train_label_data[z,:,:,:]

        data_mat, label_mat= data_aug5(data_mat, label_mat, choice)
        data[j, :, :, :] = data_mat
        label[j, :, :, :] = label_mat


    return data, label[:,0:1,:,:]


import numpy as np
import albumentations as albu
from PIL import Image
def CAM(x):
    """
    :param dataset_path: 计算整个训练数据集的平均RGB通道值
    :param image:  array， 单张图片的array 形式
    :return: array形式的cam后的结果
    """
    # 每次使用新数据集时都需要重新计算前面的RBG平均值
    # RGB-->Rshift-->CLAHE
    resize=False
    R_mea_num, G_mea_num, B_mea_num = [], [], []
    dataset_path = r'E:\CBC挑战赛2023内窥镜\CVC-ClinicDB_CVC-612\Original_png\1.png'
    # for file in os.listdir(dataset_path):
    #
    #     image = np.array(Image.open(os.path.join(dataset_path, file)))
    #     # mea.append((file,np.mean(x[:,:,0])))
    #     R_mea_num.append(np.mean(image[:, :, 0]))
    #     G_mea_num.append(np.mean(image[:, :, 1]))
    #     B_mea_num.append(np.mean(image[:, :, 2]))
    # print(np.mean(R_mea_num), np.mean(G_mea_num), np.mean(B_mea_num))
    image = np.array(Image.open(dataset_path))
    R_mea_num.append(np.mean(image[:, :, 0]))
    G_mea_num.append(np.mean(image[:, :, 1]))
    B_mea_num.append(np.mean(image[:, :, 2]))
    mea2stand = int(np.mean(R_mea_num)-np.mean(x[:,:,0]))
    mea2standg = int(np.mean(G_mea_num)-np.mean(x[:,:,1]))
    mea2standb = int(np.mean(B_mea_num)-np.mean(x[:,:,2]))

    y = albu.RGBShift(r_shift_limit=(mea2stand,mea2stand),
        g_shift_limit=(mea2standg,mea2standg),
        b_shift_limit=(mea2standb,mea2standb),p=1,always_apply=True)(image=x)
    # y = ablu.FDA(reference_images=[reference_images],beta_limit=(1,1),p=1,read_fn=lambda x:x)(image=x)
    #y = albu.CLAHE(clip_limit=(2,2),p=1,always_apply=True)(image=np.uint8(y['image']))
    return y['image']

def Dataloader_general(path, use_CAM=False,use_resize=False,reszie_w_h=(256,256)):


    ImgPath = path + "images/"
    LabelPath = path + "label/"


    if use_resize:
        resize_w_h = reszie_w_h

    ImgDir = os.listdir(ImgPath)
    LabelDir = os.listdir(LabelPath)

    Img0 = cv2.imread(ImgPath + ImgDir[0])
    Label0 = cv2.imread(LabelPath + LabelDir[0])
    image_suffix = os.path.splitext(ImgDir[0])[1]
    label_suffix = os.path.splitext(LabelDir[0])[1]
    if use_resize:
        Img0 = cv2.resize(Img0, resize_w_h)
        Label0 = cv2.resize(Label0, resize_w_h)

    Img = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Label = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)


    for i,name in enumerate(ImgDir):

        prefix = os.path.splitext(name)[0]

        Label_ = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)

        Img0 = cv2.imread(os.path.join(ImgPath, f'{prefix}{image_suffix}'))
        Label0 = cv2.imread(os.path.join(LabelPath,f'{prefix}{label_suffix}'),0)

        if use_resize:
            Img0 = cv2.resize(Img0, resize_w_h)
            Label0 = cv2.resize(Label0, resize_w_h, interpolation=cv2.INTER_NEAREST)




        Label_[(Label0> 128) ] = 1


        ImgCropped = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)

        if use_CAM:
            ImgCropped = CAM(ImgCropped)

        ImgCropped = np.float32(ImgCropped / 255.)


        Img[i, :, :, :] = np.transpose(ImgCropped, (2, 0, 1)) # HWC to CHW

        Label[i, 0, :, :] = Label_

    Img = Normalize(Img)

    return Img, Label


def modelEvalution(i,net,savePath, use_cuda=False, dataset='DRIVE', is_kill_border=True, input_ch=3, strict_mode=True,config=None):

    # path for images to save
    dataset_dict = {'CBC':'CBC','CBC_ft':'CBC_ft'}
    dataset_name = dataset_dict[dataset]

    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/images'))
    label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/label'))
    assert len(image_basename) == len(label_basename)

    image0 = cv2.imread(f'./data/{dataset_name}/test/images/{image_basename[0]}')

    data_path = os.path.join(savePath, dataset)
    metrics_file_path = os.path.join(savePath, 'metrics.txt')#_'+str(config.model_step_pretrained_G)+'.txt')
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    test_image_num = len(image_basename)
    test_image_height = image0.shape[0]
    test_image_width = image0.shape[1]
    if config.use_resize:
        test_image_height = config.reszie_w_h[1]
        test_image_width = config.reszie_w_h[0]

    PredAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    LabelAll = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    ProMap = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    LabelMap = np.zeros((test_image_num, 1, test_image_height, test_image_width), np.float32)
    MaskAll = np.ones((test_image_num, 1, test_image_height, test_image_width), np.float32)

    #Vessel = VesselProMap('./data/AV_DRIVE/test/images')

    n_classes = 1
    Net = GNet(resnet=config.use_network, input_ch=input_ch, num_classes= n_classes, use_cuda=use_cuda, pretrained=False)
    Net.load_state_dict(net)

    if use_cuda:
        Net.cuda()
    Net.eval()

    for k in tqdm(range(test_image_num)):
        Pred,Label_ = GetResult(Net,k,
                                                                                          use_cuda=use_cuda,
                                                                                          dataset_name=dataset_name,
                                                                                          is_kill_border=is_kill_border,
                                                                                          input_ch=input_ch,
                                                                                          config=config)

        PredAll[k,:,:,:] = Pred

        LabelAll[k,:,:,:] = Label_


    ProMap[:,0,:,:] =PredAll[:,0,:,:]

    LabelMap[:,0,:,:] = LabelAll[:,0,:,:]


    AUC,Acc,Sp,Se,F1,Dice,Iou = Evalution_drive(PredAll, LabelAll,MaskAll)

    #filewriter = centerline_eval(ProMap, config)
    np.save(os.path.join(savePath, "ProMap_testset.npy"),ProMap)

    for k in range(0,test_image_num):


        cv2.imwrite(os.path.join(data_path, f"{dataset}_{image_basename[k].split('.')[-1]}"+str(k).zfill(3)+".png"),PredAll[k,0,:,:]*255)

        # draw the thresh > 0.5转int
        cv2.imwrite(os.path.join(data_path, f"{dataset}_{image_basename[k].split('.')[-1]}"+str(k).zfill(3)+"_thresh.png"),(PredAll[k,0,:,:]>0.5)*255)



    print(f"========================={dataset}=============================")
    print("Strict mode:{}".format(strict_mode))

    print(f"The {i} step  AUC is:{AUC}")
    print(f"The {i} step  Acc is:{Acc}")
    print(f"The {i} step  Sens is:{Se}")
    print(f"The {i} step  Spec is:{Sp}")
    print(f"The {i} step F1 is:{F1}")
    print(f"The {i} step  Dice is:{Dice}")
    print(f"The {i} step  Iou is:{Iou}")
    print("-----------------------------------------------------------")

    if not os.path.exists(metrics_file_path):
         file_w = open(metrics_file_path,'w')
    file_w = open(metrics_file_path,'r+')
    file_w.read()
    file_w.write(f"========================={dataset}=============================" + '\n' +
                 "Strict mode:{}".format(strict_mode) + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 f"The {i} step AUC is:{AUC}" + '\n' +
                 f"The {i} step Acc is:{Acc}" + '\n' +
                 f"The {i} step  Sens is:{Se}" + '\n' +
                 f"The {i} step  Spec is:{Sp}" + '\n' +
                 f"The {i} step F1 is:{F1}" + '\n' +
                 f"The {i} step Dice is:{Dice}" + '\n' +
                 f"The {i} step Iou is:{Iou}" + '\n' +
                 "-----------------------------------------------------------" + '\n')
    #file_w.write(filewriter)
    file_w.close()



def GetResult(Net, k, use_cuda=False, dataset_name='DRIVE', is_kill_border=True, input_ch=3, config=None):

    image_basename = sorted(os.listdir(f'./data/{dataset_name}/test/images'))[k]
    label_basename = sorted(os.listdir(f'./data/{dataset_name}/test/label'))[k]
    assert image_basename.split('.')[0] == label_basename.split('.')[0]  # check if the image and label are matched


    ImgName = os.path.join(f'./data/{dataset_name}/test/images/',image_basename)
    LabelName = os.path.join(f'./data/{dataset_name}/test/label/',label_basename)

    Img0 = cv2.imread(ImgName)
    Label0 = cv2.imread(LabelName,0)


    if config.use_resize:
        Img0 = cv2.resize(Img0, config.reszie_w_h)
        Label0 = cv2.resize(Label0, config.reszie_w_h, interpolation=cv2.INTER_NEAREST)



    Label = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)

    Label[Label0>=128] = 1

    Img = Img0
    height, width = Img.shape[:2]
    n_classes = 1
    patch_height = config.patch_size
    patch_width = config.patch_size
    stride_height = config.stride_height
    stride_width = config.stride_width
    
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    # rgb2rgg
    if config.use_CAM:
        Img = CAM(Img)
    Img = np.float32(Img/255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)


    patch_size = config.patch_size
    batch_size = 4
    
    patches_imgs = extract_ordered_overlap(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs,(0,3,1,2))
    patches_imgs = Normalize(patches_imgs)

    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum/float(batch_size)))
    
    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    for i in range(max_iter):
        begin_index = i*batch_size
        end_index = (i+1)*batch_size
    
        patches_temp1 = patches_imgs[begin_index:end_index, :, :, :]

        patches_input_temp1 = torch.FloatTensor(patches_temp1)
        if use_cuda:
            patches_input_temp1 = autograd.Variable(patches_input_temp1.cuda())
        else:
            patches_input_temp1 = autograd.Variable(patches_input_temp1)
               
        output_temp = Net(patches_input_temp1)

        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())

        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)
    
        pred_patches[begin_index:end_index, :,:,:] = pred_patches_temp_sigmoid
    
        del patches_input_temp1
        del pred_patches_temp
        del patches_temp1
        del output_temp
        del pred_patches_temp_sigmoid
    
    
    new_height, new_width = Img_enlarged.shape[0], Img_enlarged.shape[1]
    pred_img = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    pred_img = pred_img[:,0:height,0:width]

    
    Pred = np.float32(pred_img[0,:,:])

    
    Pred = Pred[np.newaxis,:,:]

    Label = Label[np.newaxis,:,:]


    
    
    return Pred,Label





def draw_prediction(writer, pred, targs, step):
    target_ = targs[0:1,0,:,:]

    pred_sigmoid = pred #nn.Sigmoid()(pred)
    
    writer.add_image('bf',  torch.cat([pred_sigmoid[0:1,0,:,:], target_], dim=1), global_step=step)

    

if __name__ == '__main__':
    import pathlib

    i = sorted(os.listdir(f'./data/AV_DRIVE/test/images'), reverse=False)[0]



    print(i)
