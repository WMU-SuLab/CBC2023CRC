# -*- coding: utf-8 -*-
"""
@author: wenting
"""
import os
import torch.autograd as autograd
import torch
import cv2
import numpy as np
from tqdm import tqdm
import os
from models.network import GNet
from lib.Utils import *


def modelEvalution(i, net, savePath, use_cuda=False, dataset='out', input_ch=3, config=None):
    # path for images to save
    dataset_dict = {'out': 'out'}
    dataset_name = dataset_dict[dataset]
    print(f'evaluating {dataset_name} dataset...')
    image_basename = sorted(os.listdir(f'./data/{dataset_name}/images'))

    data_path = os.path.join(savePath)

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    print(f'num of test images: {len(image_basename)}')
    n_classes = 1
    Net = GNet(resnet=config.use_network, input_ch=input_ch, num_classes=n_classes, use_cuda=use_cuda, pretrained=False,
               )

    Net.load_state_dict(net)
    if use_cuda:
        Net.cuda()
    Net.eval()

    test_image_num = len(image_basename)
    t = tqdm(range(test_image_num))
    for k in t:
        t.set_description(f'========== Processing {k}/{test_image_num} Image: {image_basename[k]} ================')
        image0 = cv2.imread(f'./data/{dataset_name}/images/{image_basename[k]}')
        test_image_height = image0.shape[0]
        test_image_width = image0.shape[1]
        test_image_height_ = test_image_height
        test_image_width_ = test_image_width
        if config.use_resize and max(test_image_height, test_image_width) > 0:
            test_image_height_ = config.reszie_w_h[1]
            test_image_width_ = config.reszie_w_h[0]
            image0 = cv2.resize(image0, config.reszie_w_h)

        PredAll = np.zeros((1, 1, test_image_height_, test_image_width_), np.float32)

        Pred = GetResult(Net, image0,
                         use_cuda=use_cuda,
                         input_ch=input_ch,
                         config=config)
        PredAll[0, :, :, :] = Pred

        # cv2.imwrite(os.path.join(data_path, f"{image_basename[k].split('.')[0]}_map.{image_basename[k].split('.')[-1]}"), PredAll[0, 0, :, :] * 255)

        image_save = cv2.resize(PredAll[0, 0, :, :], (test_image_width, test_image_height))

        # draw the thresh > 0.5转int
        cv2.imwrite(os.path.join(data_path, image_basename[k]),
                    (image_save > 0.65) * 255)

        # file_w.write(filewriter)


def GetResult(Net, Img0, use_cuda=False, input_ch=3, config=None):
    Img = Img0
    height, width = Img.shape[:2]
    n_classes = 1
    patch_height = config.patch_size
    patch_width = config.patch_size
    stride_height = config.stride_height
    stride_width = config.stride_width
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    # rgb2rgg

    Img = np.float32(Img / 255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)

    patch_size = config.patch_size
    batch_size = 8

    patches_imgs = extract_ordered_overlap(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs, (0, 3, 1, 2))
    patches_imgs = Normalize(patches_imgs)

    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum / float(batch_size)))

    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    for i in range(max_iter):
        begin_index = i * batch_size
        end_index = (i + 1) * batch_size

        patches_temp1 = patches_imgs[begin_index:end_index, :, :, :]

        patches_input_temp1 = torch.FloatTensor(patches_temp1)
        if use_cuda:
            patches_input_temp1 = autograd.Variable(patches_input_temp1.cuda())
        else:
            patches_input_temp1 = autograd.Variable(patches_input_temp1)

        output_temp = Net(patches_input_temp1)

        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())

        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)

        pred_patches[begin_index:end_index, :, :, :] = pred_patches_temp_sigmoid

        del patches_input_temp1
        del pred_patches_temp
        del patches_temp1
        del output_temp
        del pred_patches_temp_sigmoid

    new_height, new_width = Img_enlarged.shape[0], Img_enlarged.shape[1]
    pred_img = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    pred_img = pred_img[:, 0:height, 0:width]

    Pred = np.float32(pred_img[0, :, :])

    Pred = Pred[np.newaxis, :, :]

    return Pred


def out_test(cfg):
    device = torch.device("cuda" if cfg.use_cuda else "cpu")
    model_root = cfg.model_path_pretrained_G
    model_path = os.path.join(model_root, 'G_' + str(cfg.model_step_pretrained_G) + '.pkl')

    net = torch.load(model_path, map_location=device)

    modelEvalution(cfg.model_step_pretrained_G, net,
                   result_folder,
                   use_cuda=cfg.use_cuda,
                   dataset='out',
                   input_ch=cfg.input_nc,
                   config=cfg,
                   )


if __name__ == '__main__':
    from config import config_test_general as cfg
    import argparse
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--result_folder', type=str, required=True, default='./out')
    args = parser.parse_args()
    result_folder = args.result_folder
    out_test(cfg)
