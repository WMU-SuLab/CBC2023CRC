# -*- coding: utf-8 -*-
"""
@author: wenting
"""
import os
from opt_UKBB import modelEvalution
import torch.autograd as autograd
import torch
import cv2
import numpy as np
from tqdm import tqdm
import os
from models.network import GNet
from lib.Utils import *
def test(cfg):
    device = torch.device("cuda" if cfg.use_cuda else "cpu")
    model_root = cfg.model_path_pretrained_G
    model_path = os.path.join(model_root, 'G_' + str(cfg.model_step_pretrained_G) + '.pkl')
    net = torch.load(model_path,map_location=device)
    result_folder = os.path.join(model_root, 'running_result')
    modelEvalution(cfg.model_step_pretrained_G, net,
                       result_folder,
                       use_cuda=cfg.use_cuda,
                       dataset=cfg.dataset_name,
                       input_ch=cfg.input_nc,
                       config=cfg,
                       strict_mode=True)




if __name__ == '__main__':
    from config import config_test_general as cfg
    test(cfg);
