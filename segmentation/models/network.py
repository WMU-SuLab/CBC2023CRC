# -*- coding: utf-8 -*-
import torchvision.models
from torch import nn
import torch

import torch.nn.functional as F
from models.layers import *
from torchvision.models.convnext import convnext_tiny, ConvNeXt_Tiny_Weights
import numpy as np
import math
from torchvision import models


class GNet(nn.Module):
    def __init__(self, input_ch=3, resnet='swin_t', num_classes=1, use_cuda=False, pretrained=True):
        super(GNet, self).__init__()
        self.resnet = resnet

        base_model = convnext_tiny

        # layers = list(base_model(pretrained=pretrained,num_classes=num_classes,input_ch=input_ch).children())[:cut]

        if resnet == 'swin_t':
            cut = 6
            if pretrained:
                layers = list(base_model(weights=Swin_T_Weights.IMAGENET1K_V1).features)[:cut]
            else:
                layers = list(base_model().features)[:cut]

            base_layers = nn.Sequential(*layers)
            # self.stage = [SaveFeatures(base_layers[0][2])]  # stage 1  c=96
            self.stage = []
            self.stage.append(SaveFeatures(base_layers[0][2]))  # stem c=96
            self.stage.append(SaveFeatures(base_layers[1][1]))  # stage 1 c=96
            self.stage.append(SaveFeatures(base_layers[3][1]))  # stage 2 c=192
            self.stage.append(SaveFeatures(base_layers[5][5]))  # stage 3 c=384
            # self.stage.append(SaveFeatures(base_layers[7][1]))  # stage 5 c=768

            self.up2 = DBlock(384, 192)
            self.up3 = DBlock(192, 96)
            self.up4 = DBlock(96, 96)

            # final convolutional layers
            # predict artery, vein and vessel

            self.seg_head = SegmentationHead(96, 3, 3, upsample=4)

        elif resnet == 'convnext_tiny':

            cut = 6
            if pretrained:
                layers = list(base_model(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features)[:cut]
            else:
                layers = list(base_model().features)[:cut]

            base_layers = nn.Sequential(*layers)


            # self.stage = [SaveFeatures(base_layers[0][1])]  # stage 1  c=96
            self.stage = []
            self.stage.append(SaveFeatures(base_layers[0][1]))  # stem c=96
            self.stage.append(SaveFeatures(base_layers[1][2]))  # stage 1 c=96
            self.stage.append(SaveFeatures(base_layers[3][2]))  # stage 2 c=192
            self.stage.append(SaveFeatures(base_layers[5][8]))  # stage 3 c=384
            # self.stage.append(SaveFeatures(base_layers[7][2]))  # stage 5 c=768
            self.up2 = DBlock(384, 192)
            self.up3 = DBlock(192, 96)
            self.up4 = DBlock(96, 96)

            # final convolutional layers
            # predict artery, vein and vessel

            self.seg_head = SegmentationHead(96, 1, 3, upsample=4)

        else:
            cut = 7
            # base_layers = nn.Sequential(*layers)
            layers = list(base_model(pretrained=pretrained, input_ch=input_ch).children())[:cut]

            base_layers = nn.Sequential(*layers)
            self.stage = []
            self.stage.append(SaveFeatures(base_layers[2]))  # stage 1  c=64
            self.stage.append(SaveFeatures(base_layers[4][1]))  # stage 2  c=128
            self.stage.append(SaveFeatures(base_layers[5][1]))  # stage 3  c=256
            self.up2 = DBlock_res(256, 128)
            self.up3 = DBlock_res(128, 64)
            self.up4 = DBlock_res(64, 64)
            self.seg_head = nn.Conv2d(64, 3, kernel_size=1, padding=0)

        self.sn_unet = base_layers
        self.num_classes = num_classes

        self.bn_out = nn.BatchNorm2d(3)
        # use centerness block



    def forward(self, x):
        if self.resnet == 'resnet18':
            x = F.relu(self.sn_unet(x))
        else:
            x = self.sn_unet(x)
        if len(x.shape) == 4 and x.shape[2] != x.shape[3]:
            B, H, W, C = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()
        elif len(x.shape) == 3:
            B, L, C = x.shape
            h = int(L ** 0.5)
            x = x.view(B, h, h, C)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x
        if self.resnet == 'swin_t' or self.resnet == 'convnext_tiny':
            # feature = self.stage[1:]
            feature = self.stage[::-1]
            # head = feature[0]
            skip = feature[1:]

            # x = self.up1(x,skip[0].features)
            x = self.up2(x, skip[0].features)
            x = self.up3(x, skip[1].features)
            x = self.up4(x, skip[2].features)
        else:
            x = self.up2(x, self.stage[2].features)
            x = self.up3(x, self.stage[1].features)
            x = self.up4(x, self.stage[0].features)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.seg_head(x)
        ########################
        # baseline output
        # artery, vein and vessel

        #av cross

        #output = F.relu(self.bn_out(output))
        # use centerness block


        return out




def close(self):
        for sf in self.stage: sf.remove()


# set requies_grad=Fasle to avoid computation

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
def choose_vgg(name):
    f = None

    if name == 'vgg11':
        f = models.vgg11(pretrained=True)
    elif name == 'vgg11_bn':
        f = models.vgg11_bn(pretrained=True)
    elif name == 'vgg13':
        f = models.vgg13(pretrained=True)
    elif name == 'vgg13_bn':
        f = models.vgg13_bn(pretrained=True)
    elif name == 'vgg16':
        f = models.vgg16(pretrained=True)
    elif name == 'vgg16_bn':
        f = models.vgg16_bn(pretrained=True)
    elif name == 'vgg19':
        f = models.vgg19(pretrained=True)
    elif name == 'vgg19_bn':
        f = models.vgg19_bn(pretrained=True)

    for params in f.parameters():
        params.requires_grad = False

    return f


pretrained_mean = torch.tensor([0.485, 0.456, 0.406], requires_grad=False).view((1, 3, 1, 1))
pretrained_std = torch.tensor([0.229, 0.224, 0.225], requires_grad=False).view((1, 3, 1, 1))

class VGGNet(nn.Module):

    def __init__(self, name, layers, cuda=True):

        super(VGGNet, self).__init__()
        self.vgg = choose_vgg(name)
        self.layers = layers

        features = list(self.vgg.features)[:max(layers) + 1]
        self.features = nn.ModuleList(features).eval()

        self.mean = pretrained_mean.cuda() if cuda else pretrained_mean
        self.std = pretrained_std.cuda() if cuda else pretrained_std

    def forward(self, x, retn_feats=None, layers=None):

        x = (x - self.mean) / self.std

        results = []

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.layers:
                results.append(x.reshape(x.shape[0], -1))

        return results

if __name__ == '__main__':
    s = GNet(input_ch=3, resnet='convnext_tiny', pretrained=False)


    x = torch.randn(2, 3, 256, 256)
    y = s(x)



    print(y.shape)

    # import torchvision.models as models
    # m = models.vit_b_16(pretrained=False)
    # print(m)
    # m = resnet18()
    # m_list = list(m.children())
    # def hook(module, input, output):
    #     print('fafafafgafa')
    #     print(input[0].shape)
    #     print(output[0].shape)
    # m_list[0].register_forward_hook(hook)
    #
    #
    # y = m(x)
