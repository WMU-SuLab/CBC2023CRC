# -*- coding: utf-8 -*-
import torchvision.models
from torch import nn
import torch

import torch.nn.functional as F

from torchvision.models.convnext import convnext_tiny, ConvNeXt_Tiny_Weights



class MultiAV(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiAV, self).__init__()

        base_model = convnext_tiny
        cut = 6
        layers = list(base_model(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features)[:cut]

        base_layers = nn.Sequential(*layers)

        self.high_level_classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(384, 384*2, kernel_size=(3, 3), stride=(1, 1),padding=1, bias=False),
            nn.BatchNorm2d(768),
            #nn.ReLU(inplace=True),
        )
        self.adapt_global = nn.AdaptiveAvgPool2d(1)
        self.adapt_row = nn.AdaptiveAvgPool2d((None,1))
        self.adapt_col= nn.AdaptiveAvgPool2d((1,None))
        self.mid =nn.Sequential( nn.Conv2d(384*4, 384*2, kernel_size=(3, 3), stride=(1, 1),padding=1, bias=False),nn.BatchNorm2d(384*2),
    nn.ReLU(inplace=True))

        self.sn_unet = base_layers


        self.num_classes = num_classes
        self.final = nn.Sequential(nn.Conv2d(384*2, self.num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),nn.Flatten(1))
    def forward(self, x):

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

        high_out = x.clone()
        high_out = self.high_level_classifier(high_out)

        high_out_global = self.adapt_global(high_out)
        high_out_col = self.adapt_col(high_out)
        high_out_row = self.adapt_row(high_out)
        high_out_global_2 = torch.matmul(high_out_col,high_out_row)
        high_out = torch.cat([high_out_global,high_out_global_2],dim=1)
        high_out = self.mid(high_out)
        high_out = self.final(high_out)



        return high_out
        #return x

if __name__ == '__main__':
    from loss import multiLabelLoss


    pt = torch.load(r'E:\CBC挑战赛2023内窥镜\class_1\weight\train_best_acc_model.pkl')


    m = MultiAV(num_classes=2)
    m_static = m.state_dict()
    pt_ ={k:v for k,v in pt.items() if k in m_static}

    m_static.update(pt_)

    m.load_state_dict(m_static)


    x = torch.randn((2,3,256,256))



    # y = torch.tensor([[0,1,2],[1,1,2]]).float()
    out = m(x)

    print(m(x).shape)
