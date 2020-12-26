# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 21:36:41 2020

@author: hanyue
"""

from collections import namedtuple

import torch
from torchvision import models


class Resnet50(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Resnet50, self).__init__()
        self.slice1 = torch.nn.Sequential()
        self.slice1.add_module('conv1',models.resnet50(pretrained=True).conv1)
        self.slice1.add_module('bn1',models.resnet50(pretrained=True).bn1)
        self.slice1.add_module('relu',models.resnet50(pretrained=True).relu)
        self.slice2 = models.resnet50(pretrained=True).layer1
        self.slice3 = models.resnet50(pretrained=True).layer2
        self.slice4 = models.resnet50(pretrained=True).layer3

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_3 = h
        h = self.slice2(h)
        h_relu2_4 = h
        h = self.slice3(h)
        h_relu3_6 = h
        h = self.slice4(h)
        h_relu4_3 = h
        resnet_outputs = namedtuple("ResnetOutputs", ['relu1_3', 'relu2_4', 'relu3_6', 'relu4_3'])
        out = resnet_outputs(h_relu1_3, h_relu2_4, h_relu3_6, h_relu4_3)
        return out

