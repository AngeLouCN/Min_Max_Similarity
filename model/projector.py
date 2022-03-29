# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:21:36 2022

@author: loua2
"""

import functools

import torch
import torch.nn as nn

class CONV_Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        # self.relu = nn.ReLU(inplace = True)
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace = True)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        y_1 = self.conv(x)
        y_1 = self.bn(y_1)
        y_1 = self.relu(y_1)
        

        return y_1


class projectors(nn.Module):
    def __init__(self, input_nc=1, ndf=8, norm_layer=nn.BatchNorm2d):
        super(projectors, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(input_nc, ndf)
        self.conv_2 = conv(ndf, ndf*2)
        self.final = nn.Conv2d(ndf*2, ndf*2, kernel_size=1)
    def forward(self, input):
        x_0 = self.conv_1(input)
        x_0 = self.pool(x_0)
        x_out = self.conv_2(x_0)
        x_out = self.pool(x_out)
        x_out = self.final(x_out)
        return x_out    
    
class classifier(nn.Module):
    def __init__(self, inp_dim = 1,ndf=8, norm_layer=nn.BatchNorm2d):
        super(classifier, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(inp_dim, ndf)
        self.conv_2 = conv(ndf, ndf*2)
        self.conv_3 = conv(ndf*2, ndf*4)
        self.final = nn.Conv2d(ndf*4, ndf*4, kernel_size=1)
        # self.linear = nn.Linear(in_features=ndf*4*18*12, out_features=1024)
    def forward(self,input):
        x_0 = self.conv_1(input)
        x_0 = self.pool(x_0)
        x_1 = self.conv_2(x_0)
        x_1 = self.pool(x_1)
        x_2 = self.conv_3(x_1)
        x_2 = self.pool(x_2)
        # x_out = self.linear(x_2)
        x_out = self.final(x_2)
        return x_out
   
      
            