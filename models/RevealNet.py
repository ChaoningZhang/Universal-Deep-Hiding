# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com

@version: 1.0
@file: Reveal.py
@time: 2018/3/20

"""

import torch.nn as nn


class RevealNet(nn.Module):
    def __init__(self, input_nc, output_nc, nhf=64, norm_layer=None, output_function=nn.Sigmoid):
        super(RevealNet, self).__init__()
        # input is (3) x 256 x 256

        self.conv1 = nn.Conv2d(input_nc, nhf, 3, 1, 1)        
        self.conv2 = nn.Conv2d(nhf, nhf * 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1)
        self.conv4 = nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1)
        self.conv5 = nn.Conv2d(nhf * 2, nhf, 3, 1, 1)
        self.conv6 = nn.Conv2d(nhf, output_nc, 3, 1, 1)
        self.output=output_function()
        self.relu = nn.ReLU(True)

        self.norm_layer = norm_layer
        if norm_layer != None:
            self.norm1 = norm_layer(nhf)
            self.norm2 = norm_layer(nhf*2)
            self.norm3 = norm_layer(nhf*4)
            self.norm4 = norm_layer(nhf*2)
            self.norm5 = norm_layer(nhf)

    def forward(self, input):

        if self.norm_layer != None:
            x=self.relu(self.norm1(self.conv1(input)))
            x=self.relu(self.norm2(self.conv2(x)))
            x=self.relu(self.norm3(self.conv3(x)))
            x=self.relu(self.norm4(self.conv4(x)))
            x=self.relu(self.norm5(self.conv5(x)))
            x=self.output(self.conv6(x))
        else:
            x=self.relu(self.conv1(input))
            x=self.relu(self.conv2(x))
            x=self.relu(self.conv3(x))
            x=self.relu(self.conv4(x))
            x=self.relu(self.conv5(x))
            x=self.output(self.conv6(x))

        return x
