import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import spectral
import numpy
from skimage.feature import local_binary_pattern
import numpy as np


class GlobalGatedModule2(nn.Module):
    def __init__(self, in_patch_size, out_patch_size):
        super(GlobalGatedModule2, self).__init__()
        self.in_patch_size = in_patch_size
        self.out_patch_size = out_patch_size
        self.fc1 = nn.Linear(in_patch_size * in_patch_size, int(out_patch_size * out_patch_size / 2))
        self.fc2 = nn.Linear(int(out_patch_size * out_patch_size / 2), out_patch_size * out_patch_size)

    def forward(self, x_in):
        x_1 = torch.nn.Flatten(start_dim=2, end_dim=3)(x_in)
        x_2 = nn.ReLU()(self.fc1(x_1))
        x_3 = self.fc2(x_2)
        x_3 = x_3.reshape(x_3.shape[0], x_3.shape[1], self.out_patch_size, self.out_patch_size)
        weight = nn.Sigmoid()(x_3)
        # out = x_in * weight
        # return out
        return weight


class GlobalLocalGatedConvNet(nn.Module):
    def __init__(self, classes, HSI_Data_Shape_H, HSI_Data_Shape_W, HSI_Data_Shape_C,
                 patch_size):  # band:103  classes=9
        super(GlobalLocalGatedConvNet, self).__init__()
        self.name = 'GatedConvNet'
        self.classes = classes
        self.HSI_Data_Shape_H = HSI_Data_Shape_H
        self.HSI_Data_Shape_W = HSI_Data_Shape_W
        self.band = HSI_Data_Shape_C
        self.patch_size = patch_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.band, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # self.local_gated_module1 = Gated_Conv(128, 128, ksize=3)
        # self.local_gated_module2 = Gated_Conv(256, 256, ksize=3)
        # self.local_gated_module3 = Gated_Conv(512, 512, ksize=3)

        self.local_gated_conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.local_gated_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.local_gated_conv3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.local_gated_conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        # self.global_gated_module1 = GlobalGatedModule(patch_size=patch_size)
        # self.global_gated_module2 = GlobalGatedModule(patch_size=int(patch_size/2))
        # self.global_gated_module3 = GlobalGatedModule(patch_size=int(patch_size/4))

        # self.gaijinECA1 = gaijinECA_Module(patch_size=31)
        # self.gaijinECA2 = gaijinECA_Module(patch_size=15)
        # self.gaijinECA3 = gaijinECA_Module(patch_size=7)

        self.fusion1 = nn.Sequential(
            # nn.Conv2d(128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.AdaptiveAvgPool2d(1)
            nn.Linear(128 * 2, self.classes),
        )

        self.fusion2 = nn.Sequential(
            nn.Linear(256 * 2, self.classes),
            # nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.AdaptiveAvgPool2d(1)
        )

        self.fusion3 = nn.Sequential(
            nn.Linear(512 , self.classes),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.AdaptiveAvgPool2d(1)
        )

        self.fusion4 = nn.Sequential(
            nn.Linear(512, self.classes),
        )

        self.conv4to3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv3to2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv2to1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # self.global_gate43 = GlobalGatedModule2(in_patch_size=3, out_patch_size=7)
        # self.global_gate32 = GlobalGatedModule2(in_patch_size=7, out_patch_size=15)
        # self.global_gate21 = GlobalGatedModule2(in_patch_size=15, out_patch_size=31)

        self.global_gate43 = GlobalGatedModule2(in_patch_size=int(patch_size/8), out_patch_size=int(patch_size/4))
        self.global_gate32 = GlobalGatedModule2(in_patch_size=int(patch_size/4), out_patch_size=int(patch_size/2))
        self.global_gate21 = GlobalGatedModule2(in_patch_size=int(patch_size/2), out_patch_size=patch_size)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.finally_fc_classification = nn.Linear(512 * 4, self.classes)

        self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.25]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.25]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([0.25]))
        self.coefficient4 = torch.nn.Parameter(torch.Tensor([0.25]))

    def forward(self, patchX, pixelX):  # x:(64,103,31,31)

        """------------------------branch (31×31)------------------------"""
        x_input = patchX
        x1 = self.conv1(x_input)
        x2 = self.conv2(x1)
        # local_map1 = self.local_gated_module1(x2)
        # local_map1 = self.gaijinECA1(x2)
        # global_map1 = self.global_gated_module1(x2)
        # out1 = self.fusion1(torch.cat((local_map1, global_map1), dim=1))
        local_gate1 = F.softmax(self.local_gated_conv1(x2), dim=1)

        x3 = nn.MaxPool2d(2)(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x3)

        conv2to1 = self.conv2to1(x4)
        global_gate2to1 = self.global_gate21(conv2to1)
        out1 = nn.AdaptiveAvgPool2d(1)(torch.cat((local_gate1 * x2, global_gate2to1 * x2), dim=1))
        # out1 = nn.AdaptiveAvgPool2d(1)(local_gate1 * x2)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fusion1(out1)
        out1 = F.softmax(out1, dim=1)

        # local_map2 = self.local_gated_module2(x4)
        # local_map2 = self.gaijinECA2(x4)
        # global_map2 = self.global_gated_module2(x4)
        # out2 = self.fusion2(torch.cat((local_map2, global_map2), dim=1))
        # out2 = self.fusion2(local_map2)
        # out2 = out2.view(out2.size(0), -1)
        local_gate2 = F.softmax(self.local_gated_conv2(x4), dim=1)
        # global_gate2 = self.global_gated_module2(x4)
        # out2 = nn.AdaptiveAvgPool2d(1)(torch.cat((local_gate2, global_gate2), dim=1))
        # out2 = out2.view(out2.size(0), -1)
        # out2 = self.fusion2(out2)
        # out2 = F.softmax(out2, dim=1)
        # conv2to1 = self.conv2to1(x4)
        # global_gate2to1 = self.global_gate21(conv2to1)
        # out1 = nn.AdaptiveAvgPool2d(1)(torch.cat((local_gate1 * x2, global_gate2to1 * x2), dim=1))
        # out1 = out1.view(out1.size(0), -1)
        # out1 = self.fusion1(out1)
        # out1 = F.softmax(out1, dim=1)

        x5 = nn.MaxPool2d(2)(x4)
        x5 = self.conv5(x5)
        x6 = self.conv6(x5)
        # local_map3 = self.local_gated_module3(x6)
        # local_map3 = self.gaijinECA3(x6)
        # global_map3 = self.global_gated_module3(x6)
        # out3 = self.fusion3(torch.cat((local_map3, global_map3), dim=1))
        # out3 = self.fusion3(local_map3)
        # out3 = out3.view(out3.size(0), -1)
        conv3to2 = self.conv3to2(x6)
        global_gate3to2 = self.global_gate32(conv3to2)
        out2 = nn.AdaptiveAvgPool2d(1)(torch.cat((local_gate2 * x4, global_gate3to2 * x4), dim=1))
        # out2 = nn.AdaptiveAvgPool2d(1)(local_gate2 * x4)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.fusion2(out2)
        out2 = F.softmax(out2, dim=1)

        local_gate3 = F.softmax(self.local_gated_conv3(x6), dim=1)

        x7 = nn.MaxPool2d(2)(x6)
        x7 = self.conv7(x7)

        # local_gate4 = F.softmax(self.local_gated_conv4(x7), dim=1)
        # out4 = nn.AdaptiveAvgPool2d(1)(x7 * local_gate4)
        # conv4to3 = self.conv4to3(x7)
        # global_gate4to3 = self.global_gate43(conv4to3)
        # out3 = nn.AdaptiveAvgPool2d(1)(torch.cat((local_gate3 * x6, global_gate4to3 * x6), dim=1))
        out3 = nn.AdaptiveAvgPool2d(1)(local_gate3 * x6)
        out3 = out3.view(out3.size(0), -1)
        out3 = self.fusion3(out3)
        out3 = F.softmax(out3, dim=1)

        out4 = nn.AdaptiveAvgPool2d(1)(x7)
        out4 = out4.view(out4.size(0), -1)
        out4 = self.fusion4(out4)
        out4 = F.softmax(out4, dim=1)

        output = self.coefficient1 * out1 + self.coefficient2 * out2 + self.coefficient3 * out3 + self.coefficient4 * out4

        return output, output