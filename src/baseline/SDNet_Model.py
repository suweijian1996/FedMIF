import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# DenseFuse network
class SDNet_net(nn.Module):
    def __init__(self):
        super(SDNet_net, self).__init__()

        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.DC1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.DC2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.DC3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.B1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.DB1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.DB2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.DB3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1,
                      kernel_size=1, stride=1),
            nn.Tanh())

        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.De1_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.De1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=4,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU())

        self.De1_3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.Tanh())

        self.De2_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.De2_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=4,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU())

        self.De2_3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.Tanh())

    def forward(self, input, input2):
        x1 = self.C1(input)
        x2 = self.DC1(x1)
        cat = torch.cat((x1, x2), 1)
        x3 = self.DC2(cat)
        cat = torch.cat((x1, x2, x3), 1)
        x4 = self.DC3(cat)
        cat = torch.cat((x1, x2, x3, x4), 1)

        y1 = self.B1(input2)
        y2 = self.DB1(y1)
        cat1 = torch.cat((y1, y2), 1)
        y3 = self.DB2(cat1)
        cat1 = torch.cat((y1, y2, y3), 1)
        y4 = self.DB3(cat1)
        cat1 = torch.cat((y1, y2, y3, y4), 1)

        feature = torch.cat((cat, cat1), 1)
        fused = self.C2(feature)

        feature = self.C3(fused)

        x5 = self.De1_1(feature)
        x6 = self.De1_2(x5)
        x7 = self.De1_3(x6)

        y5 = self.De2_1(feature)
        y6 = self.De2_2(y5)
        y7 = self.De2_3(y6)
        return fused, x7, y7

