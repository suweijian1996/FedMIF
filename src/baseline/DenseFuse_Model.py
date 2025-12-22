import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# DenseFuse network
class DenseFuse_net(nn.Module):
    def __init__(self):
        super(DenseFuse_net, self).__init__()

        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.DC1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.DC2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.DC3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1,
                      kernel_size=3, stride=1, padding=1))

    def encoder(self, input):
        x1 = self.C1(input)
        x2 = self.DC1(x1)
        cat = torch.cat((x1, x2), 1)
        x3 = self.DC2(cat)
        cat = torch.cat((x1, x2, x3), 1)
        x4 = self.DC3(cat)
        cat = torch.cat((x1, x2, x3, x4), 1)
        return cat

    # def fusion(self, en1, en2, strategy_type='addition'):
    #     # addition
    #     if strategy_type is 'attention_weight':
    #         # attention weight
    #         fusion_function = fusion_strategy.attention_fusion_weight
    #     else:
    #         fusion_function = fusion_strategy.addition_fusion
    #
    #     f_0 = fusion_function(en1[0], en2[0])
    #     return [f_0]

    def fusion(self, en1, en2, strategy_type='addition'):
        f_0 = (en1 + en2) / 2
        return f_0

    def decoder(self, f_en):
        x2 = self.C2(f_en)
        x3 = self.C3(x2)
        x4 = self.C4(x3)
        output = self.C5(x4)

        return output
