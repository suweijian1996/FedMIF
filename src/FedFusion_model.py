import torch
import torch.nn as nn
from torch.nn import functional as F
import numbers
from einops import rearrange


class DetailNet(nn.Module):
    def __init__(self):
        super(DetailNet, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))

        self.DC1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))

        self.DC2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))

        self.DC3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))

        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True))

        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))

        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.Tanh())


    def forward(self, input):
        x1 = self.C1(input)
        x2 = self.DC1(x1)
        cat = torch.cat((x1, x2), 1)
        x3 = self.DC2(cat)
        cat = torch.cat((x1, x2, x3), 1)
        x4 = self.DC3(cat)
        cat = torch.cat((x1, x2, x3, x4), 1)
        x = self.C2(cat)
        x = self.C3(x)
        x = self.C4(x)
        return x

class DetailNet_x(nn.Module):
    def __init__(self):
        super(DetailNet_m, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))

        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True))

        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))

        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.Tanh())
    def forward(self, input):
        x = self.C1(input)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        return x


class DetailNet_s(nn.Module):
    def __init__(self):
        super(DetailNet_s, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))

        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        

    def forward(self, input):
        x = self.C1(input)
        x = self.C4(x)
        return x


class DetailNet_m(nn.Module):
    def __init__(self):
        super(DetailNet_m, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))
        
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))
        
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))
        
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True))

        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        

    def forward(self, input):
        x = self.C1(input)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        x = self.C6(x)
        return x
       
class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.2, True))

        self.DC1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.2, True))

        self.DC2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.2, True))

        self.DC3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.2, True))
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2, True))

    def forward(self, input):
        x1 = self.C1(input)
        x2 = self.DC1(x1)
        cat = torch.cat((x1, x2), 1)
        x3 = self.DC2(cat)
        cat = torch.cat((x1, x2, x3), 1)
        x4 = self.DC3(cat)
        cat = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.C2(cat)
        return x5


class DenseBlock_wo_Norm(nn.Module):
    def __init__(self):
        super(DenseBlock_wo_Norm, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True))

        self.DC1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True))

        self.DC2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True))

        self.DC3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True))
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True))

    def get_adapter_feature(self, input):
        x1 = self.C1(input)
        x2 = self.DC1(x1)
        cat = torch.cat((x1, x2), 1)
        x3 = self.DC2(cat)
        cat = torch.cat((x1, x2, x3), 1)
        x4 = self.DC3(cat)
        return x1, x2, x3, x4

    def get_adapter_result(self, x1, x2, x3, x4):
        cat = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.C2(cat)
        return x5


    def forward(self, input):
        x1 = self.C1(input)
        x2 = self.DC1(x1)
        cat = torch.cat((x1, x2), 1)
        x3 = self.DC2(cat)
        cat = torch.cat((x1, x2, x3), 1)
        x4 = self.DC3(cat)
        cat = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.C2(cat)
        return x5

class TwoBranchDecoder(nn.Module):
    def __init__(self):
        super(TwoBranchDecoder, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True)
        )
        self.C2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.C3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, True)
        )

        self.C4 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.B1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True)
        )
        self.B2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.B3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, True)
        )
        self.B4 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.C1(input)
        x = self.C2(x)
        x = self.C3(x)
        out1 = self.C4(x)

        y = self.B1(input)
        y = self.B2(y)
        y = self.B3(y)
        out2 = self.B4(y)

        return out1, out2

class FusionDecoder(nn.Module):
    def __init__(self):
        super(FusionDecoder, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2, True)
        )
        self.C2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.2, True)
        )
        self.C3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 8),
            nn.LeakyReLU(0.2, True)
        )
        self.C4 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.C1(input)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        return x

class FusionDecoder_wo_Norm(nn.Module):
    def __init__(self):
        super(FusionDecoder_wo_Norm, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.C2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.C3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.C4 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.C1(input)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        return x
      
class FedMIFModel_GN(nn.Module):
    def __init__(self,
                 dense_encoder=DenseBlock(),
                 decoder=FusionDecoder(),
                 tow_branch_decoder=TwoBranchDecoder(),
                 detail_net=DetailNet()
                 ):
        super(FedMIFModel_GN, self).__init__()
        self.fusion_encoder = dense_encoder
        #  64 out
        self.fusion_decoder = decoder
        # 64 in
        self.two_branch_decoder = tow_branch_decoder
        # end-to-end
        self.detail_net = detail_net

    # stage I feature extract
    def feature_extract(self, input):
        x = self.fusion_encoder(input)
        out, out1 = self.two_branch_decoder(x)
        return out, out1

    # stage I common fusion
    def feature_fusion(self, input):
        x = self.fusion_encoder(input)
        x = self.fusion_decoder(x)
        return x

class FedMIFModel(nn.Module):
    def __init__(self,
                 dense_encoder=DenseBlock_wo_Norm(),
                 decoder=FusionDecoder_wo_Norm(),
                 tow_branch_decoder=TwoBranchDecoder(),
                 detail_net=DetailNet()
                 ):
        super(FedMIFModel, self).__init__()
        self.fusion_encoder = dense_encoder
        #  64 out
        self.fusion_decoder = decoder
        # 64 in
        self.two_branch_decoder = tow_branch_decoder
        # end-to-end
        self.detail_net = detail_net

    # stage I feature extract
    def feature_extract(self, input):
        x = self.fusion_encoder(input)
        out, out1 = self.two_branch_decoder(x)
        return out, out1

    # stage I common fusion
    def feature_fusion(self, input):
        x = self.fusion_encoder(input)
        x = self.fusion_decoder(x)
        return x