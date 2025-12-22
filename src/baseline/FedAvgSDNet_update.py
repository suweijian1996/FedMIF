#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import os
import torch
import itertools
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.utils import Gradient, cc, msssim, savenp, TVLoss
from data.Dataloader import unnormalize
from torchvision.utils import save_image


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image1, image2 = self.dataset[self.idxs[item]]
        return image1.clone(), image2.clone()

class DatasetSplit_ME(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image1, image2, image3 = self.dataset[self.idxs[item]]
        return image1.clone(), image2.clone(), image3.clone()

class LocalUpdate(object):
    def __init__(self, args, dataset, test_dataset, idxs=0, logger='', isTri=False):
        self.isTri = isTri
        self.args = args
        self.logger = logger
        self.trainloader = self.get_train_data(dataset, list(idxs))
        self.testloader = self.get_test_data(test_dataset)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        # Default criterion set to NLL loss function
        self.MESLoss = nn.MSELoss().to(self.device)
        self.L1Loss = nn.L1Loss().to(self.device)
        self.gradient = Gradient().to(self.device)
        self.tvLoss = TVLoss()

    def low_pass(self, input):
        # 定义滤波器，需要将其从 numpy 转换为 torch.Tensor
        # 并且增加批次大小和通道数的维度
        filter = torch.tensor([[0.0947, 0.1183, 0.0947], [0.1183, 0.1478, 0.1183], [0.0947, 0.1183, 0.0947]]).float().to(self.device)
        filter = filter.unsqueeze(0).unsqueeze(0)  # 增加批次大小和通道数的维度，对应于 [1, 1, 3, 3]

        # 应用卷积操作，padding 设置为 'same' 等效
        # PyTorch 中没有直接的 'same' padding 选项，但可以通过计算得到等效的 padding 数值
        padding = (filter.shape[-1] // 2, filter.shape[-2] // 2)  # 假设滤波器大小为奇数
        d = F.conv2d(input, filter, padding=padding)

        return d


    def get_train_data(self, dataset, idxs):
        """
        Returns train dataloaders for a given dataset and user indexes.
        """
        if self.isTri:
            trainloader = DataLoader(DatasetSplit_ME(dataset, idxs), batch_size=self.args.batch_size, shuffle=True,
                                     num_workers=8)
        else:
            trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True,
                                     num_workers=8)
        print(len(trainloader))
        return trainloader

    def get_test_data(self, dataset):
        testloader = DataLoader(dataset, batch_size=1, shuffle=False)
        print(len(testloader))
        return testloader

    def update_weights_IV(self, model, global_round, idx):
        model.train()
        epoch_loss = []
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (visible, infrared) in enumerate(self.trainloader):
                visible, infrared = visible.to(self.device), infrared.to(self.device)

                model.zero_grad()
                fused, output1, output2 = model(infrared, visible)

                S1 = torch.sign(torch.abs(self.gradient(self.low_pass(infrared))) - torch.min(
                    torch.abs(self.gradient(self.low_pass(infrared))),
                    torch.abs(self.gradient(self.low_pass(visible)))))
                loss_grad = torch.mean(S1 * (self.gradient(fused) - self.gradient(infrared)) ** 2) \
                            + torch.mean((1 - S1) * (self.gradient(fused) - self.gradient(visible)) ** 2)

                loss_int = torch.mean((fused - infrared) ** 2) + 0.5 * torch.mean((fused - visible) ** 2)
                loss_dc = torch.mean((output1 - infrared) ** 2) + torch.mean((output2 - visible) ** 2)
                loss_sf = 10*loss_grad + loss_int
                loss = loss_sf + loss_dc

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(visible),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / (len(epoch_loss))

    def update_weights_Medical(self, model, global_round, idx):
        model.train()
        epoch_loss = []
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (pet, mri) in enumerate(self.trainloader):
                pet, mri = pet.to(self.device), mri.to(self.device)

                model.zero_grad()
                fused, output1, output2 = model(mri, pet)

                S1 = torch.sign(torch.abs(self.gradient(self.low_pass(mri))) - torch.min(
                    torch.abs(self.gradient(self.low_pass(mri))),
                    torch.abs(self.gradient(self.low_pass(pet)))), out=None)
                loss_grad = torch.mean(S1 * (self.gradient(fused) - self.gradient(mri)) ** 2) \
                            + torch.mean((1 - S1) * (self.gradient(fused) - self.gradient(pet)) ** 2)

                loss_int = torch.mean((fused - mri) ** 2) + 0.5 * torch.mean((fused - pet) ** 2)
                loss_dc = torch.mean((output1 - mri) ** 2) + torch.mean((output2 - pet) ** 2)
                loss_sf = 80 * loss_grad + loss_int
                loss = loss_sf + loss_dc

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(pet),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / (len(epoch_loss))

    def update_weights_ME(self, model, global_round, idx):
        model.train()
        epoch_loss = []
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (over, under, _) in enumerate(self.trainloader):
                under, over = under.to(self.device), over.to(self.device)

                model.zero_grad()
                fused, output1, output2 = model(over, under)

                S1 = torch.sign(torch.abs(self.gradient(self.low_pass(over))) - torch.min(
                    torch.abs(self.gradient(self.low_pass(over))),
                    torch.abs(self.gradient(self.low_pass(under)))), out=None)
                loss_grad = torch.mean(S1 * (self.gradient(fused) - self.gradient(over)) ** 2) \
                            + torch.mean((1 - S1) * (self.gradient(fused) - self.gradient(under)) ** 2)

                loss_int = torch.mean((fused - over) ** 2) + 1 * torch.mean((fused - under) ** 2)
                loss_dc = torch.mean((output1 - over) ** 2) + torch.mean((output2 - under) ** 2)
                loss_sf = 50 * loss_grad + loss_int
                loss = loss_sf + loss_dc

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(under),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / (len(epoch_loss))

    def update_weights_MF(self, model, global_round, idx):
        model.train()
        epoch_loss = []
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (far, near, _) in enumerate(self.trainloader):
                far, near = far.to(self.device), near.to(self.device)

                model.zero_grad()
                fused, output1, output2 = model(far, near)
                S1 = torch.sign(torch.abs(self.gradient(self.low_pass(far))) - torch.min(
                    torch.abs(self.gradient(self.low_pass(far))),
                    torch.abs(self.gradient(self.low_pass(near)))), out=None)
                loss_grad = torch.mean(S1 * (self.gradient(fused) - self.gradient(far)) ** 2) \
                            + torch.mean((1 - S1) * (self.gradient(fused) - self.gradient(near)) ** 2)

                loss_int = torch.mean((fused - far) ** 2) + 1 * torch.mean((fused - near) ** 2)
                loss_dc = torch.mean((output1 - far) ** 2) + torch.mean((output2 - near) ** 2)
                loss_sf = 3 * loss_grad + loss_int
                loss = loss_sf + loss_dc

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(far),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / (len(epoch_loss))



    def inference(self, model, epoch, i=0):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (visible_img, infrared_img) in enumerate(self.testloader):
            infrared_img, visible_img = infrared_img.to(self.device), visible_img.to(self.device)

            # Inference
            vis_en = model.encoder(visible_img)
            ir_en = model.encoder(infrared_img)
            fusion_en = model.fusion(vis_en, ir_en)
            fused = model.decoder(fusion_en)
            #     # 保存目录
            save_dir = r"E:\Datasets\FedFusion\test\result/"
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #
            #     # 保存图像
            sd, en = savenp(fused * 0.5 + 0.5, save_dir + f"fused_{batch_idx}_{epoch}_{i}.png")
            print(sd, en)
            # save_image(fused_img, os.path.join(save_dir, f"fused_{batch_idx}_{epoch}_{i}.png"))
            # save_image(output1, os.path.join(save_dir, f"out1_{batch_idx}.png"))
            # save_image(output2, os.path.join(save_dir, f"out2_{batch_idx}.png"))

            batch_loss = self.MESLoss(self.gradient(fused), self.gradient(visible_img)) + self.MESLoss(fused,
                                                                                                       infrared_img)
            loss += batch_loss.item()
        return loss



def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss
