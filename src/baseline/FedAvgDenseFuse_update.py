#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import os
import torch
import itertools
from torch import nn
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
        image1 = self.dataset[self.idxs[item]]
        return image1.clone()
        # return torch.tensor(image1), torch.tensor(image2)

class LocalUpdate(object):
    def __init__(self, args, dataset, test_dataset, idxs=0, logger=''):
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

    def get_train_data(self, dataset, idxs):
        """
        Returns train dataloaders for a given dataset and user indexes.
        """
        trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True, num_workers=8)
        print(len(trainloader))
        return trainloader

    def get_test_data(self, dataset):
        testloader = DataLoader(dataset, batch_size=1, shuffle=False)
        print(len(testloader))
        return testloader

    def update_weights(self, model, global_round, idx):
        model.train()
        epoch_loss = []
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, natural_img in enumerate(self.trainloader):
                natural_img = natural_img.to(self.device)

                model.zero_grad()
                en = model.encoder(natural_img)
                output = model.decoder(en)
                loss_ssim = (1-msssim(output, natural_img))
                loss_mse = self.MESLoss(output, natural_img)
                loss = 1000*loss_ssim + loss_mse
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(natural_img),
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
