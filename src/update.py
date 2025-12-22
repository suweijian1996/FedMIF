#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import os

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

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
        image1, image2 = self.dataset[self.idxs[item]]
        return image1.clone(), image2.clone()
        # return torch.tensor(image1), torch.tensor(image2)

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
    def __init__(self, args, dataset, test_dataset, idxs=0, logger='', isME=False):
        self.args = args
        self.logger = logger
        self.trainloader = self.get_train_data(dataset, list(idxs), isME)
        self.testloader = self.get_test_data(test_dataset)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        # Default criterion set to NLL loss function
        self.MESLoss = nn.MSELoss().to(self.device)
        self.L1Loss = nn.L1Loss().to(self.device)
        self.gradient = Gradient().to(self.device)
        self.tvLoss = TVLoss()

    def get_train_data(self, dataset, idxs, isME=False):
        """
        Returns train dataloaders for a given dataset and user indexes.
        """
        if isME:
            trainloader = DataLoader(DatasetSplit_ME(dataset, idxs), batch_size=self.args.batch_size, shuffle=True, num_workers=8)
        else:
            trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True, num_workers=8)
        print(len(trainloader))
        return trainloader

    def get_test_data(self, dataset):
        testloader = DataLoader(dataset, batch_size=1, shuffle=False)
        print(len(testloader))
        return testloader

    def update_encoder_weights(self, model, global_round, idx):
        model.fusion_encoder.train()
        model.two_branch_decoder.eval()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.fusion_encoder.parameters(),
                lr=self.args.lr,
                momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.fusion_encoder.parameters(),
                lr=self.args.lr,
                betas=[0.5, 0.999])
        elif self.args.optimizer == 'rms':
            optimizer = torch.optim.RMSprop(
                model.fusion_encoder.parameters(),
                lr=self.args.lr)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (visible_img, infrared_img) in enumerate(self.trainloader):
                visible_img, infrared_img = visible_img.to(self.device), infrared_img.to(self.device)

                model.zero_grad()
                out, out1 = model.feature_extract(torch.cat((infrared_img, visible_img), 1))
                loss_out = self.L1Loss(out, infrared_img) + 10 *self.L1Loss(self.gradient(out),
                                                                             self.gradient(infrared_img))
                loss_out1 = self.L1Loss(out1, visible_img) + 10 * self.L1Loss(self.gradient(out1),
                                                                                self.gradient(visible_img))
                loss = loss_out + loss_out1
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(infrared_img),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.fusion_encoder.state_dict(), sum(epoch_loss) / (len(epoch_loss))

    def update_encoder_weights_MF(self, model, global_round, idx):
        model.fusion_encoder.train()
        model.two_branch_decoder.eval()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.fusion_encoder.parameters(),
                lr=self.args.lr,
                momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.fusion_encoder.parameters(),
                lr=self.args.lr,
                betas=[0.5, 0.999])
        elif self.args.optimizer == 'rms':
            optimizer = torch.optim.RMSprop(
                model.fusion_encoder.parameters(),
                lr=self.args.lr)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (far_img, near_img, gt_img) in enumerate(self.trainloader):
                far_img, near_img = far_img.to(self.device), near_img.to(self.device)

                model.zero_grad()
                out, out1 = model.feature_extract(torch.cat((far_img, near_img), 1))
                loss_out = self.L1Loss(out, far_img) + 10 * self.L1Loss(self.gradient(out),
                                                                             self.gradient(far_img))
                loss_out1 = self.L1Loss(out1, near_img) + 10 * self.L1Loss(self.gradient(out1),
                                                                                self.gradient(near_img))
                loss = loss_out + loss_out1
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(near_img),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.fusion_encoder.state_dict(), sum(epoch_loss) / (len(epoch_loss))

    def update_encoder_weights_ME(self, model, global_round, idx):
        model.fusion_encoder.train()
        model.two_branch_decoder.eval()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.fusion_encoder.parameters(),
                lr=self.args.lr,
                momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.fusion_encoder.parameters(),
                lr=self.args.lr,
                betas=[0.5, 0.999])
        elif self.args.optimizer == 'rms':
            optimizer = torch.optim.RMSprop(
                model.fusion_encoder.parameters(),
                lr=self.args.lr)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (under_img, over_img, gt_img) in enumerate(self.trainloader):
                under_img, over_img = under_img.to(self.device), over_img.to(self.device)

                model.zero_grad()
                out, out1 = model.feature_extract(torch.cat((under_img, over_img), 1))
                loss_out = 0.1*self.L1Loss(out, under_img) + 10* self.L1Loss(self.gradient(out),
                                                                             self.gradient(under_img))
                loss_out1 = 0.1*self.L1Loss(out1, over_img) + 10*self.L1Loss(self.gradient(out1),
                                                                                self.gradient(over_img))
                loss = loss_out + loss_out1
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(over_img),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.fusion_encoder.state_dict(), sum(epoch_loss) / (len(epoch_loss))

    def update_weights(self, model, global_round, idx):
        # last_round_model = copy.deepcopy(model)
        # Set mode to train model
        # model.encoder.train()
        # model.fusionDecoder.train()
        model.fusion_encoder.eval()
        model.fusion_decoder.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.fusion_decoder.parameters(),
                lr=self.args.lr,
                momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.fusion_decoder.parameters(),
                lr=self.args.lr,
                betas=[0.5, 0.999])
        elif self.args.optimizer == 'rms':
            optimizer = torch.optim.RMSprop(
                model.fusion_decoder.parameters(),
                lr=self.args.lr)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (visible_img, infrared_img) in enumerate(self.trainloader):
                visible_img, infrared_img = visible_img.to(self.device), infrared_img.to(self.device)

                model.zero_grad()
                result = model.feature_fusion(torch.cat((infrared_img, visible_img), 1))
                loss_int = self.L1Loss(result, torch.max(infrared_img, visible_img))
                # loss_int = (1 - msssim(result, infrared_img)) + (1 - msssim(result, visible_img))
                # loss_int = self.L1Loss(result, infrared_img) + 0.5 * self.L1Loss(result, visible_img)
                loss_grad = self.L1Loss(self.gradient(result),
                                        torch.max(self.gradient(infrared_img), self.gradient(visible_img)))
                loss = loss_int + 10 * loss_grad
                loss.backward()
                optimizer.step()
                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLossir: {:.6f}|Lossvis: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(infrared_img),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item(),
                            loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.fusion_decoder.state_dict(), sum(epoch_loss) / (len(epoch_loss))

    def update_weights_medical(self, model, global_round, idx):
        # last_round_model = copy.deepcopy(model)
        # Set mode to train model
        # model.encoder.train()
        # model.fusionDecoder.train()
        model.fusion_encoder.eval()
        model.fusion_decoder.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.fusion_decoder.parameters(),
                lr=self.args.lr,
                momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.fusion_decoder.parameters(),
                lr=self.args.lr,
                betas=[0.5, 0.999])
        elif self.args.optimizer == 'rms':
            optimizer = torch.optim.RMSprop(
                model.fusion_decoder.parameters(),
                lr=self.args.lr)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (visible_img, infrared_img) in enumerate(self.trainloader):
                visible_img, infrared_img = visible_img.to(self.device), infrared_img.to(self.device)

                model.zero_grad()
                result = model.feature_fusion(torch.cat((infrared_img, visible_img), 1))
                loss_int = self.L1Loss(result, torch.max(infrared_img, visible_img))
                loss_grad = self.L1Loss(self.gradient(result),
                                        torch.max(self.gradient(infrared_img), self.gradient(visible_img)))
                loss = loss_int + 10 * loss_grad
                loss.backward()
                optimizer.step()
                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLossir: {:.6f}|Lossvis: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(infrared_img),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item(),
                            loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.fusion_decoder.state_dict(), sum(epoch_loss) / (len(epoch_loss))

    def update_MF_decoder_weights(self, model, global_round, idx):
        # last_round_model = copy.deepcopy(model)
        # Set mode to train model
        # model.encoder.train()
        # model.fusionDecoder.train()
        model.fusion_encoder.eval()
        model.fusion_decoder.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.fusion_decoder.parameters(),
                lr=self.args.lr,
                momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.fusion_decoder.parameters(),
                lr=self.args.lr,
                betas=[0.5, 0.999])
        elif self.args.optimizer == 'rms':
            optimizer = torch.optim.RMSprop(
                model.fusion_decoder.parameters(),
                lr=self.args.lr)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (far_img, near_img, gt_img) in enumerate(self.trainloader):
                far_img, near_img, gt_img = far_img.to(self.device), near_img.to(self.device), gt_img.to(self.device)

                model.zero_grad()
                result = model.feature_fusion(torch.cat((far_img, near_img), 1))
                # loss_int = (1 - msssim(result, far_img)) + (1 - msssim(result, near_img))
                # loss_grad = self.L1Loss(self.gradient(result),
                #                         torch.max(self.gradient(far_img), self.gradient(near_img)))
                # loss_l1 = self.MESLoss(result, far_img) + self.MESLoss(result, near_img)
                loss_int = (1 - msssim(result, gt_img))
                loss_grad = self.L1Loss(self.gradient(result), self.gradient(gt_img))
                loss = loss_int + 10 * loss_grad
                # loss = 100 * loss_int +  loss_l1
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLossir: {:.6f}|Lossvis: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(far_img),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item(),
                            loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.fusion_decoder.state_dict(), sum(epoch_loss) / (len(epoch_loss))

    def update_ME_decoder_weights(self, model, global_round, idx):
        # last_round_model = copy.deepcopy(model)
        # Set mode to train model
        # model.encoder.train()
        # model.fusionDecoder.train()
        model.fusion_encoder.eval()
        model.fusion_decoder.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.fusion_decoder.parameters(),
                lr=self.args.lr,
                momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.fusion_decoder.parameters(),
                lr=self.args.lr,
                betas=[0.5, 0.999])
        elif self.args.optimizer == 'rms':
            optimizer = torch.optim.RMSprop(
                model.fusion_decoder.parameters(),
                lr=self.args.lr)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (under_img, over_img, gt_img) in enumerate(self.trainloader):
                under_img, over_img, gt_img = under_img.to(self.device), over_img.to(self.device), gt_img.to(self.device)

                model.zero_grad()
                result = model.feature_fusion(torch.cat((under_img, over_img), 1))
                loss_int = (1 - msssim(result, gt_img))
                loss_grad = self.L1Loss(self.gradient(result), self.gradient(gt_img))
                loss = loss_int + 10 * loss_grad
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLossir: {:.6f}|Lossvis: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(gt_img),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item(),
                            loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.fusion_decoder.state_dict(), sum(epoch_loss) / (len(epoch_loss))


    def update_detail_weights(self, model, global_round, idx):
        # last_round_model = copy.deepcopy(model)
        # Set mode to train model
        model.fusion_encoder.eval()
        model.fusion_decoder.eval()
        model.detail_net.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer_detail = torch.optim.SGD(model.detail_net.parameters(),
                                               lr=self.args.lr,
                                               momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer_detail = torch.optim.Adam(model.detail_net.parameters(),
                                                lr=self.args.lr,
                                                betas=[0.5, 0.999])
        elif self.args.optimizer == 'rms':
            optimizer_detail = torch.optim.RMSprop(model.detail_net.parameters(),
                                                   lr=self.args.lr)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (visible_img, infrared_img) in enumerate(self.trainloader):
                visible_img, infrared_img = visible_img.to(self.device), infrared_img.to(self.device)

                model.zero_grad()
                result = model.feature_fusion(torch.cat((infrared_img, visible_img), 1)).detach()
                inf = model.detail_net(torch.cat((infrared_img, visible_img), 1))
                resi_diff = inf + result
                loss_int = self.L1Loss(resi_diff, infrared_img) + 0.5*self.L1Loss(resi_diff, visible_img)
                loss_grad = self.L1Loss(self.gradient(resi_diff), self.gradient(visible_img))
                loss = loss_int + 10*loss_grad
                loss.backward()
                optimizer_detail.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLossir: {:.6f}Lossvis: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(infrared_img),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item(),
                            loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), model.detail_net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_detail_mf_weights(self, model, global_round, idx):
        # last_round_model = copy.deepcopy(model)
        # Set mode to train model
        model.fusion_encoder.eval()
        model.fusion_decoder.eval()
        model.detail_net.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer_detail = torch.optim.SGD(model.detail_net.parameters(),
                                               lr=self.args.lr,
                                               momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer_detail = torch.optim.Adam(model.detail_net.parameters(),
                                                lr=self.args.lr,
                                                betas=[0.5, 0.999])
        elif self.args.optimizer == 'rms':
            optimizer_detail = torch.optim.RMSprop(model.detail_net.parameters(),
                                                   lr=self.args.lr)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (far_img, near_img, gt_img) in enumerate(self.trainloader):
                far_img, near_img, gt_img = far_img.to(self.device), near_img.to(self.device), gt_img.to(self.device)

                model.zero_grad()
                result = model.feature_fusion(torch.cat((far_img, near_img), 1)).detach()
                inf = model.detail_net(torch.cat((far_img, near_img), 1))
                resi_diff = inf + result
                # loss_ssim = 100 * ((1-msssim(resi_diff, far_img)) + (1-msssim(resi_diff, near_img)))
                # loss_TV = self.tvLoss(resi_diff)
                loss_int = self.MESLoss(resi_diff, gt_img)
                loss_grad = self.L1Loss(self.gradient(resi_diff), self.gradient(gt_img))
                loss = loss_int + 10 * loss_grad
                # loss = self.L1Loss(resi_diff, ((far_img - self.gradient(far_img)) + (near_img - self.gradient(near_img)))/2) \
                #        + 10 * self.L1Loss(self.gradient(resi_diff), torch.max(self.gradient(far_img), self.gradient(near_img)))
                # loss = loss_TV + loss_ssim
                loss.backward()
                optimizer_detail.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLosstv: {:.6f}Lossssim: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(near_img),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss_int.item(),
                            loss_grad.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), model.detail_net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_detail_me_weights(self, model, global_round, idx):
        # last_round_model = copy.deepcopy(model)
        # Set mode to train model
        model.fusion_encoder.eval()
        model.fusion_decoder.eval()
        model.detail_net.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer_detail = torch.optim.SGD(model.detail_net.parameters(),
                                               lr=self.args.lr,
                                               momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer_detail = torch.optim.Adam(model.detail_net.parameters(),
                                                lr=self.args.lr,
                                                betas=[0.5, 0.999])
        elif self.args.optimizer == 'rms':
            optimizer_detail = torch.optim.RMSprop(model.detail_net.parameters(),
                                                   lr=self.args.lr)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (under_img, over_img, gt_img) in enumerate(self.trainloader):
                under_img, over_img, gt_img = under_img.to(self.device), over_img.to(self.device), gt_img.to(self.device)

                model.zero_grad()
                result = model.feature_fusion(torch.cat((under_img, over_img), 1)).detach()
                inf = model.detail_net(torch.cat((under_img, over_img), 1))
                resi_diff = inf + result
                loss = self.MESLoss(resi_diff, gt_img) + 10 * self.L1Loss(self.gradient(resi_diff), self.gradient(gt_img))
                loss.backward()
                optimizer_detail.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Client : {}| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLossir: {:.6f}Lossvis: {:.6f}'.format(
                            idx, global_round, iter, batch_idx * len(gt_img),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader), loss.item(),
                            loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), model.detail_net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference_person(self, model, epoch, i=0):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (visible_img, infrared_img) in enumerate(self.testloader):
            infrared_img, visible_img = infrared_img.to(self.device), visible_img.to(self.device)

            # Inference
            fused = model.feature_fusion(infrared_img, visible_img)
            ir = model.detail_net(infrared_img)
            fused = fused + ir
            # fused_img = unnormalize(fused)
            # output1 = unnormalize(output1)
            # output2 = unnormalize(output2)
            #
            #     # 保存目录
            save_dir = r"/media/data1/swj/FedFusion/test/result/"
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #
            #     # 保存图像
            sd, en = savenp((fused) * 0.5 + 0.5, save_dir + str(i) + f"fused_{batch_idx}_{epoch}_{i}.png")
            print(sd, en)
            # save_image(fused_img, os.path.join(save_dir, f"fused_{batch_idx}_{epoch}_{i}.png"))
            # save_image(output1, os.path.join(save_dir, f"out1_{batch_idx}.png"))
            # save_image(output2, os.path.join(save_dir, f"out2_{batch_idx}.png"))

            batch_loss = self.MESLoss(self.gradient(fused), self.gradient(visible_img)) + self.MESLoss(fused,
                                                                                                       infrared_img)
            loss += batch_loss.item()
        return loss

    def inference(self, model, epoch, i=0):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (visible_img, infrared_img) in enumerate(self.testloader):
            infrared_img, visible_img = infrared_img.to(self.device), visible_img.to(self.device)

            # Inference
            fused = model.feature_fusion(torch.cat((infrared_img, visible_img), 1))
            local_Detail = model.detail_net(torch.cat((infrared_img, visible_img), 1))
            fused = local_Detail + fused
            #     # 保存目录
            save_dir = r"/media/data1/swj/FedFusion/test/result/"
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

    def inference_Medical(self, model, epoch, i=0):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (pet_img, mri_img) in enumerate(self.testloader):
            mri_img, pet_img = mri_img.to(self.device), pet_img.to(self.device)

            # Inference
            fused = model.feature_fusion(torch.cat((mri_img, pet_img), 1))
            local_Detail = model.detail_net(torch.cat((mri_img, pet_img), 1))
            fused = local_Detail + fused
            #     # 保存目录
            save_dir = r"/media/data1/swj/FedFusion/test/result/"
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #
            #     # 保存图像
            sd, en = savenp(fused * 0.5 + 0.5, save_dir + f"fused_Medical_{batch_idx}_{epoch}_{i}.png")
            print(sd, en)
            # save_image(fused_img, os.path.join(save_dir, f"fused_{batch_idx}_{epoch}_{i}.png"))
            # save_image(output1, os.path.join(save_dir, f"out1_{batch_idx}.png"))
            # save_image(output2, os.path.join(save_dir, f"out2_{batch_idx}.png"))

            batch_loss = self.MESLoss(self.gradient(fused), self.gradient(mri_img)) + self.MESLoss(fused,
                                                                                                       pet_img)
            loss += batch_loss.item()
        return loss

    def inference_MF(self, model, epoch, i=0):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (far_img, near_img, gt_img) in enumerate(self.testloader):
            far_img, near_img = far_img.to(self.device), near_img.to(self.device)

            # Inference
            fused = model.feature_fusion(torch.cat((far_img, near_img), 1))
            local_Detail = model.detail_net(torch.cat((far_img, near_img), 1))
            fused = local_Detail + fused
            #     # 保存目录
            save_dir = r"/media/data1/swj/FedFusion/test/result/"
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #
            #     # 保存图像
            sd, en = savenp(fused * 0.5 + 0.5, save_dir + f"fused_MF_{batch_idx}_{epoch}_{i}.png")
            print(sd, en)
            # save_image(fused_img, os.path.join(save_dir, f"fused_{batch_idx}_{epoch}_{i}.png"))
            # save_image(output1, os.path.join(save_dir, f"out1_{batch_idx}.png"))
            # save_image(output2, os.path.join(save_dir, f"out2_{batch_idx}.png"))

            batch_loss = self.MESLoss(self.gradient(fused), self.gradient(far_img)) + self.MESLoss(fused,
                                                                                                       near_img)
            loss += batch_loss.item()
        return loss

    def inference_ME(self, model, epoch, i=0):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (under_img, over_img, gt_img) in enumerate(self.testloader):
            under_img, over_img = under_img.to(self.device), over_img.to(self.device)

            # Inference
            fused = model.feature_fusion(torch.cat((under_img, over_img), 1))
            local_Detail = model.detail_net(torch.cat((under_img, over_img), 1))
            fused = local_Detail + fused
            #     # 保存目录
            save_dir = r"/media/data1/swj/FedFusion/test/result/"
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #
            #     # 保存图像
            sd, en = savenp(fused * 0.5 + 0.5, save_dir + f"fused_ME_{batch_idx}_{epoch}_{i}.png")
            print(sd, en)
            # save_image(fused_img, os.path.join(save_dir, f"fused_{batch_idx}_{epoch}_{i}.png"))
            # save_image(output1, os.path.join(save_dir, f"out1_{batch_idx}.png"))
            # save_image(output2, os.path.join(save_dir, f"out2_{batch_idx}.png"))

            batch_loss = self.MESLoss(self.gradient(fused), self.gradient(over_img)) + self.MESLoss(fused,
                                                                                                   under_img)
            loss += batch_loss.item()
        return loss



class ServerUpdate(object):
    def __init__(self, args, dataset, logger):
        self.args = args
        self.logger = logger
        self.trainloader = self.get_train_data(dataset)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        # Default criterion set to NLL loss function
        self.MseLoss = nn.MSELoss().to(self.device)
        self.L1Loss = nn.L1Loss().to(self.device)
        self.gradient = Gradient().to(self.device)

    def get_train_data(self, dataset):
        """
        Returns train dataloaders for a given dataset and user indexes.
        """

        trainloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8)
        print(len(trainloader))
        return trainloader

    def update_weights(self, model, global_round, state):
        # all char can be instead by enum
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                itertools.chain(model.fusion_encoder.parameters(), model.two_branch_decoder.parameters()),
                lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                itertools.chain(model.fusion_encoder.parameters(), model.two_branch_decoder.parameters()),
                lr=0.0001, betas=[0.5, 0.999])
        elif self.args.optimizer == 'rms':
            optimizer = torch.optim.RMSprop(
                itertools.chain(model.fusion_encoder.parameters(), model.two_branch_decoder.parameters()),
                lr=self.args.lr)

        if state == 'init':
            epoch = 3
        else:
            epoch = self.args.server_ep

        for iter in range(epoch):
            batch_loss = []
            for batch_idx, (natural_img, natural_img1) in enumerate(self.trainloader):
                natural_img, natural_img1 = natural_img.to(self.device), natural_img1.to(self.device)
                # natural_img1 = natural_img1.to(self.device)
                model.zero_grad()
                out, out1 = model.feature_extract(torch.cat((natural_img, natural_img1), 1))
                loss_out = self.MseLoss(out, natural_img) + 10 * self.L1Loss(self.gradient(out),
                                                                             self.gradient(natural_img))
                loss_out1 = self.MseLoss(out1, natural_img1) + 10 * self.L1Loss(self.gradient(out1),
                                                                                self.gradient(natural_img1))
                loss = loss_out + loss_out1
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print(
                        'Global Training| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, iter, batch_idx * len(natural_img),
                            len(self.trainloader.dataset),
                                                100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def get_weight(self, w, model, dataset):
        valloader = DataLoader(dataset, batch_size=1, shuffle=True)
        local_ratio = []
        info = 0
        total_info = 0  # 用于存储所有信息量的总和
        for i in range(len(w)):
            model.load_state_dict(w[i])
            for batch_idx, (natural_img, natural_img1) in enumerate(valloader):
                natural_img, natural_img1 = natural_img.to(self.device), natural_img1.to(self.device)
                result = model.local_fusion(torch.cat((natural_img, natural_img1), 1))
                info += torch.sum(self.gradient(result).pow(2)).data
            print(f"client:{i}, info:{info}")
            local_ratio.append(info)
            total_info += info  # 累加信息量
            info = 0
            # torch.cuda.empty_cache()
        # 归一化处理，使得权重之和为1
        normalized_ratio = [info / total_info for info in local_ratio]

        return normalized_ratio

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (visible_img, infrared_img) in enumerate(self.testloader):
            infrared_img, visible_img = infrared_img.to(self.device), visible_img.to(self.device)

            # Inference
            output = model(torch.cat((visible_img, infrared_img), 1))
            fused_img = unnormalize(output)
            #
            #     # 保存目录
            save_dir = r"/media/data1/swj/FedFusion/test/result/"
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #
            #     # 保存图像
            save_image(fused_img, os.path.join(save_dir, f"visible_{batch_idx}.png"))
            batch_loss = self.MseLoss(self.gradient(output), self.gradient(visible_img)) + self.MseLoss(output,
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
