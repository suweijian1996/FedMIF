#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import test_inference, LocalUpdate, ServerUpdate
from FedFusion_model import FedMIFModel_GN, FedMIFModel
from data.Dataloader import getDataset
from utils import get_dataset, average_weights, exp_details, getUserGroup, Sobel, select_top_weights, \
    average_weights_ratios


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset and user groups

    NaturalDataset = getDataset('natural')
    DoubleNaturalDataset = getDataset('doubleNatural')
    ValNaturalDataset = getDataset('valNatural')
    IVDataset = getDataset('IV')
    IVTestDataset = getDataset('IVTest')
    MedicalDataset = getDataset('Medical')
    MedicalTestDataset = getDataset('MedicalTest')
    MFDataset = getDataset('MFGT')
    MFTestDataset = getDataset('MFGTTest')
    MEDataset = getDataset('ME')
    METestDataset = getDataset('METest')
    MF_user_groups = getUserGroup(args, MFDataset)
    IV_user_groups = getUserGroup(args, IVDataset)
    Medical_user_group = getUserGroup(args, MedicalDataset)
    ME_user_groups = getUserGroup(args, MEDataset)

    if args.norm == 'None':
        global_model = FedMIFModel()
    elif args.norm == 'GN':
        global_model = FedMIFModel_GN()

    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    global_IV_weights_decoder = global_model.fusion_decoder.state_dict()
    global_Medical_weights_decoder = global_model.fusion_decoder.state_dict()
    global_MF_weights_decoder = global_model.fusion_decoder.state_dict()
    global_ME_weights_decoder = global_model.fusion_decoder.state_dict()

    # # ----------------------
    # # server side
    # # ----------------------
    # ============================= training global model =============================
    global_model.train()
    server_model = ServerUpdate(args=args, dataset=DoubleNaturalDataset, logger=logger)
    local_w, loss = server_model.update_weights(model=copy.deepcopy(global_model), global_round=0, state='train')
    global_model.load_state_dict(copy.deepcopy(local_w))
    torch.save(global_model.state_dict(), r'..\save\pretrain_global.pth')

    IV_detail_weights, Medical_detail_weights, MF_detail_weights, ME_detail_weights = [], [], [], []
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, encoder_weights, IV_decoder_weights, Medical_decoder_weights, MF_decoder_weights, \
         ME_decoder_weights = [], [], [], [], [], [], [],
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        # ----------------------
        # local side
        # ----------------------

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # ============================= training encoder =============================

        # IV training Encoder
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args,
                                      dataset=IVDataset,
                                      test_dataset=IVTestDataset,
                                      idxs=IV_user_groups[idx],
                                      logger=logger)
            local_encoder_w, loss = local_model.update_encoder_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)
            local_losses.append(copy.deepcopy(loss))
            encoder_weights.append(copy.deepcopy(local_encoder_w))

        # medical training encoder
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args,
                                      dataset=MedicalDataset,
                                      test_dataset=MedicalTestDataset,
                                      idxs=Medical_user_group[idx],
                                      logger=logger)
            local_encoder_w, loss = local_model.update_encoder_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)
            local_losses.append(copy.deepcopy(loss))
            encoder_weights.append(copy.deepcopy(local_encoder_w))

        # MF training encoder
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args,
                                      dataset=MFDataset,
                                      test_dataset=MFTestDataset,
                                      idxs=MF_user_groups[idx],
                                      logger=logger,
                                      isME=True)
            local_encoder_w, loss = local_model.update_encoder_weights_MF(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)
            local_losses.append(copy.deepcopy(loss))
            encoder_weights.append(copy.deepcopy(local_encoder_w))

        # ME training encoder
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args,
                                      dataset=MEDataset,
                                      test_dataset=METestDataset,
                                      idxs=ME_user_groups[idx],
                                      logger=logger,
                                      isME=True)
            local_encoder_w, loss = local_model.update_encoder_weights_ME(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)
            local_losses.append(copy.deepcopy(loss))
            encoder_weights.append(copy.deepcopy(local_encoder_w))
        print(len(encoder_weights))
        global_weights_encoder = average_weights(encoder_weights)
        global_model.fusion_encoder.load_state_dict(global_weights_encoder)

        torch.save(global_weights_encoder, r'..\save\global_encoder.pth')
        # ===================== training fusion decoder =============================

        # IV Task
        global_model.fusion_decoder.load_state_dict(global_IV_weights_decoder)
        global_model.train()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args,
                                      dataset=IVDataset,
                                      test_dataset=IVTestDataset,
                                      idxs=IV_user_groups[idx],
                                      logger=logger)
            local_decoder_w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)

            local_losses.append(copy.deepcopy(loss))
            IV_decoder_weights.append(copy.deepcopy(local_decoder_w))
        global_IV_weights_decoder = average_weights(IV_decoder_weights)
        torch.save(global_IV_weights_decoder, r'..\save\global_IV_decoder.pth')

        # Medical Task
        global_model.fusion_decoder.load_state_dict(global_Medical_weights_decoder)
        global_model.train()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args,
                                      dataset=MedicalDataset,
                                      test_dataset=MedicalTestDataset,
                                      idxs=Medical_user_group[idx],
                                      logger=logger)
            local_decoder_w, loss = local_model.update_weights_medical(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)

            local_losses.append(copy.deepcopy(loss))
            Medical_decoder_weights.append(copy.deepcopy(local_decoder_w))

        global_Medical_weights_decoder = average_weights(Medical_decoder_weights)
        torch.save(global_Medical_weights_decoder, r'..\save\global_Medical_decoder.pth')

        # MF Task
        global_model.fusion_decoder.load_state_dict(global_MF_weights_decoder)
        global_model.train()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args,
                                      dataset=MFDataset,
                                      test_dataset=MFTestDataset,
                                      idxs=MF_user_groups[idx],
                                      logger=logger,
                                      isME=True)
            local_decoder_w, loss = local_model.update_MF_decoder_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)

            local_losses.append(copy.deepcopy(loss))
            MF_decoder_weights.append(copy.deepcopy(local_decoder_w))

        global_MF_weights_decoder = average_weights(MF_decoder_weights)
        torch.save(global_MF_weights_decoder, r'..\save\global_MF_decoder.pth')

        # ME Task
        global_model.fusion_decoder.load_state_dict(global_ME_weights_decoder)
        global_model.train()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args,
                                      dataset=MEDataset,
                                      test_dataset=METestDataset,
                                      idxs=ME_user_groups[idx],
                                      logger=logger,
                                      isME=True)
            local_decoder_w, loss = local_model.update_ME_decoder_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)

            local_losses.append(copy.deepcopy(loss))
            ME_decoder_weights.append(copy.deepcopy(local_decoder_w))
        global_ME_weights_decoder = average_weights(ME_decoder_weights)
        torch.save(global_ME_weights_decoder, r'..\save\global_ME_decoder.pth')

        # ============================= training Adapter =============================
        # IV detail
        global_model.train()
        global_model.fusion_decoder.load_state_dict(global_IV_weights_decoder)
        for idx in range(args.num_users):
            if epoch != 0:
                global_model.detail_net.load_state_dict(IV_detail_weights[idx])
            local_model = LocalUpdate(args=args,
                                      dataset=IVDataset,
                                      test_dataset=IVTestDataset,
                                      idxs=IV_user_groups[idx],
                                      logger=logger)
            local_w, fusion_detail_w, loss = local_model.update_detail_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)
            if epoch == 0:
                IV_detail_weights.append(copy.deepcopy(fusion_detail_w))
            else:
                IV_detail_weights[idx] = copy.deepcopy(fusion_detail_w)


        # Medical detail
        global_model.train()
        global_model.fusion_decoder.load_state_dict(global_Medical_weights_decoder)
        for idx in range(args.num_users):
            if epoch != 0:
                global_model.detail_net.load_state_dict(Medical_detail_weights[idx])
            local_model = LocalUpdate(args=args,
                                      dataset=MedicalDataset,
                                      test_dataset=MedicalTestDataset,
                                      idxs=Medical_user_group[idx],
                                      logger=logger)
            local_w, fusion_detail_w, loss = local_model.update_detail_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)
            if epoch == 0:
                Medical_detail_weights.append(copy.deepcopy(fusion_detail_w))
            else:
                Medical_detail_weights[idx] = copy.deepcopy(fusion_detail_w)

        # MF detail
        global_model.train()
        global_model.fusion_decoder.load_state_dict(global_MF_weights_decoder)
        for idx in range(args.num_users):
            if epoch != 0:
                global_model.detail_net.load_state_dict(MF_detail_weights[idx])
            local_model = LocalUpdate(args=args,
                                      dataset=MFDataset,
                                      test_dataset=MFTestDataset,
                                      idxs=MF_user_groups[idx],
                                      logger=logger,
                                      isME=True)
            local_w, fusion_detail_w, loss = local_model.update_detail_mf_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)
            if epoch == 0:
                MF_detail_weights.append(copy.deepcopy(fusion_detail_w))
            else:
                MF_detail_weights[idx] = copy.deepcopy(fusion_detail_w)

        # ME detail
        global_model.train()
        global_model.fusion_decoder.load_state_dict(global_ME_weights_decoder)
        for idx in range(args.num_users):
            if epoch != 0:
                global_model.detail_net.load_state_dict(ME_detail_weights[idx])
            local_model = LocalUpdate(args=args,
                                      dataset=MEDataset,
                                      test_dataset=METestDataset,
                                      idxs=ME_user_groups[idx],
                                      logger=logger,
                                      isME=True)
            local_w, fusion_detail_w, loss = local_model.update_detail_me_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)
            if epoch == 0:
                ME_detail_weights.append(copy.deepcopy(fusion_detail_w))
            else:
                ME_detail_weights[idx] = copy.deepcopy(fusion_detail_w)

        # save model
        torch.save(global_model.state_dict(), r'..\save\global.pth')

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.fusion_decoder.load_state_dict(global_IV_weights_decoder)
        global_model.eval()
        for i in range(len(IV_detail_weights)):
            torch.save(IV_detail_weights[i], rf'..\save\global_detail_IV_{i}.pth')
            global_model.detail_net.load_state_dict(IV_detail_weights[i])
            local_model = LocalUpdate(args=args, dataset=IVDataset, test_dataset=IVTestDataset,
                                      idxs=IV_user_groups[i], logger=logger)
            loss = local_model.inference(model=global_model, epoch=epoch, i=i)

        global_model.fusion_decoder.load_state_dict(global_Medical_weights_decoder)
        for i in range(len(Medical_detail_weights)):
            torch.save(Medical_detail_weights[i], rf'..\save\global_detail_medical_{i}.pth')
            global_model.detail_net.load_state_dict(Medical_detail_weights[i])
            local_model = LocalUpdate(args=args, dataset=MedicalDataset, test_dataset=MedicalTestDataset,
                                      idxs=Medical_user_group[i], logger=logger)
            loss = local_model.inference_Medical(model=global_model, epoch=epoch, i=i)

        global_model.fusion_decoder.load_state_dict(global_MF_weights_decoder)
        for i in range(len(MF_detail_weights)):
            torch.save(MF_detail_weights[i], rf'..\save\global_detail_MF_{i}.pth')
            global_model.detail_net.load_state_dict(MF_detail_weights[i])
            local_model = LocalUpdate(args=args, dataset=MFDataset, test_dataset=MFTestDataset,
                                      idxs=Medical_user_group[i], logger=logger)
            loss = local_model.inference_MF(model=global_model, epoch=epoch, i=i)

        global_model.fusion_decoder.load_state_dict(global_ME_weights_decoder)
        for i in range(len(ME_detail_weights)):
            torch.save(ME_detail_weights[i], rf'..\save\global_detail_ME_{i}.pth')
            global_model.detail_net.load_state_dict(ME_detail_weights[i])
            local_model = LocalUpdate(args=args, dataset=MEDataset, test_dataset=METestDataset,
                                      idxs=ME_user_groups[i], logger=logger)
            loss = local_model.inference_ME(model=global_model, epoch=epoch, i=i)

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
