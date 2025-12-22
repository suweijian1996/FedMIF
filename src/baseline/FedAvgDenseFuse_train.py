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

from src.options import args_parser
# from update import test_inference, LocalUpdate, ServerUpdate
from DenseFuse_Model import DenseFuse_net
from FedAvgDenseFuse_update import LocalUpdate
from data.Dataloader import getDataset
from src.utils import get_dataset, average_weights, exp_details, getUserGroup

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

    NaturalDataset = getDataset('natural')
    IVTestDataset = getDataset('IVTest')
    user_groups = getUserGroup(args, NaturalDataset)

    global_model = DenseFuse_net()
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

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        # ----------------------
        # local side
        # ----------------------

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # TODO training DenseFuse
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args,
                                      dataset=NaturalDataset,
                                      test_dataset=IVTestDataset,
                                      idxs=user_groups[idx],
                                      logger=logger)
            local_w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                idx=idx)
            local_losses.append(copy.deepcopy(loss))
            local_weights.append(copy.deepcopy(local_w))
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # save model
        torch.save(global_model.state_dict(), r'.\save\FedAvgDenseFuse_global.pth')

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for i in range(len(local_weights)):
            torch.save(local_weights[i], rf'.\save\FedAvgDenseFuse_local_{i}.pth')
            global_model.load_state_dict(local_weights[i])
            local_model = LocalUpdate(args=args, dataset=NaturalDataset, test_dataset=IVTestDataset,
                                      idxs=user_groups[i], logger=logger)
            loss = local_model.inference(model=global_model, epoch=epoch, i=i)

        # train_accuracy.append(sum(list_acc) / len(list_acc))
        #
        # # print global training loss after every 'i' rounds
        # if (epoch + 1) % print_every == 0:
        #     print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        #     print(f'Training Loss : {np.mean(np.array(train_loss))}')
        #     print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    # Test inference after completion of training
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)

    # print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
