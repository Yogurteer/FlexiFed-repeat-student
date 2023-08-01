#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding=utf-8
# Python version: 3.8
import os
import copy
import time
import pickle
from itertools import groupby
import numpy as np
import torch
import time
# from vgg import *
# from utils.utils import *
# from utils import *
from zzp import *
from vgg import *
from torch.utils.tensorboard import SummaryWriter
import warnings
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")
"""
basic-common on Speech_Common task epcoh1501 by 周臻鹏
输出文件为：results.txt，local_acc.txt，table.txt
"""

if __name__ == '__main__':
    """
    在这里编辑模型名，数据集，聚合策略，epcoh总数，父目录
    """
    # strategys = ["Basic-Common", "Clustered-Common", "Max-Common"]
    # models = ["VGG", "ResNet"]
    # datanames = ["CIFAR-10", "CINIC-10", "Speech-Command"]
    # datasetname1 = datanames[1]
    dad_dir = "/root/autodl-tmp/0725zzp03rtx3080/0731-clustered-new_speech-vgg-1-epoch502"
    sdir = dad_dir.split('/')
    totaldata = sdir[-1].split('-')
    nepoch = [''.join(list(g)) for k, g in groupby(totaldata[5], key=lambda x: x.isdigit())]
    epochs = int(nepoch[1])
    modelname, datasetname, strategyname = totaldata[3], totaldata[2], totaldata[1]
    lr = 0.001  # 学习率，根据数据集调整
    print("training started:\n{}\n{}\n{}\nepoch={}".format(modelname, datasetname, strategyname, epochs))
    l_time = time.time()
    start_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("start time:{}".format(start_time))
    fname1 = "{}/results.txt".format(dad_dir)
    fname2 = "{}/local_acc.txt".format(dad_dir)
    fname3 = "{}/table.txt".format(dad_dir)
    print("输出目录为：{}".format(dad_dir))

    file = open(fname1, 'w').close()
    file = open(fname2, 'w').close()
    file = open(fname3, 'w').close()
    device = torch.device("cuda:0")

    # load dataset and user groups
    train_dataset, test_dataset, user_groups, idx_test = get_dataset(datasetname)
    print("len train:{}\nlen test:{}".format(len(train_dataset),len(test_dataset)))
    print("shape of each data:{}\ntype:{}".format(train_dataset[0][0].shape,train_dataset[0][0].dtype))
    # print(train_dataset[0][0])

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    number_device = 8
    idxs_users = [_id for _id in range(number_device)]
    modelAccept = {_id: None for _id in range(number_device)}

    for _id in range(number_device):
        if _id < 2:
            modelAccept[_id] = vgg11_bn()

        elif _id >= 2 and _id < 4:
            modelAccept[_id] = vgg13_bn()

        elif _id >= 4 and _id < 6:
            modelAccept[_id] = vgg16_bn()

        else:
            modelAccept[_id] = vgg19_bn()

    localData_length = len(user_groups[0]) / 10
    start = 0

    local_acc = [[] for i in range(number_device)]

    writer = SummaryWriter('/root/tf-logs')
    for epoch in range(epochs):

        print(f'\n | Global Training Round : {epoch + 1} |\n')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        with open(fname1, 'a') as f:
            print(f'\n | Global Training Round : {epoch + 1} |\n', file=f)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=f)
        end = start + localData_length

        for idx in idxs_users:

            idx_train_all = list(user_groups[idx])
            idx_train_batch = set(idx_train_all[int(start):int(end)])

            if epoch == 0:
                model = modelAccept[idx]

            if epoch > 0:
                if idx < 2:
                    model = vgg11_bn()
                    # model.load_state_dict(A)
                elif idx >= 2 and idx < 4:
                    model = vgg13_bn()
                    # model.load_state_dict(B)

                elif idx >= 4 and idx < 6:
                    model = vgg16_bn()
                    # model.load_state_dict(C)

                else:
                    model = vgg19_bn()
                    # model.load_state_dict(D)

                model.load_state_dict(modelAccept[idx], strict=False)

            # print(type(model),"here",type(test_dataset))
            acc = test_inference(model, test_dataset, list(idx_test[idx]), device)
            local_acc[idx].append(round(acc, 4))
            if epoch % 10 == 0:
                print("idxs_users:", idx, "local_accuracy latest:", local_acc[idx][epoch], "epoch:", epoch)

                with open(fname1, 'a') as f:
                    print("idxs_users:", idx, "local_accuracy latest:", local_acc[idx][epoch], "epoch:", epoch, file=f)
                with open(fname2, 'a') as ff:
                    print("idxs_users:", idx, "local_accuracy latest:", local_acc[idx][epoch], "epoch:", epoch, file=ff)

            Model = copy.deepcopy(model)
            localModel = local_train(Model, train_dataset, idx_train_batch, device,lr=lr)
            modelAccept[idx] = copy.deepcopy(localModel)
        for i in range(0,8,2): # 将acc实时结果写入tensorboard，acc目录下，四个线图
            writer.add_scalar('acc/model:{}--{}'.format(modelname,i/2+1), local_acc[i][-1], epoch)
        start = end % 2500

        # modelAccept = common_max(modelAccept)
        modelAccept = common_basic(modelAccept)
        # modelAccept = common_clustered(modelAccept)
        with open(fname3, 'w') as fff:
            print(str(local_acc), file=fff)

    print("训练模型完成，输出文件为：results.txt，local_acc.txt，table.txt")
    r_time = time.time()
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    consuming_time=round(r_time-l_time,2)
    print("start time:{}\nending time:{}\nconsuming time:{}s".format(start_time, end_time, consuming_time))
    writer.close()
