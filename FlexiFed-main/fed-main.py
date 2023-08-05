#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding=utf-8
# Python version: 3.8
from itertools import groupby
from utils import *
from model_family.vgg import *
from model_family.ResNet import *
from model_family.CharCNN import *
from model_family.VDCNN import *
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
    在这里编辑训练文件目录
    """
    # strategys = ["Basic-Common", "Clustered-Common", "Max-Common"]
    # models = ["VGG", "ResNet"]
    # datanames = ["CIFAR-10", "CINIC-10", "Speech-Command"]
    # datasetname1 = datanames[1]
    # 通过目录名获取日期,策略名,数据集名,模型名,训练标签,训练轮数
    dad_dir = "/root/autodl-tmp/0725zzp03rtx3080/results_all/0804-clustered-agnews-vdcnn-fake810-epoch100"
    sdir = dad_dir.split('/')
    server_name = sdir[-3]
    totaldata = sdir[-1].split('-')
    nepoch = [''.join(list(g)) for k, g in groupby(totaldata[5], key=lambda x: x.isdigit())]
    epochs = int(nepoch[1])
    train_date = totaldata[0]
    trainlabel, modelname, datasetname, strategyname = totaldata[4], totaldata[3], totaldata[2], totaldata[1]
    lr = 0.01  # 学习率,根据数据集调整,目前来看speech_commands设0.001,其余设0.01
    batch_size = 64  # batch_size,影响显存使用大小，越大，训练周期，越不易收敛
    number_clients = 8  # client个数
    inputchannels = 3  # 输入样本的C
    classes_dict = {'cifar10': 10, 'cinic10': 10, 'new_speech': 30, 'agnews': 4}
    numclasses = classes_dict[datasetname]  # 输出维度
    fname1 = "{}/results.txt".format(dad_dir)
    fname3 = "{}/table.txt".format(dad_dir)
    print("输出目录为：{}".format(dad_dir))
    print("--------" * 8)
    file1 = open(fname1, 'w').close()
    file3 = open(fname3, 'w').close()
    device = torch.device("cuda:0")
    # load dataset and user groups 加载数据集和每个client划分的数据分区
    opti_list = [8, 20, 40]
    opti_number_clients = opti_list[0]
    train_dataset, test_dataset, user_groups, idx_test = get_dataset(datasetname, opti_number_clients, modelname)
    localData_length = int(len(user_groups[0]) / 10)
    print("len train:{} len test:{} shape data:{}\n"
          "client batch size:{} local train batch size:{}\n"
          "in_channels:{} out_classes:{}"
          .format(len(train_dataset), len(test_dataset), train_dataset[0][0].shape,
                  len(user_groups[0]), localData_length,
                  inputchannels, numclasses))
    print("--------" * 8)
    # Training
    train_loss, train_accuracy = [], []  # 无用current
    idxs_users = [_id for _id in range(number_clients)]  # clients下标集合
    modelAccept = {_id: None for _id in range(number_clients)}  # modelAccept[]表示sever传给client的模型中间量
    # 初始化modelAccept
    if modelname == "vgg":
        for _id in range(number_clients):
            if _id < 2:
                modelAccept[_id] = vgg11_bn()
            elif 2 <= _id < 4:
                modelAccept[_id] = vgg13_bn()
            elif 4 <= _id < 6:
                modelAccept[_id] = vgg16_bn()
            else:
                modelAccept[_id] = vgg19_bn()
    elif modelname == "resnet":
        for _id in range(number_clients):
            if _id < 2:
                modelAccept[_id] = resnet20(inputchannels, numclasses)
            elif 2 <= _id < 4:
                modelAccept[_id] = resnet32(inputchannels, numclasses)
            elif 4 <= _id < 6:
                modelAccept[_id] = resnet44(inputchannels, numclasses)
            else:
                modelAccept[_id] = resnet56(inputchannels, numclasses)
    elif modelname == "charcnn":
        for _id in range(number_clients):
            if _id < 2:
                modelAccept[_id] = charcnn3()
            elif 2 <= _id < 4:
                modelAccept[_id] = charcnn4()
            elif 4 <= _id < 6:
                modelAccept[_id] = charcnn5()
            else:
                modelAccept[_id] = charcnn6()
    elif modelname == "vdcnn":
        for _id in range(number_clients):
            if _id < 2:
                modelAccept[_id] = vdcnn9()
            elif 2 <= _id < 4:
                modelAccept[_id] = vdcnn17()
            elif 4 <= _id < 6:
                modelAccept[_id] = vdcnn29()
            else:
                modelAccept[_id] = vdcnn49()
    start = 0  # start-end 表示每次local training时一个client训练样本个数，在cifar下为625，即localData_length
    local_acc = [[] for i in range(number_clients)]
    local_loss = [[] for i in range(number_clients)]
    writer = SummaryWriter('/root/tf-logs')
    """
    开始FlexiFed全局训练
    """
    print("training started:\n"
          "server name:{}\n"
          "model family:{}\n"
          "dataset:{} \n"
          "strategy:{}\n"
          "lr:{} batch_size:{}\n"
          "communication rounds:{}\n"
          "train label:{}"
          .format(server_name, modelname, datasetname, strategyname, lr, batch_size, epochs, trainlabel))
    l_time = time.time()
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("start time:{}".format(start_time))
    using_time = [0]
    for epoch in range(0, epochs + 1, 1):
        cur_epoch_time = time.time()
        using_time.append(cur_epoch_time)
        if epoch != 0:
            print("Last epoch use time:{}s".format(round(using_time[-1] - using_time[-2]), 2))
        print(f'| Global Training Round : {epoch + 1} |\n')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        with open(fname1, 'a') as f:
            print(f'\n | Global Training Round : {epoch + 1} |\n', file=f)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), file=f)
        end = start + localData_length
        for idx in idxs_users:
            idx_train_all = list(user_groups[idx])
            idx_train_batch = set(idx_train_all[int(start):int(end)])
            # model初始化结构,从modelAccept中加载参数
            if modelname == "vgg":
                if epoch == 0:
                    model = modelAccept[idx]
                if epoch > 0:
                    if idx < 2:
                        model = vgg11_bn()
                    elif idx >= 2 and idx < 4:
                        model = vgg13_bn()
                    elif idx >= 4 and idx < 6:
                        model = vgg16_bn()
                    else:
                        model = vgg19_bn()
                    model.load_state_dict(modelAccept[idx], strict=False)
            elif modelname == "resnet":
                if epoch == 0:
                    model = modelAccept[idx]
                if epoch > 0:
                    if idx < 2:
                        model = resnet20(inputchannels, numclasses)
                    elif 2 <= idx < 4:
                        model = resnet32(inputchannels, numclasses)
                    elif 4 <= idx < 6:
                        model = resnet44(inputchannels, numclasses)
                    else:
                        model = resnet56(inputchannels, numclasses)
                    model.load_state_dict(modelAccept[idx], strict=False)
            elif modelname == "charcnn":
                if epoch == 0:
                    model = modelAccept[idx]
                if epoch > 0:
                    if idx < 2:
                        model = charcnn3()
                    elif 2 <= idx < 4:
                        model = charcnn4()
                    elif 4 <= idx < 6:
                        model = charcnn5()
                    else:
                        model = charcnn6()
                    model.load_state_dict(modelAccept[idx], strict=False)
            elif modelname == "vdcnn":
                if epoch == 0:
                    model = modelAccept[idx]
                if epoch > 0:
                    if idx < 2:
                        model = vdcnn9()
                    elif 2 <= idx < 4:
                        model = vdcnn17()
                    elif 4 <= idx < 6:
                        model = vdcnn29()
                    else:
                        model = vdcnn49()
                    model.load_state_dict(modelAccept[idx], strict=False)
            # 验证当前model的准确性
            acc = test_inference(model, test_dataset, list(idx_test[idx]), device)
            # acc写入local_acc[]
            local_acc[idx].append(round(acc, 4))
            # 写入local_acc[]到文件result
            # if epoch % 10 == 0:
            with open(fname1, 'a') as f:
                print("idxs_client:", idx,
                      "local_accuracy latest:", local_acc[idx][epoch], "epoch:", epoch, file=f)
            # Model深拷贝自model,训练Model,每个epoch迭代10次,存入localModel,modelAccept[idx]深拷贝自localModel
            Model = copy.deepcopy(model)
            localModel = local_train(idx, Model, train_dataset, idx_train_batch,
                                     device, lr=lr, epoch=epoch, local_loss=local_loss)
            # print("epoch {} id {} loss:{}".format(epoch, idx, local_loss[idx]))
            modelAccept[idx] = copy.deepcopy(localModel)
        # 将acc和loss实时结果写入tensorboard，acc目录下，四个线图
        for i in range(0, 8, 2):
            writer.add_scalar(
                'acc--{}-{}-{}-{}-{}-epoch{}/version--{}'.format(train_date,
                                                                 trainlabel,
                                                                 datasetname,
                                                                 strategyname,
                                                                 modelname,
                                                                 epochs,
                                                                 i / 2 + 1),
                local_acc[i][-1],  # 纵坐标
                epoch  # 横坐标
            )
        for i in range(0, 8, 2):
            writer.add_scalar(
                'loss--{}-{}-{}-{}-{}-epoch{}/version--{}'.format(train_date,
                                                                  trainlabel,
                                                                  datasetname,
                                                                  strategyname,
                                                                  modelname,
                                                                  epochs,
                                                                  i / 2 + 1),
                local_loss[i][-1],  # 纵坐标
                epoch  # 横坐标
            )
        # 如果当前client batch跑完,重新循环
        start = end % int(len(user_groups[0]))
        # 全局模型聚合,每个epoch一次
        if strategyname == "basic":
            modelAccept = common_basic(modelAccept)
        elif strategyname == "clustered":
            modelAccept = common_clustered(modelAccept)
        elif strategyname == "max":
            modelAccept = common_max(modelAccept)
        elif strategyname == "alone":
            continue
        # 将local_acc写入文件table
        with open(fname3, 'w') as fff:
            print(str(local_acc), file=fff)
    print("训练模型完成，输出文件为：results.txt，local_acc.txt，table.txt")
    r_time = time.time()
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    consuming_time = round(r_time - l_time, 2)
    # 打印log记录
    print("training started:\n"
          "server name:{}\n"
          "model family:{}\n"
          "dataset:{} \n"
          "strategy:{}\n"
          "lr:{} batch_size:{}\n"
          "communication rounds:{}\n"
          "train label:{}"
          .format(server_name, modelname, datasetname, strategyname, lr, batch_size, epochs, trainlabel))
    print("start time:{}\nending time:{}\nconsuming time:{}s".format(start_time, end_time, consuming_time))
    writer.close()
