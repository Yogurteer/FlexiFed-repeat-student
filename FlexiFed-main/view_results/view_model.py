#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding=utf-8
# Python version: 3.8
from utils import *
from model_family.VDCNN import *
from torch.utils.tensorboard import SummaryWriter
import warnings
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")

model1 = vdcnn9()
model2 = vdcnn17()
model3 = vdcnn29()
train_dataset, test_dataset, user_groups, idx_test = get_dataset("agnews", 8, "vdcnn")
print(model1)
print(model2)
print(model3)
writer = SummaryWriter('/root/tf-logs')
x = train_dataset[0][0]
x = x.unsqueeze(0)
writer.add_graph(model2, x)
writer.close()
