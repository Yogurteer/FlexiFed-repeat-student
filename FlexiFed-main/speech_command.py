from typing import List, Union
from numpy import *
import matplotlib.pyplot as plt  # 导入画图工具库
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import torch.nn.functional as F
import copy
from collections import defaultdict
import numpy as np
import copy
import math
from torchvision import datasets, transforms
import os
import deeplake
import fnmatch
import librosa

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# DatasetSplit 类继承自父类 Dataset
class DatasetSplit1(Dataset):  # DatasetSplit 类继承自父类 Dataset
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # image = self.dataset[self.idxs[item]]
        return image, label


def cifar_iid1(dataset, num_users):
    """
    主实验中默认使用iid划分方式，文章中写40个用户随机分配40个独立同分布的分区，种类打乱
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index

    实现方法：
    dict_users字典中的每一个元素是一个set集合,set不可以取重复元素
    all_idxs初始化为0-样本总数的list
    num_items表示每个分区中的样本个数
    已知有num_users个分区
    每次分区时,从当前的all_idxs中随机取num_items个数表示已加入的样本序号,序号不可重复
    分区加入序号之后,all_idxs列表中删除当前分区中的num_items个元素,表示分完当前区后剩下的总样本
    """
    num_items = int(len(dataset) / num_users)  # 每个分区的样本个数
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def local_train1(net, dataset, idxs, device):
    net.to(device)
    net.train()

    # train and update
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # SGD 随机梯度下降
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    ldr_train = DataLoader(DatasetSplit1(dataset, list(idxs)), batch_size=64, shuffle=True)
    for iter in range(10):
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(device), labels.to(device)
            net.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return net.state_dict()


def get_dataset1(dataset_name):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if dataset_name == "CINIC-10":
        cinic_directory = '/root/CINIC-10'
        common_mean, common_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # Imagenet的均值和标准差
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        # 推测：CINIC-10上计算出的均值和标准差
        train_dataset = datasets.ImageFolder(cinic_directory + '/train',
                                             transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                                           transforms.RandomHorizontalFlip(),
                                                                           transforms.ToTensor(),
                                                                           transforms.Normalize(mean=cinic_mean,
                                                                                                std=cinic_std)]))

        # print("CINIC-dataset label[0]:\n{}".format(train_dataset[0][1]))
        # print("len of set:{}".format(len(train_dataset)))
        # print("CINIC tensor dtype:{}".format(train_dataset[0][0].dtype))
        test_dataset = datasets.ImageFolder(cinic_directory + '/test',
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize(mean=cinic_mean,
                                                                                               std=cinic_std)]))

        user_groups = cifar_iid1(train_dataset, 20)  # 存储每个分区的样本序号的set组成的dict
        user_groups_test = cifar_iid1(test_dataset, 20)  # 存储每个分区的样本序号的set组成的dict

        return train_dataset, test_dataset, user_groups, user_groups_test

    if dataset_name == "Speech-Command":
        speech_directory = "/root/speech_command"
        train_dataset = Audio_DataLoader(speech_directory + '/train', sr=16000, dimension=8192)
        test_dataset = Audio_DataLoader(speech_directory + '/test', sr=16000, dimension=8192)
        user_groups = cifar_iid1(train_dataset, 20)  # 存储每个分区的样本序号的set组成的dict
        user_groups_test = cifar_iid1(test_dataset, 20)  # 存储每个分区的样本序号的set组成的dict

        return train_dataset, test_dataset, user_groups, user_groups_test


class Audio_DataLoader(Dataset):
    wav_list: List[Union[bytes, str]]

    def __init__(self, data_folder, sr=16000, dimension=8192):
        self.data_folder = data_folder
        self.sr = sr
        self.dim = dimension
        self.labellist = []
        # 获取音频名列表
        self.wav_list = []
        for root, dirnames, filenames in os.walk(data_folder):
            for filename in fnmatch.filter(filenames, "*.wav"):  # 实现列表特殊字符的过滤或筛选,返回符合匹配“.wav”字符列表
                self.wav_list.append(os.path.join(root, filename))
        # print("len of wav list:{}\n{}".format(len(self.wav_list),self.wav_list[0]))

    def __getitem__(self, item):
        # 读取一个音频文件，返回每个音频数据
        filename = self.wav_list[item]
        # print("cur label:{}".format(filename))
        # print(filename)
        wb_wav, _ = librosa.load(filename, sr=self.sr)
        # sr为采样率，通过KMplayer查看sampling rate，确认过speech commands为16000

        # 取 帧
        if len(wb_wav) > self.dim:  # self.dim=8196
            # print("yes:len of wb_wav{}:{}".format(filename, len(wb_wav)))
            max_audio_start = len(wb_wav) - self.dim
            audio_start = np.random.randint(0, max_audio_start)
            wb_wav = wb_wav[audio_start: audio_start + self.dim]
        else:
            wb_wav = np.pad(wb_wav, (0, self.dim - len(wb_wav)), "constant")
        wb_wav.dtype = np.float32
        wav = torch.tensor(wb_wav)
        wav = wav.view(-1, 32, 32)
        # wav = wav.mul(1000)
        # mean = -0.0756
        # std = 95.2344
        # """
        # 先标准化
        # mean tensor(-0.0756)
        # std tensor(95.2344)
        # """
        # wav = wav.add(mean)
        # wav = wav.div(std)
        # """
        # 后归一化
        # """
        # min = -10.501201629638672
        # max = 10.499293327331543
        # dif = max-min
        # wav = wav.add(0-min)
        # wav = wav.div(dif)
        label_d = filename
        labels_d = label_d.split('/')
        label = labels_d[-2]
        # label = int(1)
        label_dict = {
            'bed': 0,
            'bird': 1,
            'cat': 2,
            'dog': 3,
            'down': 4,
            'eight': 5,
            'five': 6,
            'four': 7,
            'go': 8,
            'happy': 9,
            'house': 10,
            'left': 11,
            'marvin': 12,
            'nine': 13,
            'no': 14,
            'off': 15,
            'on': 16,
            'one': 17,
            'right': 18,
            'seven': 19,
            'sheila': 20,
            'six': 21,
            'stop': 22,
            'three': 23,
            'tree': 24,
            'two': 25,
            'up': 26,
            'wow': 27,
            'yes': 28,
            'zero': 29,
            '_background_noise_': 30
        }
        numlabel = label_dict[label]
        return wav, numlabel

    def __len__(self):
        # 音频文件的总数
        return len(self.wav_list)


class ZZP_Data(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):  # 默认是item，但常改为idx，是index的缩写
        pass


train_dataset1, test_dataset1, user_groups1, idx_test1 = get_dataset1("Speech-Command")
a = train_dataset1
# a, b, c, d = get_dataset1("Speech-Command")

print("len:", len(a))
print("shape:", a[0][0].shape)
# for i in range(0,64727,200):
#     print("label", a[i][1], type(a[i][1]))
print(a[0][0])
data = a[0][0].flatten()
x1 = range(len(data))
ax1 = plt
ax1.hist(data, bins=50, range=(-0.001,0.001))
# ax1.xtricks(range(-0.1,0.1,20))
# x,bins = 50,range=(0,55)
plt.tight_layout()  # 自动调整各子图间距
plt.show()


# print(a[0])
# print(type(a))

def get_mean_std(loader):
    # Var[x] = E[X**2]-E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    print(num_batches)
    print(channels_sum)
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def get_min_max(loader):
    min, max = 10000, -10000

    for data, _ in loader:
        cmin = torch.min(data)
        if cmin < min:
            min = cmin
        cmax = torch.max(data)
        if cmax > max:
            max = cmax
    return min, max
