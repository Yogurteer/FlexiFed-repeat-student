import csv
import time

import h5py
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
import fnmatch
import librosa

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def cifar_iid(dataset, num_users):
    """
    主实验中默认使用iid划分方式，文章中写40个用户随机分配40个独立同分布的分区，种类打乱
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)  # num_items表示每个client分得的样本个数
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))  # 从all_idxs中随机抽取num_items个元素,不可重复
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 40, 1250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def get_dataset(dataset_name: object, number_clients, model_name) -> object:
    """
    主实验中默认使用iid划分方式，文章中写40个用户随机分配40个独立同分布的分区，种类打乱
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: train,test,train groups,test groups

    实现方法：
    dict_users字典中的每一个元素是一个set集合,set不可以取重复元素
    all_idxs初始化为0-样本总数的list
    num_items表示每个分区中的样本个数
    已知有num_users个分区
    每次分区时,从当前的all_idxs中随机取num_items个数表示已加入的样本序号,序号不可重复
    分区加入序号之后,all_idxs列表中删除当前分区中的num_items个元素,表示分完当前区后剩下的总样本
    """
    if dataset_name == "cifar10":
        path = "/root/autodl-tmp/CIFAR-10"
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                # normalize:the param is the mean value and standard deviation of the three channel
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),
        }

        train_dataset = datasets.CIFAR10(path + "/train", train=True, download=False,
                                         transform=data_transforms['train'])
        test_dataset = datasets.CIFAR10(path + "/test", train=False, download=False,
                                        transform=data_transforms['test'])
        user_groups = cifar_iid(train_dataset, 20)
        user_groups_test = cifar_iid(test_dataset, 20)

        return train_dataset, test_dataset, user_groups, user_groups_test
    if dataset_name == "cinic10":
        cinic_directory = '/root/autodl-tmp/CINIC-10'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        train_dataset = datasets.ImageFolder(cinic_directory + '/train',
                                             transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                                           transforms.RandomHorizontalFlip(),
                                                                           transforms.ToTensor(),
                                                                           transforms.Normalize(mean=cinic_mean,
                                                                                                std=cinic_std)]))
        test_dataset = datasets.ImageFolder(cinic_directory + '/test',
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize(mean=cinic_mean,
                                                                                               std=cinic_std)]))

        user_groups = cifar_iid(train_dataset, 40)
        user_groups_test = cifar_iid(test_dataset, 40)

        return train_dataset, test_dataset, user_groups, user_groups_test
    if dataset_name == "speech_commands":
        speech_directory = "/root/autodl-tmp/speech_commands"
        dataset = Aduio_DataLoader(speech_directory + '/train', sr=16000, dimension=16000)
        # print(dataset.wav_list)
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        # 随机划分数据集和测试集
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        user_groups = cifar_iid(train_dataset, number_clients)  # 存储每个分区的样本序号的set组成的dict
        user_groups_test = cifar_iid(test_dataset, number_clients)  # 存储每个分区的样本序号的set组成的dict

        return train_dataset, test_dataset, user_groups, user_groups_test
    if dataset_name == "new_speech":  # 目标h5文件下存放有预处理过后的数据
        speech_directory = "/root/autodl-tmp/{}".format("zzph5-1.h5")
        dataset = h5Data(data_path=speech_directory)
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        user_groups = cifar_iid(train_dataset, number_clients)  # 存储每个分区的样本序号的set组成的dict
        user_groups_test = cifar_iid(test_dataset, number_clients)  # 存储每个分区的样本序号的set组成的dict
        return train_dataset, test_dataset, user_groups, user_groups_test
    if dataset_name == "agnews":
        path = "/root/autodl-tmp/AG-News"
        if model_name == "charcnn":
            alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
            train_dataset = AG_NewsDataset(path + "/train/train.csv", 1014, alphabet)
            test_dataset = AG_NewsDataset(path + "/test/test.csv", 1014, alphabet)
            train_group = cifar_iid(train_dataset, number_clients)
            test_group = cifar_iid(test_dataset, number_clients)
            return train_dataset, test_dataset, train_group, test_group
        elif model_name == "vdcnn":
            alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
            train_dataset = AG_NewsDataset(path + "/train/train.csv", 1024, alphabet)
            test_dataset = AG_NewsDataset(path + "/test/test.csv", 1024, alphabet)
            train_group = cifar_iid(train_dataset, number_clients)
            test_group = cifar_iid(test_dataset, number_clients)
            return train_dataset, test_dataset, train_group, test_group

class AG_NewsDataset(Dataset):
    def __init__(self, path, l0, alphabet):
        super(AG_NewsDataset, self).__init__()
        self.alphabet = alphabet
        self.path = path
        self.l0 = l0
        self.data, self.label = self.load_data()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        data = self.data[item]
        data_tensor = torch.zeros(self.l0).long()
        for i, char in enumerate(data):
            if i == self.l0:
                break
            index = self.alphabet.find(char)
            if index != -1:
                data_tensor[i] = index
        label_tensor = torch.tensor(self.label[item])
        print(data_tensor)
        return data_tensor, label_tensor

    def load_data(self):
        data = []
        label = []
        with open(self.path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in csv_reader:
                text = ' '.join(row[1:]).lower()
                data.append(text)
                label.append(int(row[0]) - 1)
                # print(data[0])
        return data, label

class h5Data(Dataset):
    def __init__(self, data_path):
        self.data_path = "/root/autodl-tmp/0724zzp01rtx3080/test_useless/{}".format("zzph5-1.h5")
        self.data_path = data_path
        self.wb_list = None
        self.label_list = None
        a = None
        b = None
        with h5py.File(self.data_path, 'r') as h5f:
            print("keys: ", h5f.keys())
            a = copy.deepcopy(h5f['wavs'][:])
            b = copy.deepcopy(h5f['labels'][:])
        a = torch.tensor(a)
        self.wb_list = copy.deepcopy(a)
        b = torch.tensor(b)
        self.label_list = copy.deepcopy(b)

    def __len__(self):
        return len(self.wb_list)

    def __getitem__(self, index):
        return self.wb_list[index], self.label_list[index]

class Aduio_DataLoader(Dataset):
    def __init__(self, data_folder, sr=16000, dimension=16000):  # 8281=91*91
        self.data_folder = data_folder
        self.sr = sr
        self.dim = dimension
        self.labellist = []
        # 获取音频名列表
        self.wav_list = []
        for root, dirnames, filenames in os.walk(data_folder):
            for filename in fnmatch.filter(filenames, "*.wav"):  # 实现列表特殊字符的过滤或筛选,返回符合匹配“.wav”字符列表
                self.wav_list.append(os.path.join(root, filename))
        print("len of wav list:{}\n{}".format(len(self.wav_list), self.wav_list[0]))

    def __getitem__(self, item):
        # 读取一个音频文件，返回每个音频数据
        filename = self.wav_list[item]
        # print("cur label:{}".format(filename))
        # print(filename)
        wb_wav, _ = librosa.load(filename, sr=self.sr)
        # sr为采样率，通过KMplayer查看sampling rate，确认过speech commands为16000

        # 取 帧
        if len(wb_wav) > self.dim:  # self.dim=8281
            # print("yes:len of wb_wav{}:{}".format(filename, len(wb_wav)))
            max_audio_start = len(wb_wav) - self.dim
            audio_start = np.random.randint(0, max_audio_start)
            wb_wav = wb_wav[audio_start: audio_start + self.dim]
        else:
            wb_wav = np.pad(wb_wav, (0, self.dim - len(wb_wav)), "constant")

        wav = np.array(wb_wav, dtype=float)
        y = wav
        # 帧长为2048个点，帧移为512个点，由librosa官方文档指定,默认n_mels=128,所以输出的第一维度一定为128
        mel_spect = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=2048, hop_length=512)
        mel_spect = librosa.power_to_db(mel_spect)
        mel_spect = torch.tensor(mel_spect, dtype=torch.float32)
        mel_spect = mel_spect.unsqueeze(0)

        # 数据预处理，no.1 中心化
        mean1 = -35.430775023935055
        mel_spect = mel_spect.add((0 - mean1))
        # no.2 normalize
        mean2 = -0.35955178793979636
        std2 = 19.740814100800822
        mel_spect = mel_spect.add((0 - mean2))
        mel_spect = mel_spect.div(std2)
        # no.3 standardlize
        min = -2.966360110260923
        max = 3.859975074498963
        mel_spect = mel_spect.add(0 - min)
        mel_spect = mel_spect.div(max - min)
        mel_spect = mel_spect.add(-0.5)
        mel_spect = mel_spect.div(0.5)

        # # print("type mel_spect", mel_spect.dtype)
        # # mel_spect = torch.log(mel_spect)

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
        }
        numlabel = label_dict[label]
        return mel_spect, numlabel

    def __len__(self):
        # 音频文件的总数
        return len(self.wav_list)

class DatasetSplit(Dataset):  # DatasetSplit 类继承自父类 Dataset,按照idxs顺序有序排列
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # image = self.dataset[self.idxs[item]]
        return image, label
