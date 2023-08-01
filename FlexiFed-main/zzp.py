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
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
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


def get_dataset(dataset_name: object) -> object:
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
        # data_dir = '../data/cifar10'
        data_dir = os.getcwd()
        print(data_dir)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # train_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=train_transform)
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=False,
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=False,
                                        transform=test_transform)

        user_groups = cifar_iid(train_dataset, 20)  # 用户个数20个
        user_groups_test = cifar_iid(test_dataset, 20)

        return train_dataset, test_dataset, user_groups, user_groups_test

    if dataset_name == "cinic10":
        cinic_directory = '/root/CINIC-10'
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

        user_groups = cifar_iid(train_dataset, 20)
        user_groups_test = cifar_iid(test_dataset, 20)

        return train_dataset, test_dataset, user_groups, user_groups_test

    if dataset_name == "speech_commands":
        speech_directory = "/root/autodl-tmp/speech_commands"
        dataset = Aduio_DataLoader(speech_directory + '/train', sr=16000, dimension=16000)
        # print(dataset.wav_list)
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        # 随机划分数据集和测试集
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        user_groups = cifar_iid(train_dataset, 20)  # 存储每个分区的样本序号的set组成的dict
        user_groups_test = cifar_iid(test_dataset, 20)  # 存储每个分区的样本序号的set组成的dict

        return train_dataset, test_dataset, user_groups, user_groups_test

    if dataset_name == "new_speech": # 目标h5文件下存放有预处理过后的数据
        speech_directory = "/root/autodl-tmp/{}".format("zzph5-1.h5")
        dataset = h5Data(data_path=speech_directory)
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        user_groups = cifar_iid(train_dataset, 20)  # 存储每个分区的样本序号的set组成的dict
        user_groups_test = cifar_iid(test_dataset, 20)  # 存储每个分区的样本序号的set组成的dict
        return train_dataset, test_dataset, user_groups, user_groups_test

class h5Data(Dataset):
    def __init__(self, data_path):
        self.data_path ="/root/autodl-tmp/0724zzp01rtx3080/test_useless/{}".format("zzph5-1.h5")
        self.data_path =data_path
        self.wb_list = None
        self.label_list = None
        a = None
        b = None
        with h5py.File(self.data_path, 'r') as h5f:
            print("keys: ", h5f.keys())
            a = copy.deepcopy(h5f['wavs'][:])
            b = copy.deepcopy(h5f['labels'][:])
        a = torch.tensor(a)
        self.wb_list=copy.deepcopy(a)
        b = torch.tensor(b)
        self.label_list=copy.deepcopy(b)

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
        print("len of wav list:{}\n{}".format(len(self.wav_list),self.wav_list[0]))

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
        mel_spect = torch.tensor(mel_spect,dtype=torch.float32)
        mel_spect = mel_spect.unsqueeze(0)

        #数据预处理，no.1 中心化
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

def common_basic(w):
    """
    w=modelaccept
    modelaccept 维护八个用户端的迭代model
    """
    minIndex = 0
    minLength = 10000000

    # 求层数最少的w[i]，用minLength表示，下标为minIndex
    for i in range(0, len(w)):
        if len(w[i]) < minLength:
            minIndex = i
            minLength = len(w[i])
    # print("minIndex:", minIndex, "minLength:", minLength)

    # commonList用于存全局相同的基础层的keys（）
    commonList = [s for s in w[minIndex].keys()]

    for i in range(0, len(w)):
        # local_weights_names存每个w[i]的keys()
        local_weights_names = [s for s in w[i].keys()]
        # 比较每个model中与minmodel不一样的第一层，将commonlist从j到minlength删除
        for j in range(len(commonList)):
            if commonList[j] == local_weights_names[j]:
                continue
            else:
                del commonList[j:len(commonList) + 1]  # 删除函数
                break
    # 此时commonList表示8个model共享的相同的基础层,完全一致

    # 从0-commonList中,对每一层求均值存在comWeight中,赋值给8个model
    for k in commonList:
        comWeight = copy.deepcopy(w[0][k])
        # comWeight深拷贝w[0][k],model 0的第k层
        # print("comWeight1:",comWeight)
        for i in range(1, len(w)):
            comWeight += w[i][k]
        # 此时comWeight为8个model第k层的总和
        comWeight = comWeight / len(w)

        # 将每个第k层的均值comWeight赋值给每个model的第k层
        for i in range(0, len(w)):
            w[i][k] = comWeight
    return w

def common_max(w):
    w_copy = copy.deepcopy(w)
    count = [[] for i in range(len(w))]
    for i in range(len(w)):
        local_weights_names = [s for s in w[i].keys()]
        count[i] = [1 for m in range(len(local_weights_names))]

    for i in range(0, len(w)):

        local_weights_names1 = [s for s in w[i].keys()]

        for j in range(i + 1, len(w)):
            if i == j:
                continue
            local_weights_names2 = [s for s in w[j].keys()]
            for k in range(0, len(local_weights_names1)):
                if local_weights_names2[k] == local_weights_names1[k]:
                    name = local_weights_names1[k]
                    w[i][name] += w_copy[j][name]
                    w[j][name] += w_copy[i][name]
                    count[i][k] += 1
                    count[j][k] += 1
                else:
                    break

    for c in range(0, len(w)):
        local_weights_names = [s for s in w[c].keys()]
        for k in range(0, len(local_weights_names)):
            w[c][local_weights_names[k]] = w[c][local_weights_names[k]].cpu() / count[c][k]

    return w

def compare(w1, w2):
    if len(w1) != len(w2):
        return False
    keys1 = list(w1.keys())
    keys2 = list(w2.keys())
    # print("compare\n",keys1, '\n', keys2)
    for j in range(len(keys1)):
        if keys1[j] == keys2[j]:
            continue
        else:
            return False
    return True

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        # print("len of w", len(w))
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def common_clustered(w):
    """
    step1:对w使用common_basic,聚合所有的全局公共基础层
    step2:将结构完全一致的model划分集群
    step3:对每个集群采取FedAvg,得到一个聚合后的w_avg,赋值给该集群中的所有model
    """
    # step1:对w使用common_basic,聚合所有的全局公共基础层
    w_basic = common_basic(w)
    # step2:将结构完全一致的model划分集群
    clu = [[]]
    cluids = [[]]
    c_fid = 0
    find = 0
    for i in range(len(w_basic)):
        # 第一个
        if len(clu[0]) == 0:
            clu[c_fid].append(w_basic[i])
            cluids[c_fid].append((i))
            # clu[c_fid][i] = w_basic[i]
            continue
        # 找到同类
        for j in range(len(clu)):
            # c1=clu[j].get(next(iter(clu[j])))
            c1 = clu[j][0]
            if compare(c1, w_basic[i]):
                # compare
                clu[j].append(w_basic[i])
                cluids[j].append((i))
                find = 1
                break
        # 找不到同类
        if find == 0:
            clu.append([w_basic[i]])
            cluids.append(([i]))
        find = 0
    # 此时集群划分已完成，存储在clu中

    # step3: 对每个集群采取FedAvg, 得到一个聚合后的w_avg, 赋值给该集群中的所有model
    for i in range(len(clu)):
        avg = FedAvg(clu[i])
        for j in range(len(clu[i])):
            clu[i][j] = copy.deepcopy(avg)
    # 最终汇总输出w
    w_out = {_id: None for _id in range(8)}
    id_out = 0
    for i in range(len(clu)):
        for j in range(len(clu[i])):
            out_index = cluids[i][j]
            w_out[out_index] = copy.deepcopy(clu[i][j])
            id_out += 1

    return w_out

def local_train(net, dataset, idxs, device, lr):

    net.to(device)
    net.train()
    # train and update
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    ldr_train = DataLoader(DatasetSplit(dataset, list(idxs)), batch_size=64, shuffle=True)
    for iter in range(10):
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(device), labels.to(device)

            net.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return net.state_dict()

def test_inference(model, dataset, idxs, device):
    """ Returns the test accuracy and loss.
    """
    # model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    model.to(device)
    model.train()

    ldr_test = DataLoader(DatasetSplit(dataset, list(idxs)), batch_size=64, shuffle=False)

    for _, (images, labels) in enumerate(ldr_test):
        model.zero_grad()
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)

        # Prediction
        _, pred_labels = torch.max(outputs, 1)

        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct * 1.0 / total
    return accuracy

def utils():
    return None
