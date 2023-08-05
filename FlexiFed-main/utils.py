from datasets.deldataset import *
import copy
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


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
                name = local_weights_names1[k]
                # 遇到全连接层直接跳过
                if "fc" in name:
                    # print("fc current，skip,i:{} j:{}".format(i, j))
                    break
                elif local_weights_names2[k] == local_weights_names1[k]:
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
        # 不聚合全连接层
        if "fc" in k:
            # print("fc current,skip")
            break
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


def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def local_train(client_id, net, dataset, idxs, device, lr, epoch, local_loss):
    net.to(device)
    net.train()
    # train and update
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # adjust_learning_rate(optimizer, epoch, lr)
    ldr_train = DataLoader(DatasetSplit(dataset, list(idxs)), batch_size=64, shuffle=False)
    # 打印当前lr
    # print("Client:{} Epoch:{}  Lr:{:.2E}"
    # .format(client_id, epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    for iter_t in range(10):
        # print("[client id:%d,epoch:%d] Local Training..." % (client_id, iter))
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 每次backward之前必须zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            if iter_t==9 and batch_idx==0:
                local_loss[client_id].append(round(loss.item(), 6))
            loss.backward()
            optimizer.step()
    return net.state_dict()


def test_inference(model, dataset, idxs, device):
    """ Returns the test accuracy and loss.
    """
    # model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    model.to(device)
    model.eval()
    ldr_test = DataLoader(DatasetSplit(dataset, list(idxs)), batch_size=64, shuffle=False)
    for _, (images, labels) in enumerate(ldr_test):
        # model.zero_grad()
        images, labels = images.to(device), labels.to(device)
        # Inference
        outputs = model(images)
        total += len(labels)
        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
    accuracy = correct * 1.0 / total
    return accuracy


def utils():
    return None
