import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

#本地数据的划分,用于将原始数据集划分为多个子集，以便在联邦学习中分配给每个客户端
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        #idxs：一个列表，包含当前客户端所分配的样本的索引。通过idxs，每个客户端可以获得一个属于自己的数据子集
    def __len__(self):
        return len(self.idxs)#当前客户端持有的样本数

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
'''
idxs 就是一个包含这些索引的列表，表示每个客户端应该访问的数据。例如：

客户端 0：idxs_0 = [0, 1, 2, ..., 1999]（从数据集中的前 2000 张图片）
客户端 1：idxs_1 = [2000, 2001, ..., 3999]（从数据集中的接下来的 2000 张图片）
客户端 2：idxs_2 = [4000, 4001, ..., 5999]
客户端 3：idxs_3 = [6000, 6001, ..., 7999]
客户端 4：idxs_4 = [8000, 8001, ..., 9999]
'''
#本地模型的训练
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args#包含各种配置参数的对象，如本地批量大小 local_bs、本地训练轮数 local_epochs、学习率 lr 等
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        #DataLoader 是用于从数据集中按批次加载数据的工具，所以ldr_train是加载的数据
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()#设置模型为训练模式
        # train and update
        # 使用随机梯度下降（SGD）优化器，设置学习率为 args.learning_rate，动量为 0.5
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.learning_rate, momentum=0.5, weight_decay=1e-4)

        epoch_loss = []
        for iter in range(self.args.local_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()#梯度清零
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net, sum(epoch_loss) / len(epoch_loss)

        #return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
#返回训练后的模型参数和所有轮次的平均损失
'''
net.state_dict() 是 PyTorch 中 nn.Module 类的一个方法，
用于返回模型的所有参数（权重、偏置）及其状态。
具体来说，state_dict 是一个字典，包含了模型的所有可训练参数以及一些非模型参数
'''