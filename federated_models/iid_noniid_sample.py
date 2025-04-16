#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
#主要实现了在联邦学习场景下，对 MNIST 和 CIFAR-10 数据集进行独立同分布（I.I.D.）
# 和非独立同分布（Non-I.I.D.）采样的功能
#dict_users 的键是客户端的编号（0, 1, ..., num_users-1），
#每个键对应一个值，值是一个 set，该 set 包含该客户端分配到的数据集样本的索引。
from collections import defaultdict


def mnist_noniid_dirichlet(dataset, num_users, alpha=0.5):
    labels = dataset.targets.numpy() if hasattr(dataset, 'targets') else dataset.train_labels.numpy()
    num_classes = 10
    data_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    dict_users = defaultdict(list)
    for c in range(num_classes):
        np.random.shuffle(data_indices[c])
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = (np.cumsum(proportions) * len(data_indices[c])).astype(int)[:-1]
        split_data = np.split(data_indices[c], proportions)
        for i in range(num_users):
            dict_users[i] += list(split_data[i])

    return {i: np.array(v) for i, v in dict_users.items()}


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    该函数用于从 MNIST 数据集中进行独立同分布（I.I.D.）采样，
    将数据集均匀地分配给指定数量的客户端。
    dataset：MNIST 数据集对象。
    num_users：客户端的数量。
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
'''
返回结果是这样的
{
    0: {0, 1, 2, 3, ..., n-1},  # 客户端0的样本索引集合
    1: {n, n+1, n+2, ..., 2n-1},  # 客户端1的样本索引集合
    2: {2n, 2n+1, 2n+2, ..., 3n-1},  # 客户端2的样本索引集合
}
'''


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300#数据集被划分为的分片数量，每个分片包含的样本数量
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
