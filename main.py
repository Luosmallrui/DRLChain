import os
import csv
import random
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from DQN import RLAgent
from FederatedLearningBlockchainEnv import FederatedLearningBlockchainEnv
from federated_models.iid_noniid_sample import mnist_iid, mnist_noniid, cifar_iid
from federated_models.model_nets import CNNCifar, CNNMnist, MLP
from Node import Node

def args_parser():
    """参数解析"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset: mnist or cifar')
    parser.add_argument('--iid', action='store_true', help='Use IID data distribution')
    parser.add_argument('--num_users', type=int, default=100, help='Number of users')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID, -1 for CPU')
    parser.add_argument('--model', type=str, default='cnn', help='Model: cnn or mlp')
    parser.add_argument('--global_rounds', type=int, default=25, help='Number of global training rounds')
    parser.add_argument('--local_epochs', type=int, default=10, help='Number of local epochs for each user')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for local training')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the model')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results and plots')
    parser.add_argument('--num_channels', type=int, default=3, help='number of channels of image')
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    return parser.parse_args()

def ensure_directory(path):
    """确保保存目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_data(args):
    """根据参数获得数据集和用户划分"""
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(os.path.join(data_path, 'mnist'), train=True, download=True, transform=transform)
        test_set = datasets.MNIST(os.path.join(data_path, 'mnist'), train=False, download=True, transform=transform)
        dict_users = mnist_iid(train_set, args.num_users) if args.iid else mnist_noniid(train_set, args.num_users)
    elif args.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(os.path.join(data_path, 'cifar'), train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(os.path.join(data_path, 'cifar'), train=False, download=True, transform=transform)
        dict_users = cifar_iid(train_set, args.num_users)
    else:
        raise ValueError('Unrecognized dataset')
    return train_set, test_set, dict_users

def get_model(args, dataset_train):
    """选择模型"""
    if args.model == 'cnn':
        if args.dataset == 'cifar':
            return CNNCifar(args=args)
        elif args.dataset == 'mnist':
            return CNNMnist(args=args)
    elif args.model == 'mlp':
        img_size = dataset_train[0][0].shape
        len_in = int(np.prod(img_size))
        return MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes)
    else:
        raise ValueError('Unrecognized model')

def plot_from_csv(file_path, x_label, y_label, title, save_path):
    """从 CSV 文件读取数据并绘制曲线"""
    x_data, y_data = [], []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            x_data.append(int(row[0]))
            y_data.append(float(row[1]))
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, marker='o', linestyle='-', color='b', label=y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"图像已保存到 {save_path}")
    plt.close()

def plot_results(results_dir):
    """绘制所有结果的曲线图"""
    metrics = [
        ("Reward_results.csv", "Training Steps", "Reward", "Reward Curve", "reward_curve.png"),
        ("Throughput_results.csv", "Training Steps", "Throughput", "Throughput Curve", "throughput_curve.png"),
        ("Accuracy_results.csv", "Global Rounds", "Accuracy", "Accuracy Curve", "accuracy_curve.png"),
        ("Delay_results.csv", "Training Steps", "Delay (ms)", "Delay Curve", "delay_curve.png")
    ]
    for file_name, x_label, y_label, title, img_name in metrics:
        file_path = os.path.join(results_dir, file_name)
        save_path = os.path.join(results_dir, img_name)
        if os.path.exists(file_path):
            plot_from_csv(file_path, x_label, y_label, title, save_path)
        else:
            print(f"文件 {file_name} 不存在，跳过绘图")

def main():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print("device",args.device)
    ensure_directory(args.results_dir)

    # 节点初始化
    nodes = {i: Node(i, random.uniform(5, 20) * 1e9, random.uniform(1, 10) * 1e6) for i in range(args.num_users)}

    # 数据加载
    dataset_train, dataset_test, dict_users = get_data(args)

    # 模型初始化
    net_glob = get_model(args, dataset_train).to(args.device)
    net_glob.train()
    print(net_glob)

    # 联邦学习环境和RL agent
    env = FederatedLearningBlockchainEnv(args, dataset_train, dataset_test, net_glob, dict_users,use_wandb=True)
    agent = RLAgent(state_dim=30, action_dim=2 + args.num_users)

    # 联邦学习训练主循环
    for epoch in range(args.global_rounds):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            if epoch % 10 == 0:
                agent.update_target()
            state = next_state

    plot_results(args.results_dir)

if __name__ == '__main__':
    main()