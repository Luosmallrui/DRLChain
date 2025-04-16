import random
import numpy as np
import torch
from torchvision import datasets, transforms

from DQN import RLAgent
from FederatedLearningBlockchainEnv import FederatedLearningBlockchainEnv
from FederatedLearningModel import FederatedLearningModel, train_local_model, federated_averaging
from Node import Node
from federated_models.iid_noniid_sample import mnist_iid, mnist_noniid, cifar_iid
from federated_models.model_nets import CNNCifar, CNNMnist, MLP
import os
import csv
import matplotlib.pyplot as plt

def args_parser():
    import argparse
    parser = argparse.ArgumentParser()

    # 数据集选择
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset: mnist or cifar')

    # 数据分布选择
    parser.add_argument('--iid', action='store_true', help='Use IID data distribution')

    # 用户数量
    parser.add_argument('--num_users', type=int, default=100, help='Number of users')

    # 设备选择 (GPU or CPU)
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID, -1 for CPU')

    # 模型选择
    parser.add_argument('--model', type=str, default='cnn', help='Model: cnn or mlp')

    # 全局训练轮次，！！！！！
    parser.add_argument('--global_rounds', type=int, default=5, help='Number of global training rounds')

    # 每个用户本地训练轮次
    parser.add_argument('--local_epochs', type=int, default=10, help='Number of local epochs for each user')

    # 批次大小
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for local training')

    # 学习率
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the model')

    # 结果保存路径
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results and plots')

    #图像通道数
    parser.add_argument('--num_channels', type=int, default= 3, help='number of channels of image')
    # 每个客户端本地训练的batch大小
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    # 测试时的batch大小
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    # SDG中的动量参数
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    # 每种卷积核的数量
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    # 用于卷积操作的卷积核大小
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    # 数据集中的类别数量
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    #是否打印详细信息
    parser.add_argument('--verbose', action='store_true', help='verbose print')



    return parser.parse_args()


def plot_from_csv(file_path, x_label, y_label, title, save_path):
    """从 CSV 文件读取数据并绘制曲线"""
    x_data, y_data = [], []

    # 读取 CSV 文件
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            x_data.append(int(row[0]))  # 轮次
            y_data.append(float(row[1]))  # 结果数据

    # 绘制曲线
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, marker='o', linestyle='-', color='b', label=y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig(save_path)
    print(f"图像已保存到 {save_path}")
    plt.show()


def plot_results():
    """绘制所有结果的曲线图"""
    results_dir = os.path.join(os.path.dirname(__file__), "results")

    # 定义要绘制的 CSV 文件
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




if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 生成 100 个节点，随机初始化算力和通信能力
    num_nodes = 100
    nodes = {i: Node(i, random.uniform(5, 20) * 1e9, random.uniform(1, 10) * 1e6) for i in range(num_nodes)}
    # 初始化全局模型和本地模型
    #global_model = FederatedLearningModel(input_dim=32, output_dim=10)  # 示例：32维输入，10分类输出
    #local_models = [FederatedLearningModel(input_dim=32, output_dim=10) for _ in range(num_nodes)]

    # 载入数据集
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(os.path.join(os.path.dirname(__file__), 'data', 'mnist'), train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(os.path.join(os.path.dirname(__file__), 'data', 'mnist'),train=False, download=True, transform=transform)
        dict_users = mnist_iid(dataset_train, args.num_users) if args.iid else mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform)
        #dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform)
        dataset_train = datasets.CIFAR10(os.path.join(os.path.dirname(__file__), 'data', 'cifar'), train=True,
                                         download=True, transform=transform)
        dataset_test = datasets.CIFAR10(os.path.join(os.path.dirname(__file__), 'data', 'cifar'), train=False,
                                        download=True, transform=transform)

        dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # 选择模型
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = np.prod(img_size)
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_users).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob.to(args.device)
    print(net_glob)
    net_glob.train()

    global_model = net_glob
    # 创建区块链联邦学习环境
    # args, dataset_train, global_model
    env = FederatedLearningBlockchainEnv(args, dataset_train, dataset_test, global_model, dict_users)
    # agent = RLAgent(state_dim=env.observation_space.shape[0], action_dim=3)
    agent = RLAgent(state_dim=30, action_dim=2 + num_nodes)
    # 训练过程
    for epoch in range(10):  # 100 轮联邦学习训练
        state = env.reset()
        done = False
        transaction_count = 0
        total_training_time = 0



        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            if epoch % 10 == 0:
                agent.update_target()
            state = next_state
    plot_results()  # 训练完后绘制所有结果曲线
