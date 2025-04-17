import copy
import csv
import math
import os

import gym
import numpy as np
import torch
from gym.spaces import Box

from Node import Node
from federated_models.local_update import DatasetSplit, LocalUpdate

# 状态空间最大值
MAX_NUM_NODES = 200
# MAX_HASH_RATE = 300##kGH/s，节点计算hash能力,随时间动态变化
# 这个和节点通信能力有关吗
MAX_COMM_RATE = 100  # 1 * np.power(2,20)是1MBps，传输速度，10-100Mbps
# MAX_TRAINING_DELAY = 20#20s
MAX_GLOBAL_ROUNDS = 30  # 30轮

# 动作空间最大值
# MAX_EPOCH_LEN = 100#这个是训练轮次吗，是做出来每一个动作吗，step的总数吗
MAX_SHARD_SIZE = 128
MAX_BLOCK_SIZE = 8  # 区块最大8MB
# MAX_BLOCK_SIZE = 10 * np.power(2, 10)  # 10 KB
# MAX_BLOCK_SIZE = np.power(2, 30)  # 1 GB = 1024 MB

# 状态空间
num_nodes = 100  # 节点数量
# hash_rate = 200#哈希速率##kGH/s，节点计算hash能力
comm_rate = 20  # 数据传输速率
# training_delay = 20#训练时长,20s
global_rounds = 20  # 全局通信轮次!!!!!这里设置了全局通信伦茨
# 动作空间
# epoch_len = 100
shard_size = 5  # 这里注意是次方，2的几次方
block_size = 32
node_selection = np.zeros(num_nodes)
# 固定的动作空间
# fix_epoch = 1
fix_shard = 4
fix_block = 32
fix_node_selection = np.zeros(num_nodes, dtype=np.int32)
# 超参数
# 时延方面
# aggregation_delay = 0.3  # 聚合时延
# validation_delay = 0.1  # 验证时延,0.1s
ver_time = 0.1  # 验证时延？？？和上面一样
rand_time = 0.5  # 随机时延
z_time = 1  # Ts，稳定区块创建的时延
# a_time = 0.2#添加一个区块的时延
v_time = 0.5  # 一个新节点的身份提交时延
r_time = 2  # 重新构建委员会的时延
# 各种速率
# comm_rate = trans_rate #通信速率,这两个参数代表一个,往区块链上面上传东西的速率，区块链内部节点互相传信息
# 安全系数
gamma = 1 / 6  # 强化学习中的折扣因子

# 分片安全性方面
system_tolerance = 1 / 4
shard_tolerance = 1 / 3
msg_size = 1  # 消息大小,1MB-16MB，区块链包装的消息大小
model_data_size = 10  # 10M,联邦学习模型大小
# 模型数量
local_models_count = 0
shard_models_count = 0
global_models_count = 0
# 奖励惩罚方面
illegal_punishment = -1000
delay_punishment = -1000
assign_punishment = -1000
reputation_punishment = -1000
# 奖励参数
sigma = 0.8
omega = 0.2
# 优化问题的约束指标相关参数
beta = 0.8
D_threshold = 1  # 信誉值的门槛
security_parameter = 5  # 安全系数
# 扩展动作空间维度
# 创建动作空间的 high 数组
high = np.array([MAX_SHARD_SIZE, MAX_BLOCK_SIZE] + [1] * num_nodes)
# 确保 shapes 一致
assert high.shape == (2 + num_nodes,)
action_space = Box(low=np.array([0, 0] + [0] * num_nodes),
                   high=high,
                   dtype=np.float32
                   )  # 第三个参数32位浮点数
observation_space = Box(
    low=np.array([1, 0, 1]),
    high=np.array([MAX_NUM_NODES, MAX_COMM_RATE, MAX_GLOBAL_ROUNDS]),
    dtype=np.float32
)


class FederatedLearningBlockchainEnv(gym.Env):
    # skychain的状态空间是节点数量n，交易队列q
    # skychain的动作空间是epoch_len，shard_size分片大小，区块大小
    # DRL文章的状态空间是节点数量，数据传输速率，训练时延和全局通信轮次
    # DRL文章的动作空间是数据大小M，shard_num分片数量，l分片的领导者，共识算法delta
    # 本文的动作空间是shard_num分片数量，l分片的领导者，epoch_len
    # def _initialize_csv(self, file_path, header):
    #     """初始化 CSV 文件，确保有表头"""
    #     if not os.path.exists(file_path):
    #         with open(file_path, mode='w', newline='') as file:
    #             writer = csv.writer(file)
    #             writer.writerow(header)  # 写入表头

    def _initialize_csv(self, file_path, header):
        """初始化 CSV 文件，清空旧数据并写入表头"""
        with open(file_path, mode='w', newline='') as file:  # 使用 'w' 模式清空文件
            writer = csv.writer(file)
            writer.writerow(header)  # 写入表头

    def _append_to_csv(self, file_path, data_row):
        """追加数据到 CSV 文件"""
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_row)

    def __init__(self, args, dataset_train, dataset_test, global_model, dict_users):
        self.dict_users = dict_users
        # 确保 results 目录存在
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)  # 创建 results 目录（如果不存在）
        # 定义 CSV 文件路径
        self.reward_csv_file_path = os.path.join(results_dir, "Reward_results.csv")
        self.throughput_csv_file_path = os.path.join(results_dir, "Throughput_results.csv")
        self.accuracy_csv_file_path = os.path.join(results_dir, "Accuracy_results.csv")
        # 初始化 CSV 文件，写入表头（如果文件不存在）
        self._initialize_csv(self.reward_csv_file_path, ["Episode", "Reward"])
        self._initialize_csv(self.throughput_csv_file_path, ["Episode", "Throughput"])
        self._initialize_csv(self.accuracy_csv_file_path, ["Episode", "Accuracy"])

        # 这里对父类的继承是否有必要
        super(FederatedLearningBlockchainEnv, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space

        self.num_nodes = num_nodes  # 节点数量
        # self.hash_rate = hash_rate  # 哈希速率
        self.comm_rate = comm_rate  # 数据传输速率
        # 和main有关,init初始化需要的参数
        self.args = args
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.global_model = global_model

        # 和联邦学习优化目标有关，吞吐量
        self.local_models_count = local_models_count
        self.shard_models_count = shard_models_count
        self.global_models_count = global_models_count
        # 与计算时延有关
        self.msg_size = msg_size  # 消息大小

        self.ver_time = ver_time

        self.comm_rate = comm_rate
        self.rand_time = rand_time
        self.z_time = z_time
        self.v_time = v_time
        self.r_time = r_time
        self.gamma = gamma
        # 与分片安全性有关
        self.security_parameter = security_parameter
        # 优化目标约束
        self.D_threshold = D_threshold
        self.beta = beta
        # 状态空间
        self.num_nodes = num_nodes  # 节点数量
        # 与区块链相关的参数
        # self.hash_rate = hash_rate#哈希速率
        # self.block_size = block_size#区块大小（到底应该是状态空间还是动作空间）

        # 与联邦学习相关的参数

        # self.training_delay = training_delay#训练时长,有其他类型的时延需要补充吗？
        self.global_rounds = global_rounds  # 全局通信轮次
        # 状态空间的数组表示形式，需要进行修改
        self.observation = np.array([self.num_nodes, self.comm_rate, self.global_rounds])
        # 动作空间
        # self.epoch_len = epoch_len#吞吐量的分母
        self.shard_size = shard_size  # 分片大小
        # self.shard_num = shard_num#分片数量
        self.block_size = block_size  # 区块大小，在skychain中这是一个超参数
        self.node_selection = np.zeros(num_nodes, dtype=np.int32)  # 节点选择策略
        # 动作空间的数组表示形式
        self.action = [self.shard_size, self.block_size, self.node_selection]
        # 超参数

        # self.aggregation_delay = aggregation_delay  # 聚合时延
        # self.validation_delay = validation_delay  # 验证时延

        self.system_tolerance = system_tolerance
        self.shard_tolerance = shard_tolerance

        # 联邦学习相关
        self.local_models = [np.random.rand(10) for _ in range(num_nodes)]
        # self.global_model = np.mean(self.local_models, axis=0)
        # print(f"global_model type: {type(self.global_model)}")  # 这里的 net 应该是 PyTorch 模型

        # 信誉评价相关
        self.sigma = sigma  # 信誉评价的权重系数
        self.omega = omega  # 信誉评价的权重系数

        self.node_shards = np.full(self.num_nodes, -1)  # -1 表示初始状态未分配分片

        # 生成节点的属性
        # 生成 100 个节点，节点 ID 从 0 到 99
        # num_nodes = 100
        self.nodes = {}
        # 模拟生成节点的算力和通信能力
        for i in range(num_nodes):
            computation_power = np.random.uniform(5000, 20000)  # 假设算力在 # 5 GFLOPS-20 之间均匀分布
            communication_power = np.random.uniform(10, 30)  # 假设通信能力在 10M-100M 之间均匀分布
            self.nodes[i] = Node(i, computation_power, communication_power)

        self.step_number = 0  # 初始化环境步数
        self.eposide_Reward = 0
        self.eposide_Throughput = 0
        self.eposide_Accuracy = 0
        '''fix'''
        # self.fix_epoch = fix_epoch
        self.fix_shard = fix_shard
        self.fix_block = fix_block
        self.fix_node_selection = fix_node_selection

        self.model_data_size = model_data_size  # 假设模型大小为10M，可根据实际调整

    def reset(self):
        # 随机初始化状态，包含每个节点的计算能力、交易队列等
        # 状态空间
        self.num_nodes = num_nodes  # 节点数量
        # self.hash_rate = hash_rate  # 哈希速率
        self.comm_rate = 10 * np.random.uniform(0.95, 1.05)  # 假设传输速率波动
        # self.training_delay = training_delay  # 训练时长,有其他类型的时延需要补充吗？
        self.global_rounds = global_rounds  # 全局通信轮次
        # 动作空间
        # self.epoch_len = epoch_len
        self.shard_size = shard_size
        self.block_size = block_size
        self.node_selection = node_selection
        self.gamma = gamma  # 强化学习中的折扣因子
        # 状态空间包含：节点数量，哈希速率，数据传输率，训练时长，全局通信轮次
        obs_len_0 = 15
        obs_len_1 = 10
        obs_len_2 = 5
        # new_obs = observation_space.copy()
        new_obs = np.copy(observation_space.sample())  # 生成 Box 空间的样本并复制

        bi_obs_0 = self.int_to_binary_array(int(new_obs[0]), obs_len_0)
        bi_obs_1 = self.int_to_binary_array(int(new_obs[1]), obs_len_1)
        bi_obs_2 = self.int_to_binary_array(int(new_obs[2]), obs_len_2)

        # 这里需要进行修改，结合状态空间的格式
        return np.hstack((bi_obs_0, bi_obs_1, bi_obs_2))

    # 返回值是{新状态，奖励，是否完成（环境是否结束），信息}
    def step(self, action):
        # 1.解析动作：时间、分片数量/大小、区块大小、节点选择策略
        # action[1] = int(action[1] * 8)
        # action[1] = np.power(2, action[1])  # 变成2的action[1]次方,分片大小，分片内节点数量
        '''
        self.epoch_len = action[0]
        self.shard_size = int(action[1])
        self.block_size = int(action[2])#这个可能是个超参数
        self.node_selection = action[3:]
        '''

        # 这里是2的action[0]次方不能超过MAX_SHARD_SIZE
        self.shard_size = min(MAX_SHARD_SIZE, int(np.power(2, action[0])))  # 限制分片大小不超过最大值
        self.block_size = int(action[1])
        self.node_selection = (action[2:] > 0.5).astype(int)  # 确保 node_selection 只有 0 和 1

        # 没有self.的变量是静态的，有self.的是动态的
        self.fix_shard = np.power(2, shard_size)
        self.fix_block = block_size  # 这个可能是个超参数
        self.fix_node_selection = node_selection

        # 计算选出的Leader数量
        num_leaders = np.sum(self.node_selection)  # 统计有多少个Leader

        # 设定最小Leader数量
        MIN_LEADER_NUM = max(1, self.num_nodes // 10)  # 至少1个Leader，或者至少占10%
        MAX_LEADER_NUM = self.num_nodes - 1  # 不能所有节点都是Leader

        # 状态空间包含：节点数量，数据传输率，全局通信轮次
        obs_len_0 = 15
        obs_len_1 = 10
        obs_len_2 = 5
        # 这里action是action还是self.action
        info = {'observation': self.observation, 'action': action, 'step_number': self.step_number}

        # 非法情况检查
        if (
                # self.fix_epoch >= MAX_EPOCH_LEN or
                self.shard_size >= self.num_nodes or
                num_leaders == 0 or  # 不能没有Leader
                num_leaders == self.num_nodes or  # 不能所有节点都是Leader
                num_leaders < MIN_LEADER_NUM or  # Leader太少
                num_leaders > MAX_LEADER_NUM  # Leader太多
        ):
            print("Invalid selection: too large epoch/shard, or leader selection issue.")
            reward = illegal_punishment
            # new_obs = self.observation.copy()
            new_obs = np.copy(self.observation_space.sample())  # 生成 Box 空间的样本并复制

            bi_obs_0 = self.int_to_binary_array(int(new_obs[0]), obs_len_0)
            bi_obs_1 = self.int_to_binary_array(int(new_obs[1]), obs_len_1)
            bi_obs_2 = self.int_to_binary_array(int(new_obs[2]), obs_len_2)

            self.eposide_Reward += 1
            self._append_to_csv(self.reward_csv_file_path, [self.eposide_Reward, reward])
            # 这里需要进行修改，结合状态空间的格式
            return np.hstack((bi_obs_0, bi_obs_1, bi_obs_2)), reward, False, info
        self.step_number += 1
        reward = 0
        done = 0  # false
        illegal = False  # 后续并没有用到

        # 2.确定分片信息
        # 被选中的节点索引列表
        selected_nodes = np.where(self.node_selection == 1)[0]
        shard_num = len(selected_nodes)
        unselected_nodes = np.setdiff1d(np.arange(self.num_nodes), selected_nodes)
        np.random.shuffle(unselected_nodes)
        shards = [[] for _ in range(shard_num)]
        for i, node in enumerate(unselected_nodes):
            shards[i % shard_num].append(node)
        for i, leader in enumerate(selected_nodes):
            shards[i].append(leader)
        # 更新 node_shards 数组
        self.node_shards = np.full(self.num_nodes, -1)  # 重置为未分配状态
        for shard_index, shard_nodes in enumerate(shards):
            for node in shard_nodes:
                self.node_shards[node] = shard_index
        # 3. 启动分片内训练
        shard_models = []
        local_models_count = 0  # 统计本地模型数量
        shard_models_count = 0  # 统计分片模型数量

        # 进行 global_rounds 轮全局训练
        for epoch in range(self.global_rounds):
            shard_models = [None] * len(shards)  # 存储每个分片的聚合模型

            # 遍历每个分片,这里local_models存储暂时的分片内模型，但是self.local_model是存储整体的全局的local_models
            for shard_index, shard in enumerate(shards):
                local_models = [None] * len(shard)  # 存储该分片内的本地模型

                # 训练分片内的 Follower 节点
                for node_index, node in enumerate(shard):
                    if node == selected_nodes[shard_index]:  # Leader 节点不进行本地训练
                        continue
                    local_model = self.local_train(node)  # 本地训练
                    local_models[node_index] = local_model  # 保存模型
                    local_models_count += 1  # 统计本地模型数量

                # 训练 Leader 节点
                leader_node_index = shard.index(selected_nodes[shard_index])  # 获取 Leader 在当前分片中的索引
                leader_model = self.local_train(selected_nodes[shard_index])  # Leader 本地训练
                local_models[leader_node_index] = leader_model  # 将 Leader 的模型保存
                local_models_count += 1  # Leader 本地模型数量 +1

                # **对当前分片内的本地模型进行聚合**
                shard_models[shard_index] = self.federated_averaging(local_models)  # 聚合当前分片的模型
                shard_models_count += 1  # 统计分片模型数量

            # **所有分片完成后，计算全局模型**
            # self.global_model = np.mean([m.state_dict() for m in shard_models], axis=0)
            # 获取所有分片聚合后的模型参数（每个shard都有一个模型）
            shard_model_dicts = [m.state_dict() for m in shard_models]

            # 初始化全局模型参数为深拷贝第一个分片的模型参数
            global_model_dict = copy.deepcopy(shard_model_dicts[0])

            # 计算所有分片模型的平均值
            for key in global_model_dict:
                global_model_dict[key] = torch.mean(
                    torch.stack([shard_model_dicts[i][key] for i in range(len(shard_model_dicts))]), dim=0)

            # 加载到 global_model
            self.global_model.load_state_dict(global_model_dict)

            # 计算 global_model 的 accuracy
            global_accuracy = self.evaluate_model(self.global_model)

            self.eposide_Accuracy += 1
            # 记录到 CSV
            print("accury", self.eposide_Accuracy, global_accuracy)
            self._append_to_csv(self.accuracy_csv_file_path, [self.eposide_Accuracy, global_accuracy])

        # 迭代每个分片
        # for shard_index, shard in enumerate(shards):
        #     local_models = [None] * len(shard)  # 保证索引位置与节点编号对应
        #     # 进行 shard_epoch_len 次训练（每轮训练）
        #     for epoch in range(self.global_rounds):#这是全局训练的轮次
        #         # 训练每个分片的 Follower 节点
        #         for node_index, node in enumerate(shard):
        #             if node == selected_nodes[shard_index]:  # Leader 节点不进行此操作
        #                 continue
        #             local_model = self.local_train(node)  # 本地训练
        #             local_models[node_index] = local_model  # 保存模型到对应的索引位置
        #             local_models_count += 1  # 本地模型数量 +1
        #
        #         # Leader 也进行本地训练，保存到对应的索引位置
        #         leader_node_index = shard_index  # Leader 节点是当前分片中的最后一位
        #         leader_model = self.local_train(shard[leader_node_index])  # Leader 本地训练
        #         local_models[leader_node_index] = leader_model  # 将Leader节点的模型保存到对应位置
        #         local_models_count += 1  # Leader 本地模型数量 +1
        #
        #         # 进行模型聚合，使用所有节点的本地模型
        #         aggregated_model = self.federated_averaging(local_models)  # 聚合所有模型
        #         shard_models.append(aggregated_model)
        #         shard_models_count += 1  # 分片模型数量 +1
        #         self.global_model = np.mean(shard_models, axis=0)
        #         # 计算 global_model 的 accuracy
        #         global_accuracy = self.evaluate_model(self.global_model, self.test_dataset)
        #
        #         self.eposide_Accuracy += 1
        #         # 记录到 CSV
        #         self._append_to_csv(self.accuracy_csv_file_path, [self.eposide_Accuracy, global_accuracy])

        # 计算时延，
        training_data_size = 100
        T_intra_shards = []
        for shard in shards:
            K = len(shard)
            T_prop = 2 * K * (K - 1) * self.msg_size / self.comm_rate  # PBFT消息传播时间
            T_val = 3 * self.ver_time  # 交易验证时延
            # 计算每个分片内节点完成一轮训练的时间
            training_times = [training_data_size / self.nodes[node].computation_power for node in shard]
            # 取最大值作为该分片的训练时延
            # T_comp = max(training_times)
            T_comp = np.percentile(training_times, 90)  # 取 90% 分位数

            # 计算Leader上传分片模型到区块链的时延
            leader_node = shard[-1]  # 假设分片的最后一个节点为Leader
            # model_data_size = self.model_size  # 分片模型的大小
            # 数据传输速率comm_rate相关的配置在前面，一个是一开始的参数设置为20，另外一个是初始化节点时
            T_upload = self.model_data_size / self.nodes[leader_node].communication_power  # Leader上传时延
            # 计算单个分片内部时延
            T_intra_shard = T_prop + T_val + T_comp * self.args.local_epochs + T_upload
            T_intra_shards.append(T_intra_shard)

        T_intra = max(T_intra_shards)
        T_inter = shard_num * self.msg_size / self.comm_rate
        T_reco = self.rand_time + self.z_time + self.v_time + self.r_time
        T_round = T_intra + T_inter + T_reco

        # 检查时延约束
        if T_round > self.gamma * (T_intra + T_inter):
            print("时延超出约束！")

        # 判断分片安全性
        is_secure = self._shard_security_hyp(shards)
        if not is_secure:
            print("分片存在安全风险！")

        # 计算分片内节点信誉均衡程度
        balance_degree = self._reputation_balance_degree(shards)
        if balance_degree >= self.beta * self.D_threshold:
            print("节点信誉值分布不够均衡")

        # 4. 全局聚合
        # 计算吞吐量
        total_models_count = local_models_count + shard_models_count + 1
        Throughput = total_models_count / (T_round * self.block_size * self.global_rounds)
        self.eposide_Throughput += 1
        self._append_to_csv(self.throughput_csv_file_path, [self.eposide_Throughput, Throughput])
        # 对约束进行判断,满足约束条件
        if T_round <= self.gamma * (T_intra + T_inter) and self._shard_security_hyp(
                shards) and self._reputation_balance_degree(shards) < self.beta * self.D_threshold:
            reward = Throughput
            # new_obs = self.observation.copy()
            new_obs = np.copy(self.observation_space.sample())  # 生成 Box 空间的样本并复制

            bi_obs_0 = self.int_to_binary_array(int(new_obs[0]), obs_len_0)
            bi_obs_1 = self.int_to_binary_array(int(new_obs[1]), obs_len_1)
            bi_obs_2 = self.int_to_binary_array(int(new_obs[2]), obs_len_2)
            self.eposide_Reward += 1
            self._append_to_csv(self.reward_csv_file_path, [self.eposide_Reward, reward])
            # 这里需要进行修改，结合状态空间的格式
            return np.hstack((bi_obs_0, bi_obs_1, bi_obs_2)), reward, done, info

        else:
            if T_round > self.gamma * (T_intra + T_inter):
                reward = delay_punishment
            elif not self._shard_security_hyp(shards):
                reward = assign_punishment
            else:
                reward = reputation_punishment
            # new_obs = self.observation.copy()
            new_obs = np.copy(self.observation_space.sample())  # 生成 Box 空间的样本并复制

            bi_obs_0 = self.int_to_binary_array(int(new_obs[0]), obs_len_0)
            bi_obs_1 = self.int_to_binary_array(int(new_obs[1]), obs_len_1)
            bi_obs_2 = self.int_to_binary_array(int(new_obs[2]), obs_len_2)
            self.eposide_Reward += 1
            self._append_to_csv(self.reward_csv_file_path, [self.eposide_Reward, reward])
            # 这里需要进行修改，结合状态空间的格式
            return np.hstack((bi_obs_0, bi_obs_1, bi_obs_2)), reward, done, info

        # 9. 判断是否完成
        done = self.step_number >= 100  # 预设条件：达到 100 轮全局通信
        # new_obs = self.observation.copy()
        new_obs = np.copy(self.observation_space.sample())  # 生成 Box 空间的样本并复制

        bi_obs_0 = self.int_to_binary_array(int(new_obs[0]), obs_len_0)
        bi_obs_1 = self.int_to_binary_array(int(new_obs[1]), obs_len_1)
        bi_obs_2 = self.int_to_binary_array(int(new_obs[2]), obs_len_2)
        self.eposide_Reward += 1
        print("reward", self.eposide_Reward, reward)
        self._append_to_csv(self.reward_csv_file_path, [self.eposide_Reward, reward])
        # 这里需要进行修改，结合状态空间的格式
        return np.hstack((bi_obs_0, bi_obs_1, bi_obs_2)), reward, done, info

    def federated_averaging(self, local_models):
        # 确保 local_models 不是空的
        if not local_models:
            raise ValueError("local_models 不能为空")
        # 深拷贝第一个模型，作为基础模型
        aggregated_model = copy.deepcopy(local_models[0])
        aggregated_model_dict = aggregated_model.state_dict()
        # 聚合各个节点的本地模型参数
        local_weights = [model.state_dict() for model in local_models]

        # 对每个参数进行平均
        for key in aggregated_model_dict:
            aggregated_model_dict[key] = torch.mean(
                torch.stack([local_weights[i][key] for i in range(len(local_weights))]),
                dim=0)

        aggregated_model.load_state_dict(aggregated_model_dict)
        return aggregated_model

    def aggregate_models(self, local_models):
        """
        对本地模型进行简单平均聚合
        :param local_models: List[torch.nn.Module]，包含多个本地模型
        :return: 聚合后的 state_dict
        """
        if len(local_models) == 0:
            raise ValueError("local_models 不能为空")

        # 深拷贝第一个模型的参数
        aggregated_model = copy.deepcopy(local_models[0].state_dict())

        # 遍历所有参数进行累加
        for key in aggregated_model.keys():
            for i in range(1, len(local_models)):
                aggregated_model[key] += local_models[i].state_dict()[key]

            # 计算平均值
            aggregated_model[key] /= len(local_models)

        return aggregated_model  # 返回聚合后的 state_dict

    def evaluate_model(self, model):
        """计算全局模型的测试精度"""
        correct = 0
        total = 0
        model.eval()  # 设置为评估模式

        with torch.no_grad():
            for data, labels in self.dataset_test:
                # label是int类型
                if data.dim() == 2:  # 可能是 (batch_size, 784)，需要 reshape
                    data = data.view(-1, 1, 28, 28)
                elif data.dim() == 3:  # 可能是 (batch_size, 28, 28)，也需要增加通道维度
                    data = data.unsqueeze(1)
                # 确保 labels 是 Tensor 并扩展维度
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels).unsqueeze(0)  # 变成 1 维张量

                outputs = model(data)

                # for data, labels in self.dataset_test:
                #     outputs = model(data)  # 前向传播
                _, predicted = torch.max(outputs, 1)  # 预测类别
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    # 计算组合数Cnk
    def _calculate_combination(self, n, k):
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

    def _shard_security_hyp(self, shards):
        pro = 0.0
        for shard in shards:
            shard_size = len(shard)  # 获取分片中节点的数量
            malicious_count = sum([self.nodes[node].is_malicious for node in shard])
            if malicious_count > shard_size * self.shard_tolerance:
                return False
            for x in range(math.ceil(shard_size * self.shard_tolerance), shard_size + 1):
                k = malicious_count
                f = k / shard_size
                term = self._calculate_combination(shard_size, x) * (f ** x) * ((1 - f) ** (shard_size - x))
                pro += term
        return pro < math.pow(2, -2)
        # return pro < math.pow(2, -self.security_parameter)

    def _reputation_balance_degree(self, shards):
        """
        计算分片内节点信誉均衡程度
        """
        max_reputations = [max([self.nodes[node].reputation for node in shard]) for shard in shards]
        mean_max_reputation = np.mean(max_reputations)
        degree = np.sum([(r_max - mean_max_reputation) ** 2 for r_max in max_reputations])
        return degree

    '''
    def local_train(self, client_id):
        local_model = copy.deepcopy(self.global_model)
        optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.5)

        client_idxs = [i for i in range(len(self.dataset_train)) if self.node_shards[i] == self.node_shards[client_id]]
        client_dataset = torch.utils.data.Subset(self.dataset_train, client_idxs)
        client_loader = DataLoader(client_dataset, batch_size=self.args.batch_size, shuffle=True)

        for epoch in range(self.args.local_ep):
            local_model.train()
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(client_loader):
                optimizer.zero_grad()
                output = local_model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        return local_model.state_dict()
    '''

    # bendi1
    def local_train(self, node):
        # 获取当前节点的数据索引
        idxs = self.dict_users[node]

        # 通过 DatasetSplit 来切分数据
        local_dataset = DatasetSplit(self.dataset_train, idxs)

        # 创建 LocalUpdate 对象并进行训练
        local_update = LocalUpdate(args=self.args, dataset=self.dataset_train, idxs=idxs)

        # print(f"global_model type: {type(self.global_model)}")  # 这里的 net 应该是 PyTorch 模型
        local_model, _ = local_update.train(self.global_model)

        # local_model = local_update.train(self.global_model)[0]  # 训练并返回模型的参数字典

        return local_model

    def cos_similarity(self, w1, w2):
        """计算两个向量的余弦相似度"""
        return np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))

    def intra_shard_reputation(self, node_index, shard_index):
        """计算分片内信誉值"""
        node_w = self.local_models[node_index]
        shard_nodes = np.where(self.node_shards == shard_index)[0]
        # 这里是否要根据数据量执行加权
        shard_w = np.mean([self.local_models[i] for i in shard_nodes], axis=0)

        similarity = self.cos_similarity(node_w, shard_w)
        leader_node = shard_nodes[-1]
        is_leader = node_index == leader_node  # 这是一个布尔值
        if similarity == 1:
            # 假设节点 0 为领导者，实际中根据具体逻辑判断，is_leader指的是leader节点的索引
            # is_leader = node_index == 0
            bonus = self.sigma * self.model_data_size - self.omega / self.model_data_size if is_leader else self.sigma * self.model_data_size
        else:
            bonus = 0
        return self.model_data_size * similarity + bonus

    def inter_shard_reputation(self, shard_index):
        """计算分片外信誉值"""
        shard_nodes = np.where(self.node_shards == shard_index)[0]
        shard_w = np.mean([self.local_models[i] for i in shard_nodes], axis=0)
        # 这里需要修改
        # model_data_size = 1  # 简单模拟，实际中可能根据数据量计算
        return self.model_data_size * self.cos_similarity(shard_w, self.global_model)

    def intermediate_reputation(self, node_index):
        """计算中间信誉值"""
        shard_index = self.node_shards[node_index]
        intra_rep = self.intra_shard_reputation(node_index, shard_index)
        inter_rep = self.inter_shard_reputation(shard_index)
        return intra_rep * inter_rep

    def comprehensive_reputation(self):
        """计算所有节点的综合信誉值"""
        all_intermediate_rep = [self.intermediate_reputation(i) for i in range(self.num_nodes)]
        avg_reputation = np.mean(all_intermediate_rep)
        comprehensive_rep = []
        for i in range(self.num_nodes):
            r_i = all_intermediate_rep[i]
            f_i = self.self.nodes[i].computation_power
            c_i_prime = f_i * (r_i - avg_reputation)
            if c_i_prime <= 0:
                c_i = np.exp(c_i_prime)
            else:
                c_i = 1 + np.log(1 + c_i_prime)
            comprehensive_rep.append(c_i)
        return comprehensive_rep

    # 整数转化为二进制
    def int_to_binary_array(self, value, bit_length):
        '''

        return np.unpackbits(np.array([value], dtype=np.uint8), bitorder='big')[-bit_length:]


        '''
        # 将value转换为整数（假设value应为整数）
        integer_value = int(value)

        # 确保整数值对应的二进制形式足够长
        binary_value = np.binary_repr(integer_value, width=bit_length)

        # 返回固定长度的二进制数组
        return np.array([int(b) for b in binary_value], dtype=np.uint8)
