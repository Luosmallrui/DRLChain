import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):#state就是input，
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RLAgent:
    def __init__(self, state_dim, action_dim, epsilon=0.1, gamma=0.99, lr=1e-3, batch_size=32, memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)#重放缓存区
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.target_model.load_state_dict(self.model.state_dict())
        # 定义动作空间的范围
        self.max_shard_size = 128  # MAX_SHARD_SIZE
        self.max_block_size = 8    # MAX_BLOCK_SIZE
        self.num_nodes = 100
        self.high = np.array([self.max_shard_size, self.max_block_size] + [1] * self.num_nodes)

    # def choose_action(self, state):
    #     if random.random() < self.epsilon:
    #         # 以 ε 的概率执行随机探索，生成符合 (2 + num_nodes,) 形状的动作
    #         action = np.random.uniform(low=0, high=1, size=self.action_dim) * self.high
    #     else:
    #         # 以 (1 - ε) 的概率利用 Q 值最大的动作
    #         state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 增加 batch 维度
    #         q_values = self.model(state)  # 通过 DQN 计算 Q 值
    #         q_values = q_values.squeeze(0)  # 变成 (2 + num_nodes,)
    #
    #         # if q_values.shape[0] != self.action_dim:
    #         #     raise ValueError(f"模型输出的维度 {q_values.shape[0]} 不等于期望的 {self.action_dim}")
    #
    #         # **核心逻辑**: 归一化 Q 值，使其符合 `Box` 约束
    #         action = torch.sigmoid(q_values).detach().cpu().numpy()  # 归一化到 (0,1)
    #         action = action * self.high  # 缩放到 `high` 规定的范围内
    #
    #     return action.astype(np.float32)  # 确保返回类型是 np.float32

    def choose_action(self, state):
        if random.random() < self.epsilon:
            # 随机探索
            action = np.zeros(self.action_dim, dtype=np.float32)
            action[0] = np.floor(np.random.uniform(0, np.log2(self.max_shard_size)))

            #action[0] = np.random.uniform(0, np.log2(self.max_shard_size))  # 0 ~ log2(MAX_SHARD_SIZE)
            action[1] = np.random.uniform(0, self.max_block_size)  # 0 ~ MAX_BLOCK_SIZE
            action[2:] = np.random.uniform(0, 1, self.num_nodes)  # 0 ~ 1
        else:
            # 基于 Q 值的选择
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 增加 batch 维度
            q_values = self.model(state).squeeze(0)  # 变成 (2 + num_nodes,)

            action = torch.sigmoid(q_values).detach().cpu().numpy()  # 归一化到 (0,1)

            # **修改 action[0] 约束**
            action[0] *= np.log2(self.max_shard_size)  # 0 ~ log2(MAX_SHARD_SIZE)
            action[1] *= self.max_block_size  # 0 ~ MAX_BLOCK_SIZE

        return action.astype(np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # def update(self):
    #     if len(self.memory) < self.batch_size:
    #         return
    #
    #     batch = random.sample(self.memory, self.batch_size)
    #     states, actions, rewards, next_states, dones = zip(*batch)
    #
    #     states = torch.tensor(states, dtype=torch.float32)
    #     actions = torch.tensor(actions, dtype=torch.long)
    #     rewards = torch.tensor(rewards, dtype=torch.float32)
    #     next_states = torch.tensor(next_states, dtype=torch.float32)
    #     dones = torch.tensor(dones, dtype=torch.float32)
    #
    #     q_values = self.model(states)
    #     next_q_values = self.target_model(next_states)
    #
    #     # 计算目标Q值
    #     target_q_values = rewards + self.gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)
    #
    #     # 更新Q值
    #     loss = nn.MSELoss()(q_values.gather(1, actions.unsqueeze(1)).squeeze(1), target_q_values)
    #
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)  # 改为 float32
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states).squeeze(1)  # (batch_size, action_dim)

        next_q_values = self.target_model(next_states).detach()
        target_q_values = rewards + self.gamma * (1 - dones) * next_q_values.mean(dim=1)

        target_q_values = target_q_values.unsqueeze(1)
        loss = nn.MSELoss()(q_values, target_q_values.expand_as(q_values))

        #loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

