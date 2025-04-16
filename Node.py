import random

# 定义节点类
class Node:
    def __init__(self, node_id, computation_power, communication_power):
        self.node_id = node_id  # 节点 ID
        self.computation_power = computation_power  # 计算能力
        self.communication_power = communication_power  # 通信能力
        self.reputation = 1  # 初始信誉值在 0 到 1 之间
        self.is_malicious = random.random() < 0.2  # 假设20%的概率为恶意节点

    def update_reputation(self, new_reputation):
        """更新节点的信誉值"""
        self.reputation = new_reputation

    def __repr__(self):
        """表示节点的信息"""
        return f"Node(ID: {self.node_id}, Computation Power: {self.computation_power}, Communication Power: {self.communication_power}, Reputation: {self.reputation:.2f})"


