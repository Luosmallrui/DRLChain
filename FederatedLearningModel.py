import torch
import torch.nn as nn
import torch.optim as optim


class FederatedLearningModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FederatedLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_local_model(model, data_loader, epochs=1, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def federated_averaging(global_model, local_models):
    global_state_dict = global_model.state_dict()

    # 聚合各个节点的本地模型参数
    local_weights = [model.state_dict() for model in local_models]

    # 对每个参数进行平均
    for key in global_state_dict:
        global_state_dict[key] = torch.mean(torch.stack([local_weights[i][key] for i in range(len(local_weights))]),
                                            dim=0)

    global_model.load_state_dict(global_state_dict)

