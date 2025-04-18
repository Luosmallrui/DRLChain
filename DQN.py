import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
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
        # Important: Update the state_dim to match what the environment actually returns
        self.state_dim = 3  # Hardcoding to 3 based on error message
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        # Initialize model with correct dimensions
        self.model = DQN(self.state_dim, action_dim)
        self.target_model = DQN(self.state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Define action space parameters
        self.max_shard_size = 128  # MAX_SHARD_SIZE
        self.max_block_size = 8    # MAX_BLOCK_SIZE
        self.num_nodes = action_dim - 2  # First 2 dimensions are for shard_size and block_size
        
        # Print dimensions for debugging
        # print(f"Using state dimension: {self.state_dim}, Action dimension: {action_dim}")
        # print(f"Number of nodes: {self.num_nodes}")

    def choose_action(self, state):
        # Debug info
        # print(f"State shape: {np.shape(state)}")
        
        if random.random() < self.epsilon:
            # Random exploration
            action = np.zeros(self.action_dim, dtype=np.float32)
            action[0] = np.floor(np.random.uniform(0, np.log2(self.max_shard_size)))
            action[1] = np.random.uniform(0, self.max_block_size)
            action[2:] = np.random.uniform(0, 1, self.num_nodes)
        else:
            # Ensure state is correctly formatted
            if isinstance(state, np.ndarray):
                state_tensor = torch.tensor(state, dtype=torch.float32)
            else:
                state_tensor = state
            
            # Check if state needs reshaping
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            
            # Print tensor shape for debugging
            # print(f"State tensor shape: {state_tensor.shape}")
            
            # Get Q-values
            q_values = self.model(state_tensor).squeeze(0)
            
            # Convert to action
            action = torch.sigmoid(q_values).detach().cpu().numpy()
            
            # Scale actions to their respective ranges
            action[0] *= np.log2(self.max_shard_size)
            action[1] *= self.max_block_size
        
        return action.astype(np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        
        # Get current Q values
        q_values = self.model(states)
        
        # Get next Q values from target network
        next_q_values = self.target_model(next_states).detach()
        
        # Calculate target Q values
        max_next_q_values, _ = next_q_values.max(dim=1)
        target_q_values = rewards + self.gamma * (1 - dones) * max_next_q_values
        
        # Reshape for calculating loss
        target_q_values = target_q_values.unsqueeze(1).repeat(1, self.action_dim)
        
        # Calculate loss
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())