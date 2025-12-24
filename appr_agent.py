import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995 # Slower decay for stability
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=20000)
        
        # Device Check (MPS for Mac, else CPU)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Agent using device: {self.device}")
        
        # Networks
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
        return np.argmax(q_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Mini-batch sampling
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)
        
        # Q(s, a)
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q = r + gamma * max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(1)[0]
            expected_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Loss
        loss = self.criterion(q_value, expected_q)
        
        # Optimization Step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
