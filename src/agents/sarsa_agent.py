import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """Simple feed-forward NN to approximate Q(s,a)"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.fc3(x)

class SARSAAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.1):
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_t)
            return q_values.argmax().item()

    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update using function approximation"""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0) if next_state is not None else None

        q_values = self.q_network(state_t)
        q_sa = q_values[0, action]

        if done:
            target = torch.tensor(reward, dtype=torch.float32)
        else:
            q_next = self.q_network(next_state_t)
            q_snext_anext = q_next[0, next_action]
            target = reward + self.gamma * q_snext_anext

        loss = (q_sa - target.detach())**2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
