import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment
env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)

# Neural Network for approximating Q-values
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = zip(*random.sample(self.buffer, batch_size))
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.buffer)

class ProjectAgent:
    def __init__(self):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = ReplayBuffer(100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state)
            action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_size)
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        (states, actions, rewards, next_states, dones), _ = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q = self.model(states).gather(1, actions)
        max_next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

def train_agent(agent, episodes=500, save_interval=50):
    rewards = []
    for episode in range(1, episodes + 1):
        state = env.reset()
        state = np.reshape(state[0], [1, agent.state_size])
        episode_reward = 0
        for _ in range(200):
            action = agent.act(state)
            next_state, reward, done, _,_ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            agent.replay()
            if done:
                break
        rewards.append(episode_reward)
        if episode % save_interval == 0:
            agent.save(f'model_{episode}.pth')
            agent.update_target()
        print(f'Episode: {episode}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.2f}')
    return rewards

if __name__ == "__main__":
    agent = ProjectAgent()
    rewards = train_agent(agent)
