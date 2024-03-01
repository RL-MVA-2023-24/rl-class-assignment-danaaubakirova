import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# Neural Network for approximating Q-values
class DQN(nn.Module):
    def __init__(self,):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 128),  # Adjust the layer sizes as needed
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

class ProjectAgent:
    def __init__(self):
        # Directly use the environment's properties to set state_size and action_siz
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Default values for other parameters
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Initialize the DQN model with the state_size and action_size
        self.model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()
        
    def act(self, observation, use_random=False):
        if use_random or np.random.rand() <= self.epsilon:
            return np.random.randint(4)
        observation = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(observation.to(self.device))
        return np.argmax(action_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = self.memory.sample(self.batch_size)
        #print(len(minibatch))
        for state, action, reward, next_state, done in zip(*minibatch):
            target = reward + (not done) * self.gamma * np.max(self.model(next_state).cpu().detach().numpy())
            target_f = self.model(state)
            target_f[0][action.long()] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path='model.pth'):
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        else:
            print("No model file found at '{}'".format(path))
            
# Initialize the agent without arguments
#agent = ProjectAgent()

# The training loop remains the same
#num_episodes = 100  # Total number of episodes to train
#save_every = 10 

#for episode in range(num_episodes):
#    state = env.reset()
#    state = np.reshape(state[0], [1, 6])
#    total_reward = 0

#    for step in range(200):  # Max steps per episode
#        action = agent.act(state)
#        next_state, reward, done, _, _ = env.step(action)
#        next_state = np.reshape(next_state, [1, 6])

#        agent.remember(state, action, reward, next_state, done)  # Store experience
#        state = next_state
#        total_reward += reward

#        if done:
#            break

#        agent.replay()  # Train the model with a batch of experiences

#    print(f"Episode: {episode+1}, Total Reward: {total_reward}")

#    if (episode + 1) % save_every == 0:
#        agent.save(f"model_episode_{episode+1}.pth")  # Save the model