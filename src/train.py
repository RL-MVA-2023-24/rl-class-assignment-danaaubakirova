from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import os
import random
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np
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
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
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
        config = {'nb_actions': 4,
          'learning_rate': 0.0005,
          'gamma': 0.95,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 10000,
          'epsilon_delay_decay': 20,
          'batch_size': 20,
          'gradient_steps': 1,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 50,
          'update_target_tau': 0.005,
          'criterion':  torch.nn.MSELoss()}#torch.nn.SmoothL1Loss()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,self.device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
    
    def act(self, observation, use_random=False):
        if use_random:# or np.random.rand() <= self.epsilon:
            return np.random.randint(4)
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return int(torch.argmax(Q).item())

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  
    
    def train(self, env, max_episode):
        episode_return = []
        highest_score = float('-inf')  # Initialize the highest score as negative infinity
        best_episode = 0  # To keep track of which episode had the highest score
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act(state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                episode_return.append(episode_cum_reward)
                print(f"Episode {episode:3d}, Epsilon {epsilon:6.2f}, Batch Size {len(self.memory):5d}, Episode Return {episode_cum_reward:4.1f}")

                # Check if this episode's return is the highest so far
                if episode_cum_reward > highest_score:
                    highest_score = episode_cum_reward  # Update the highest score
                    best_episode = episode  # Update the best episode number
                    self.save(f"best_model_episode_{best_episode}.pth")  # Save the model as the best model

                episode_cum_reward = 0  # Reset cumulative reward for the next episode
                state, _ = env.reset()  # Reset the environment state for the next episode
            else:
                state = next_state

        print(f"Best Episode: {best_episode} with a score of {highest_score}")
        return episode_return
    
    def save(self, path):
        torch.save(self.target_model.state_dict(), path)

    def load(self, path='/home/runner/work/rl-class-assignment-danaaubakirova/rl-class-assignment-danaaubakirova/src/model.pth'): ##
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            self.model.eval()
        else:
            print("No model file found at '{}'".format(path))

#state_dim = env.observation_space.shape[0]
#n_action = env.action_space.n 
#agent = ProjectAgent()
#scores = agent.train(env, 200)
