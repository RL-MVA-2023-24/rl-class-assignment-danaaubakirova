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
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),  # Adjust the layer sizes as needed
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ProjectAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, observation, use_random=False):
        if use_random or np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        observation = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(observation)
        return np.argmax(action_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward + (not done) * self.gamma * np.max(self.model(torch.FloatTensor(next_state)).detach().numpy())
            target_f = self.model(torch.FloatTensor(state))
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(torch.FloatTensor(state)))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path='model_episode_100.pth'):
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        else:
            print("No model file found at '{}'".format(path))
            
#state_size = env.observation_space.shape[0]
#action_size = env.action_space.n

#agent = ProjectAgent(state_size=state_size, action_size=action_size)
#num_episodes = 100  # Total number of episodes to train
#save_every = 10 

#for episode in range(num_episodes):
    #state = env.reset()
    #state = np.reshape(state[0], [1, state_size])
    #total_reward = 0

    #for step in range(200):  # Max steps per episode
    #    action = agent.act(state)
    #    next_state, reward, done, _, _ = env.step(action)
    #    next_state = np.reshape(next_state, [1, state_size])

    #    agent.remember(state, action, reward, next_state, done)  # Store experience
    #    state = next_state
    #    total_reward += reward

    #    if done:
    #        break

    #    agent.replay()  # Train the model with a batch of experiences

    #print(f"Episode: {episode+1}, Total Reward: {total_reward}")

    #if (episode + 1) % save_every == 0:
    #    agent.save(f"model_episode_{episode+1}.pth")  # Save the model
        
