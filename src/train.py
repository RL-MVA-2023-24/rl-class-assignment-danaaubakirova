from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from sklearn.ensemble import RandomForestRegressor
import torch
import os
import random
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import joblib
import numpy as np
env = TimeLimit(
 env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

#from sklearn.ensemble import RandomForestRegressor
#import numpy as np
#import random
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 256),  # Input layer size matches observation space dimension
            nn.ReLU(),  # Activation function
            nn.Linear(256, 128),  # First hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(128, 64),  # Second hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(64, 4)  # Output layer size matches action space dimension
        )
    
    def forward(self, x):
        return self.network(x)
class ProjectAgent:
    def __init__(self):
        self.nb_actions = 4  # Number of actions
        self.gamma = 0.95  # Discount factor
        self.iterations = 400  # Number of FQI iterations
        self.models = []  # To store models for each FQI iteration
        self.is_trained = False

        # Initialize the current DQN model
        self.current_model = DQN()
        self.optimizer = optim.Adam(self.current_model.parameters())
        self.criterion = nn.MSELoss()
        print("Agent initialized.")

    def collect_samples(self, env, horizon):
        print("Collecting samples...")
        S, A, R, S2, D = [], [], [], [], []
        s, _ = env.reset()
        for _ in range(horizon):
            a = self.act(s, use_random=True)
            s2, r, done, _, _ = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done:
                s, _ = env.reset()
                print("Episode finished. Resetting environment.")
            else:
                s = s2
        print("Sample collection complete.")
        return np.array(S), np.array(A).reshape((-1, 1)), np.array(R), np.array(S2), np.array(D)

    def train(self, env, horizon):
        print("Starting training...")
        S, A, R, S2, D = self.collect_samples(env, horizon)
        self.fqi(S, A, R, S2, D)
        print("Training completed.")

    def fqi(self, S, A, R, S2, D):
        for i in range(self.iterations):
            print(f"Training iteration {i+1}/{self.iterations}...")
            S_tensor = torch.FloatTensor(S)
            A_tensor = torch.LongTensor(A)  # A_tensor is already in the correct shape, no need to unsqueeze twice
            R_tensor = torch.FloatTensor(R)
            S2_tensor = torch.FloatTensor(S2)
            D_tensor = torch.FloatTensor(D)

            with torch.no_grad():
                Q_next = self.current_model(S2_tensor).detach().max(1)[0]

            target_Q = R_tensor + self.gamma * (1 - D_tensor) * Q_next

            Q_pred = self.current_model(S_tensor)
            Q_current = Q_pred.gather(1, A_tensor)  # Use A_tensor directly without extra unsqueeze

            loss = self.criterion(Q_current, target_Q.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.is_trained = True
        print("All iterations completed.")


    def act(self, observation, use_random=False):
        if use_random or not self.is_trained:
            return np.random.randint(0, self.nb_actions)
        else:
            return self.greedy_action(observation)

    def greedy_action(self, s):
        s_tensor = torch.FloatTensor(s).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            Q_values = self.current_model(s_tensor)
        return Q_values.max(1)[1].item()  # Return the action with the highest Q-value

    def save(self, path):
        """Save the current model to a file."""
        print(f"Saving model to {path}...")
        torch.save(self.current_model.state_dict(), path)
        print("Model saved successfully.")

    def load(self,):
        """Load a model from a file."""
        
        path = '/home/runner/work/rl-class-assignment-danaaubakirova/rl-class-assignment-danaaubakirova/src/model.pth'
        if os.path.isfile(path):
            print(f"Loading model from {path}...")
            self.current_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            self.current_model.eval()
            self.is_trained = True
            print("Model loaded successfully and ready for use.")
        else:
            print(f"No model file found at '{path}'.")

# Initialize yo

# Example usage
#agent = ProjectAgent()
#agent.train(env, horizon=10000)
#model_save_path = 'model.pth'

# Save the trained Q-functions to the specified path
#agent.save(model_save_path)
#print(f"Model saved to {model_save_path}")