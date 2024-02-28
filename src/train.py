from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, action_size, state_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
        self.action_size = action_size
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((state_size, action_size))
    
    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        if use_random or np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[observation])

    def train(self, state, action, reward, next_state, done):
        target = reward + self.discount_factor * np.max(self.q_table[next_state]) * (not done)
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
        if done:
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def save(self, path: str) -> None:
        np.save(path, self.q_table)

    def load(self, path: str = 'agent_model.npy') -> None:
        if os.path.exists(path):
            self.q_table = np.load(path)
        else:
            print(f"File {path} not found. Unable to load model.")
