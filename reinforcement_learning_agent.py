from collections import deque
import random

class ReinforcementLearningAgent:
    def __init__(self, state_space_size: int):
        self.state_space_size = state_space_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
    def remember(self, state: tuple, action: int, reward: float, next_state: tuple) -> None:
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state))

    def act