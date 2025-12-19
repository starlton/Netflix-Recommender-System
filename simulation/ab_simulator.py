import random

class ABSimulator:
    """
    Simulates A/B testing with epsilon-greedy exploration
    """
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.rewards = {"A": [], "B": []}

    def choose_model(self):
        if random.random() < self.epsilon:
            return random.choice(["A", "B"])
        return "A" if self.avg("A") >= self.avg("B") else "B"

    def update(self, model, reward):
        self.rewards[model].append(reward)

    def avg(self, model):
        r = self.rewards[model]
        return sum(r) / len(r) if r else 0.0
