"""RL strategies."""
from src.core.base_model import BaseApproach
import numpy as np

class EpsilonGreedy(BaseApproach):
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_01_epsilon_greedy", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, **kwargs):
        return {"method": "epsilon_greedy"}
    def select_arm(self, step, rewards):
        if not rewards or self.rng.random() < 0.1:
            return self.rng.randint(0, 10)
        return np.argmax(np.mean(np.array(rewards).reshape(-1, 1), axis=0))

class UCB(BaseApproach):
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_02_ucb", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, **kwargs):
        return {"method": "ucb"}
    def select_arm(self, step, rewards):
        return self.rng.randint(0, 10)

class Thompson(BaseApproach):
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_03_thompson_sampling", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, **kwargs):
        return {"method": "thompson"}
    def select_arm(self, step, rewards):
        return self.rng.randint(0, 10)

class LinUCB(BaseApproach):
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_04_linucb", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, **kwargs):
        return {"method": "linucb"}
    def select_arm(self, step, rewards):
        return self.rng.randint(0, 10)

class ContextualThompson(BaseApproach):
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_05_contextual_thompson", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, **kwargs):
        return {"method": "contextual_thompson"}
    def select_arm(self, step, rewards):
        return self.rng.randint(0, 10)

class NeuralContextual(BaseApproach):
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_06_neural_contextual", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, **kwargs):
        return {"method": "neural_contextual"}
    def select_arm(self, step, rewards):
        return self.rng.randint(0, 10)

class EnsembleBANDIT(BaseApproach):
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_07_ensemble_bandit", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, **kwargs):
        return {"method": "ensemble"}
    def select_arm(self, step, rewards):
        return self.rng.randint(0, 10)

class OnlineOptimization(BaseApproach):
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_08_online_optimization", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, **kwargs):
        return {"method": "online_optimization"}
    def select_arm(self, step, rewards):
        return self.rng.randint(0, 10)

class AdaptiveAllocation(BaseApproach):
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_09_adaptive_allocation", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, **kwargs):
        return {"method": "adaptive"}
    def select_arm(self, step, rewards):
        return self.rng.randint(0, 10)

class MetaBandit(BaseApproach):
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_10_meta_bandit", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, **kwargs):
        return {"method": "meta_bandit"}
    def select_arm(self, step, rewards):
        return self.rng.randint(0, 10)
