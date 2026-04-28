"""MAB environment generator."""
import numpy as np
from typing import Dict, Any

def generate_bandit_environment(n_arms: int = 10, n_steps: int = 1000, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    arm_means = rng.uniform(0.3, 0.8, n_arms)
    arm_stds = rng.uniform(0.1, 0.3, n_arms)
    optimal_arm = np.argmax(arm_means)
    return {"n_arms": n_arms, "n_steps": n_steps, "arm_means": arm_means, "arm_stds": arm_stds, "optimal_arm": optimal_arm}

def simulate_bandit(strategy_fn, environment: Dict[str, Any], seed: int = 42):
    rng = np.random.RandomState(seed)
    n_arms = environment["n_arms"]
    n_steps = environment["n_steps"]
    arm_means = environment["arm_means"]
    arm_stds = environment["arm_stds"]
    rewards = []
    for step in range(n_steps):
        arm = strategy_fn(step, rewards)
        reward = rng.normal(arm_means[arm], arm_stds[arm])
        rewards.append(reward)
    return {"total_reward": sum(rewards), "avg_reward": np.mean(rewards), "regret": environment["arm_means"].max() * n_steps - sum(rewards)}
