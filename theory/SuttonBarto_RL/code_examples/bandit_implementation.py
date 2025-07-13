import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedyBandit:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        # True action values (unknown to algorithm)
        self.q_true = np.random.normal(0, 1, k)
        
        # Algorithm's estimates and counts
        self.Q = np.zeros(k)  # Action value estimates
        self.N = np.zeros(k)  # Action selection counts
        
        # For tracking performance
        self.rewards = []
        self.optimal_actions = []
        
    def select_action(self):
        """ε-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)  # Explore
        else:
            return np.argmax(self.Q)  # Exploit (greedy)
    
    def get_reward(self, action):
        """Get reward for selected action (with noise)"""
        return np.random.normal(self.q_true[action], 1)
    
    def update_estimate(self, action, reward):
        """Incremental update using sample average"""
        self.N[action] += 1
        # Incremental update: Q(a) ← Q(a) + (1/N(a))[R - Q(a)]
        self.Q[action] = self.Q[action] + (1/self.N[action]) * (reward - self.Q[action])
    
    def step(self):
        """One step of the bandit algorithm"""
        action = self.select_action()
        reward = self.get_reward(action)
        self.update_estimate(action, reward)
        
        # Track performance
        self.rewards.append(reward)
        self.optimal_actions.append(action == np.argmax(self.q_true))
        
        return action, reward

# Run a simple experiment
def run_experiment(steps=1000, epsilon=0.1):
    bandit = EpsilonGreedyBandit(k=10, epsilon=epsilon)
    
    print(f"True action values: {bandit.q_true}")
    print(f"Optimal action: {np.argmax(bandit.q_true)} (value: {np.max(bandit.q_true):.3f})")
    print("\nRunning bandit algorithm...")
    
    for t in range(steps):
        action, reward = bandit.step()
        
        # Print some progress
        if (t + 1) % 200 == 0:
            avg_reward = np.mean(bandit.rewards[-200:])
            optimal_pct = np.mean(bandit.optimal_actions[-200:]) * 100
            print(f"Steps {t-199:4d}-{t+1:4d}: Avg reward = {avg_reward:.3f}, "
                  f"Optimal action = {optimal_pct:.1f}%")
    
    print(f"\nFinal Q estimates: {bandit.Q}")
    print(f"Action selection counts: {bandit.N}")
    print(f"Overall average reward: {np.mean(bandit.rewards):.3f}")
    print(f"Overall optimal action %: {np.mean(bandit.optimal_actions)*100:.1f}%")

# Run the experiment
if __name__ == "__main__":
    run_experiment(steps=1000, epsilon=0.1)