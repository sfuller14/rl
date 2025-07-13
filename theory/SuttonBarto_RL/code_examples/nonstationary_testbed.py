import numpy as np
import matplotlib.pyplot as plt

class NonStationaryBandit:
    def __init__(self, k=10, walk_std=0.01):
        self.k = k
        self.walk_std = walk_std
        # Initialize true values
        self.q_true = np.zeros(k)
        
    def step(self):
        """Take a random walk step for all action values"""
        self.q_true += np.random.normal(0, self.walk_std, self.k)
        
    def get_reward(self, action):
        """Get reward for action (true value + noise)"""
        return np.random.normal(self.q_true[action], 1.0)

class BanditAgent:
    def __init__(self, k=10, epsilon=0.1, alpha=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha  # If None, use sample average
        
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        
    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.Q)
    
    def update(self, action, reward):
        self.N[action] += 1
        if self.alpha is None:
            # Sample average
            step_size = 1.0 / self.N[action]
        else:
            # Constant step size
            step_size = self.alpha
            
        self.Q[action] += step_size * (reward - self.Q[action])

def run_nonstationary_experiment(steps=10000):
    """Compare sample-average vs constant step-size on nonstationary bandit"""
    
    # Create two bandits (same random walk)
    np.random.seed(42)  # For reproducibility
    bandit1 = NonStationaryBandit(k=10, walk_std=0.01)
    bandit2 = NonStationaryBandit(k=10, walk_std=0.01)
    
    # Create agents
    agent_sample = BanditAgent(k=10, epsilon=0.1, alpha=None)      # Sample average
    agent_constant = BanditAgent(k=10, epsilon=0.1, alpha=0.1)     # Constant α=0.1
    
    # Tracking
    rewards_sample = []
    rewards_constant = []
    optimal_sample = []
    optimal_constant = []
    
    for t in range(steps):
        # Update environment (random walk)
        bandit1.step()
        bandit2.step()
        
        # Agent 1: Sample average
        action1 = agent_sample.select_action()
        reward1 = bandit1.get_reward(action1)
        agent_sample.update(action1, reward1)
        rewards_sample.append(reward1)
        optimal_sample.append(action1 == np.argmax(bandit1.q_true))
        
        # Agent 2: Constant step-size
        action2 = agent_constant.select_action()
        reward2 = bandit2.get_reward(action2)
        agent_constant.update(action2, reward2)
        rewards_constant.append(reward2)
        optimal_constant.append(action2 == np.argmax(bandit2.q_true))
    
    return (rewards_sample, rewards_constant, 
            optimal_sample, optimal_constant)

def plot_results(rewards_sample, rewards_constant, optimal_sample, optimal_constant):
    """Plot comparison results"""
    window = 500  # Moving average window
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Average reward
    avg_reward_sample = np.convolve(rewards_sample, np.ones(window)/window, mode='valid')
    avg_reward_constant = np.convolve(rewards_constant, np.ones(window)/window, mode='valid')
    
    steps = np.arange(len(avg_reward_sample))
    ax1.plot(steps, avg_reward_sample, label='Sample Average', alpha=0.8)
    ax1.plot(steps, avg_reward_constant, label='Constant α=0.1', alpha=0.8)
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Nonstationary 10-Armed Bandit (Random Walk)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Optimal action percentage
    optimal_pct_sample = np.convolve(optimal_sample, np.ones(window)/window, mode='valid') * 100
    optimal_pct_constant = np.convolve(optimal_constant, np.ones(window)/window, mode='valid') * 100
    
    ax2.plot(steps, optimal_pct_sample, label='Sample Average', alpha=0.8)
    ax2.plot(steps, optimal_pct_constant, label='Constant α=0.1', alpha=0.8)
    ax2.set_ylabel('% Optimal Action')
    ax2.set_xlabel('Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final performance
    print(f"Final 1000 steps average reward:")
    print(f"  Sample Average: {np.mean(rewards_sample[-1000:]):.3f}")
    print(f"  Constant α=0.1: {np.mean(rewards_constant[-1000:]):.3f}")
    print(f"Final 1000 steps optimal action %:")
    print(f"  Sample Average: {np.mean(optimal_sample[-1000:])*100:.1f}%")
    print(f"  Constant α=0.1: {np.mean(optimal_constant[-1000:])*100:.1f}%")

# Run the experiment
if __name__ == "__main__":
    print("Running nonstationary bandit experiment...")
    print("This demonstrates why constant step-size outperforms sample-average")
    print("when the environment changes over time.\n")
    
    results = run_nonstationary_experiment(steps=10000)
    plot_results(*results)