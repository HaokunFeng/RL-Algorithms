import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    """ Plot class for plotting the results of different algorithms. """
    def __init__(self, solvers, solver_names):
        self.solvers = solvers
        self.solver_names = solver_names
    
    def plot_results(self):
        for idx, solver in enumerate(self.solvers):
            time_list = np.arange(len(solver.regrets))
            plt.plot(time_list, solver.regrets, label=self.solver_names[idx])
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Regrets')
        plt.title('%d-armed bandit' % self.solvers[0].bandit.num_arms)
        plt.legend()
        plt.show()

class BernoulliBandit:
    """ Bernoulli Bandit class for simulating a Bernoulli bandit problem. num_arms means the number of arms. """
    def __init__(self, num_arms):
        self.probs = np.random.uniform(size=num_arms)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.num_arms = num_arms
    
    def step(self, k):
        """ Simulate pulling arm k. """
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

class Solver:
    """ Basic Frame of MAB Algorithm.""" 
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.num_arms)
        self.regret = 0.
        self.actions = []
        self.regrets = []
    
    def update_regret(self, idx_arm):
        """k is the index of the arm we choose to pull."""
        self.regret += self.bandit.best_prob - self.bandit.probs[idx_arm]
        self.regrets.append(self.regret)
    
    def run_one_step(self):
        """Run one step of the algorithm. This function should be implemented in the subclass."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def run(self, num_steps):
        """Run the algorithm for a given number of steps."""
        for _ in range(num_steps):
            idx_arm = self.run_one_step()
            self.counts[idx_arm] += 1
            self.actions.append(idx_arm)
            self.update_regret(idx_arm)

#--------------------------------------------------------------------------------------------------------------
# Epsilon Greedy Algorithm
class EpsilonGreedy(Solver):
    """ Epsilon-Greedy Algorithm. """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.num_arms)
        self.name = "Epsilon-Greedy"
    
    def run_one_step(self):
        if np.random.random() < self.epsilon:
            idx_arm = np.random.randint(0, self.bandit.num_arms)
        else:
            idx_arm = np.argmax(self.estimates)
        r = self.bandit.step(idx_arm)
        self.estimates[idx_arm] += (r - self.estimates[idx_arm]) / (self.counts[idx_arm] + 1)
        return idx_arm

class DecayingEpsilonGreedy(Solver):
    """ epsilon decay with time. """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.num_arms)
        self.total_count = 0
    
    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            idx_arm = np.random.randint(0, self.bandit.num_arms)
        else:
            idx_arm = np.argmax(self.estimates)
        
        r = self.bandit.step(idx_arm)
        self.estimates[idx_arm] += (r - self.estimates[idx_arm]) / (self.counts[idx_arm] + 1)
        return idx_arm

    
#--------------------------------------------------------------------------------------------------------------
# UCB Algorithm
class UCB(Solver):
    """ Upper Confidence Bound Algorithm. """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.num_arms)
        self.coef = coef
        self.name = "UCB"
    
    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        idx_arm = np.argmax(ucb)
        r = self.bandit.step(idx_arm)
        self.estimates[idx_arm] += (r - self.estimates[idx_arm]) / (self.counts[idx_arm] + 1)
        return idx_arm

#--------------------------------------------------------------------------------------------------------------
# Thompson Sampling Algorithm
class ThompsonSampling(Solver):
    """ Thompson Sampling Algorithm. """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.num_arms)
        self._b = np.ones(self.bandit.num_arms)
    
    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        idx_arm = np.argmax(samples)
        r = self.bandit.step(idx_arm)

        self._a[idx_arm] += r
        self._b[idx_arm] += (1 - r)
        return idx_arm