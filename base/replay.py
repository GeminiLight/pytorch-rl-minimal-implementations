import copy


class ReplayBuffer:

    def __init__(self):
        self.observations = []
        self.actions = []
        self.action_logprobs = []
        self.values = []
        self.rewards = []
        self.masks = []

    def append(self, obs, action, action_logprob, value, reward, is_terminal):
        self.observations.append(obs)
        self.actions.append(action)
        self.action_logprobs.append(action_logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.masks.append(is_terminal)

    def clear(self):
        del self.observations[:]
        del self.actions[:]
        del self.action_logprobs[:]
        del self.values[:]
        del self.rewards[:]
        del self.masks[:]

    def size(self):
        return len(self.rewards)

    def calc_returns(self, next_value=0, method='gae', gamma=0.99, gae_lambda=0.95):
        r"""Calculate expected returns
        
        1. mc:     Monte Carlo
        2. td:     Temproal Difference
        3. gae:    Generalized Advantage Estimator
        3. n_step: n-step Advantage
        """
        returns = [0] * len(self.rewards)
        extra_values = copy.deepcopy(self.values).append(next_value)
        if method == 'mc':
            for i in reversed(range(len(self.rewards))):
                last_return = 0 if i == len(self.rewards) -1 else returns[i + 1]
                returns[i] = self.rewards[i] + gamma * last_return * self.masks[i]
        elif method == 'td':
            returns = self.rewards + gamma * extra_values[1:] * self.masks
        elif method == 'gae':
            last_gae = 0
            for i in reversed(range(self.buffer.size())):
                last_value = next_value if i == len(self.rewards)-1 else returns[i + 1]
                delta = self.rewards[i] + gamma * last_value * self.masks[i] - self.values[i]
                last_gae = delta + gamma * gae_lambda * self.masks[i] * last_gae
                self.returns[i] = last_gae + self.values[i]
        elif method == 'n_step':
            for i in reversed(range(len(self.rewards))):
                last_return = next_value if i == len(self.rewards)-1 else returns[i + 1]
                returns[i] = self.rewards[i] + gamma * last_return * self.masks[i]
        else:
            raise NotImplementedError
        return returns