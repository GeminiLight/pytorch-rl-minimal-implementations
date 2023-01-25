import copy


class ReplayBuffer:
    
    def __init__(self, gamma, gae_lambda, max_size=None):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.curr_idx = 0
        self.max_size = max_size

        self.basic_items = ['observations', 'actions', 'rewards', 'dones', 'log_probs', 'values']
        self.calc_items = ['advantages', 'returns']
        self.extend_items = ['hiddens', 'masks']

        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        self.advantages = []
        self.returns = []

        self.hiddens = []
        self.masks = []
    
    def reset(self):
        for item in self.basic_items + self.calc_items + self.extend_items:
            item_list = getattr(self, item)
            del item_list[:]

    def size(self):
        return len(self.actions)

    def is_full(self):
        if self.max_size is None:
            return False
        return self.curr_idx == self.max_size

    def add(self, obs, action, raward, done, log_prob, value):

        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(raward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

        self.curr_idx += 1
    
    def merge(self, buffer):
        for item in self.basic_items + self.calc_items + self.extend_items:
            main_item_list = getattr(self, item)
            sub_item_list = getattr(buffer, item)
            main_item_list += copy.deepcopy(sub_item_list)
        # self.observations += copy.deepcopy(buffer.observation)
        # self.actions += copy.deepcopy(buffer.actions)
        # self.rewards += copy.deepcopy(buffer.rewards)
        # self.dones += copy.deepcopy(buffer.dones)
        # self.log_probs += copy.deepcopy(buffer.log_probs)
        # self.values += copy.deepcopy(buffer.values)
        # self.advantages += copy.deepcopy(buffer.advantages)
        # self.returns += copy.deepcopy(buffer.returns)
        # self.hiddens += copy.deepcopy(buffer.hiddens)

    def compute_returns_and_advantages(self, last_value) -> None:
        # calculate expected return (Genralized Advantage Estimator)
        buffer_size = self.size()
        self.returns = [0] * buffer_size
        self.advantages = [0] * buffer_size

        last_gae_lam = 0
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_values = last_value
            else:
                next_values = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
            self.returns[step] = self.advantages[step] + self.values[step]


if __name__ == '__main__':
    buffer = ReplayBuffer(1, 1)
    
    temp = [1, 2, 3]
    for i in range(10):
        buffer.temp = temp
        buffer.observations.append(buffer.temp)
        temp.append(i)

    print(buffer.observations)
