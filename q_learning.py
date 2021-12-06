import os
import gym
import random
import pickle
import numpy as np
from collections import defaultdict


class Transform:
    def __init__(self, obs_low, obs_high, obs_dim=100) -> None:
        self.obs_low = obs_low
        self.obs_high = obs_high
        if isinstance(obs_dim, list):
            assert len(obs_dim) == len(self.obs_low)
            obs_dim = np.array(obs_dim)
        self.obs_dim = obs_dim
    
    @staticmethod
    def create_for_env(env, obs_dim=100):
        return Transform(obs_low=env.observation_space.low,
                   obs_high=env.observation_space.high,
                   obs_dim=obs_dim)

    def transform_obs(self, obs):
        r"""Transform the continuous action space to discrete action space"""
        transformed_obs = self.obs_dim * (obs - self.obs_low) / (self.obs_high - self.obs_low)
        return tuple(transformed_obs.astype(np.int32))


class QLearningAgent:
    r"""
    A value-based reinforcement learning algorithm,
    using Temporal Difference Estimator to update Q-table.

    1. Initialize the Q-tables for observation-action pairs
    2. while not STOP:
        2.1 Agent selects action (a) in the current observation (s) with the Q-table estimations
        2.2 Enviroment performs the action (a) and return the new observation (s') and reward (r)
        2.3 Agent updates the Q-table with
                Q(s, a) := Q(s, a) + \alpha [r + \gamma max(Q(s', a') - Q(s, a))]
            where
                \alpha: learning rate
                \gamma: gamma

    Epsilon-greedy: Exploration / Exploitation
    """
    def __init__(self, 
                 transform,
                 action_dim, 
                 gamma=0.95,
                 epsilon=0.99,
                 epsilon_decay=1-1e-5,
                 min_epsilon=0.01,
                 lr=0.5,
                 lr_decay=1-1e-5,
                 min_lr=0.1,
                 save_dir='save/q_learning',
                 open_tqdm=False,
                 verbose=True):
        self.transform = transform
        self.q_table = defaultdict(lambda: [random.random()] * action_dim)
        self.actions = list(range(action_dim))
        self.gamma = gamma
        self.lr = lr
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        if not os.path.exists(save_dir): 
            os.mkdir(save_dir)
        self.save_dir = save_dir
        self.open_tqdm = open_tqdm
        self.verbose = verbose

    def preprocess_obs(self, obs):
        r"""Transform the continuous action space to discrete action space"""
        return self.transform(obs)

    def select_action(self, obs, sample=True):
        obs_q_values = self.q_table[obs]
        # epsilon-greedy
        if not sample or random.random() > self.epsilon:
            action = obs_q_values.index(max(obs_q_values))
        else:
            action = random.choice(self.actions)
        return action

    def update(self, obs, action, reward, done, next_obs):
        obs = self.preprocess_obs(obs)
        if not done:
            next_obs = self.preprocess_obs(next_obs)
            max_next_obs_q_value = max(self.q_table[next_obs])
            temporal_difference = reward + self.gamma * max_next_obs_q_value - self.q_table[obs][action]
            self.q_table[obs][action] += self.lr * temporal_difference
        else:
            self.q_table[obs][action] = reward
        self.epsilon = max(self.epsilon_decay * self.epsilon, self.min_epsilon)
        self.lr = max(self.lr_decay * self.lr, self.min_lr)

    def train(self, mode=True):
        pass

    def eval(self):
        pass

    def save(self, fname='model', epoch_idx=0):
        fpath = os.path.join(self.save_dir, f'{fname}-{epoch_idx}.pickle')
        with open(fpath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
            print(f'Save model at {fpath}')

    def load(self, fpath):
        try:
            print(f'Loading checkpoint from {fpath}')
            with open(fpath, 'rb') as f:
                self.q_table = pickle.load(f)
                self.q_table = defaultdict(lambda: [0] * len(self.actions), self.q_table)
            print(f'Load successfully!\n')
        except:
            print(f'Load failed! Initialize parameters randomly\n')


def train(env, agent, num_epochs=1, start_epoch=0, max_step=200, render=False):
    agent.train()
    cumulative_rewards = []
    for epoch_idx in range(start_epoch, start_epoch + num_epochs):
        obs = env.reset()
        one_epoch_reward = []
        for i in range(max_step):
            env.render() if render else None
            action = agent.select_action(agent.preprocess_obs(obs))
            next_obs, reward, done, info = env.step(action)
            agent.update(obs, action, reward, done, next_obs)
            obs = next_obs
            one_epoch_reward.append(reward)
            if done:
                break
        cumulative_rewards.append(sum(one_epoch_reward))
        print(f'epoch: {epoch_idx}, cumulative_reward (max): {cumulative_rewards[-1]:5.1f}' +
                f'({max(cumulative_rewards):5.1f}), epsilon: {agent.epsilon:4.3f}, lr: {agent.lr:4.3f}')
    env.close()
    # save model
    agent.save(epoch_idx=epoch_idx)

def evaluate(env, agent, num_epochs=10, max_step=200, render=False):
    agent.load(f'./vanilla/q_learning/save/{env_name}.pickle')
    agent.eval()
    cumulative_rewards = []
    for epoch_idx in range(num_epochs):
        obs = env.reset()
        one_epoch_reward = []
        for i in range(max_step):
            env.render() if render else None
            action = agent.select_action(agent.preprocess_obs(obs))
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            one_epoch_reward.append(reward)
            if done:
                break
        cumulative_rewards.append(sum(one_epoch_reward))
        print(f'epoch: {epoch_idx}, cumulative_reward (max): {cumulative_rewards[-1]:5.1f}' +
                f'({max(cumulative_rewards):5.1f}), epsilon: {agent.epsilon:4.3f}, lr: {agent.lr:4.3f}')
    env.close()


if __name__ == '__main__':
    env_name = 'CartPole-v0'
    num_epochs = 10000
    start_epoch = 0
    max_step = 200
    obs_dim = [1, 1, 6, 6]
    lr = 0.5

    # initialize
    env = gym.make(env_name)
    action_dim = action_dim=env.action_space.n
    transform = Transform.create_for_env(env, obs_dim=obs_dim)
    agent = QLearningAgent(transform=transform.transform_obs, action_dim=action_dim, lr=lr)

    # train
    train(env, agent, num_epochs=num_epochs, start_epoch=start_epoch, max_step=max_step, render=False)
 
    # test
    evaluate(env, agent, num_epochs=10, max_step=max_step, render=True)