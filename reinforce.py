import os
import gym
import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter

from common.net import Actor
from common.replay import ReplayBuffer


class ReinforceAgent:
    """
    A Policy-commond reinforcement learning algorithm, 
    using Monte Carlo Estimator to calculate gradients.
    """
    def __init__(self,
                 obs_dim, 
                 action_dim, 
                 embedding_dim=64,
                 gamma=0.99,
                 lr_policy=1e-2,
                 lr_decay=0.99,
                 max_grad_norm=0.5,
                 log_interval=10,
                 log_dir='logs/reinforce',
                 save_dir='save/reinforce',
                 use_cuda=True,
                 clip_grad=True,
                 open_tb=True,
                 open_tqdm=False,
                 verbose=True):
        self.device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device.type}\n')
        self.policy = Actor(obs_dim, action_dim, embedding_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr_policy},
        ])
        self.writer = SummaryWriter(log_dir)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: lr_decay ** epoch)
        self.buffer = ReplayBuffer()

        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval

        if not os.path.exists(save_dir): 
            os.mkdir(save_dir)
        self.save_dir = save_dir

        self.clip_grad = clip_grad
        self.open_tb = open_tb
        self.open_tqdm = open_tqdm
        self.verbose = verbose

        self.update_time = 0

    def preprocess_obs(self, obs):
        return torch.FloatTensor(obs).to(self.device).unsqueeze(0)

    def select_action(self, obs, sample=True):
        action_logits = self.policy(obs)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        if sample:
            action = dist.sample()
        else:
            action = action_probs.argmax(-1)
        action_logprob = dist.log_prob(action)
        # collect
        self.buffer.observations.append(obs)
        self.buffer.action_logprobs.append(action_logprob)
        self.buffer.actions.append(action)
        return action.item()

    def update(self):
        action_logprobs = torch.cat(self.buffer.action_logprobs, dim=-1)
        masks = torch.IntTensor(self.buffer.masks).to(self.device)
        rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        # calculate expected return (Monte Carlo Estimator)
        returns = torch.zeros_like(rewards).to(self.device)
        for i in reversed(self.buffer.size()):
            last_return = 0 if i == self.buffer.size() - 1 else returns[i + 1]
            returns[i] = rewards[i] + self.gamma * last_return * masks[i]
        # calculate policy loss = - (Q_value * action_log_prob)
        loss = - (returns * action_logprobs).mean()
        # update parameters
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad: 
            policy_grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        # log info
        if self.open_tb and self.update_time % self.log_interval == 0:
            self.writer.add_scalar('train_lr', self.optimizer.defaults['lr'], self.update_time)
            self.writer.add_scalar('train_loss/loss', loss, self.update_time)
            self.writer.add_scalar('train_value/returns', returns.mean(), self.update_time)
            self.writer.add_scalar('train_value/rewards', rewards.mean(), self.update_time)
            self.writer.add_scalar('train_grad/policy_grad_clipped', policy_grad_clipped, self.update_time)

        self.update_time += 1
        self.buffer.clear()

        return loss.detach()

    def train(self, mode=True):
        self.policy.train(mode=mode)

    def eval(self):
        self.policy.eval()

    def save(self, fname='model', epoch_idx=0):
        checkpoint_path = os.path.join(self.save_dir, f'{fname}-{epoch_idx}.pkl')
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
            }, checkpoint_path)
        print(f'Save checkpoint to {checkpoint_path}\n')

    def load(self, checkpoint_path):
        try:
            print(f'Loading checkpoint from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print(f'Load successfully!\n')
        except:
            print(f'Load failed! Initialize parameters randomly\n')


def train(env, agent, num_epochs=100, start_epoch=0, max_step=200, render=False):
    agent.train()
    cumulative_rewards = []
    pbar = tqdm.tqdm(desc='epoch', total=num_epochs) if agent.open_tqdm else None
    for epoch_idx in range(start_epoch, start_epoch + num_epochs):
        one_epoch_rewards = []
        obs, info = env.reset()
        for step_idx in range(max_step):
            env.render() if render else None
            action = agent.select_action(agent.preprocess_obs(obs))
            next_obs, reward, done, info = env.step(action)
            # collect experience
            agent.buffer.rewards.append(reward)
            agent.buffer.masks.append(not done)
            one_epoch_rewards.append(reward)
            # obs transition
            obs = next_obs
            # episode done
            if done:
                loss = agent.update()
                cumulative_rewards.append(sum(one_epoch_rewards))
                print(f'epoch {epoch_idx:3d} | cumulative reward (max): {cumulative_rewards[-1]:4.1f} ' + 
                    f'({max(cumulative_rewards):4.1f}), loss: {loss:2.4f}') if agent.verbose else None
        # save model
        if epoch_idx % 1000 == 0 or epoch_idx == num_epochs - 1:
            agent.save(epoch_idx=epoch_idx)

        pbar.update(1) if pbar is not None else None
    pbar.close() if pbar is not None else None
    env.close()

def evaluate(env, agent, checkpoint_path, num_epochs=10, max_step=200, render=False):
    agent.load(checkpoint_path)
    agent.eval()
    cumulative_rewards = []
    for epoch_idx in range(num_epochs):
        rewards = []
        obs, info = env.reset()
        for step_idx in range(max_step):
            env.render() if render else None
            action = agent.select_action(agent.preprocess_obs(obs), sample=False)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            rewards.append(reward)
            # episode done
            if done:
                cumulative_rewards.append(sum(rewards))
                print(f'epoch {epoch_idx:3d} | cumulative reward (max): {cumulative_rewards[-1]:4.1f} ' + 
                    f'({max(cumulative_rewards):4.1f})')
                break
    env.close()


if __name__ == '__main__':
    # config
    env_name = 'CartPole-v0'
    num_epochs = 500
    embedding_dim = 64
    start_epoch = 0
    max_step = 200

    # initialize
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ReinforceAgent(obs_dim=obs_dim, action_dim=action_dim, embedding_dim=embedding_dim)

    # train
    train(env, agent, num_epochs=num_epochs, start_epoch=start_epoch, max_step=max_step, render=False)

    # test
    checkpoint_path = f'save/reinforce/model-{num_epochs-1}.pkl'
    evaluate(env, agent, checkpoint_path, num_epochs=10, max_step=max_step, render=True)
