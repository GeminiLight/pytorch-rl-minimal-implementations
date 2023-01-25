import os
import abc
import copy
import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter

from .net import ActorCritic
from .replay import ReplayBuffer


class Agent:
    """
    Abstract class of RL agent
    """
    def __init__(self,
                 embedding_dim=64,
                 gamma=0.99,
                 lr_actor=1e-3,
                 lr_critic=3e-3,
                 lr_decay=0.99,
                 coef_critic_loss=0.5,
                 coef_entropy_loss=0.01,
                 max_grad_norm=0.5,
                 log_dir='logs/algo',
                 save_dir='save/algo',
                 use_cuda=True,
                 norm_advantage=True,
                 norm_critic_loss=False,
                 clip_grad=True,
                 open_tb=True,
                 open_tqdm=False,
                 verbose=1):
        self.device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device.type}\n')
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_decay = lr_decay
        self,lr_scheduler = None

        self.coef_critic_loss = coef_critic_loss
        self.coef_entropy_loss = coef_entropy_loss
        self.max_grad_norm = max_grad_norm
        self.embedding_dim = embedding_dim

        self.log_dir = log_dir
        self.save_dir = save_dir

        self.writer = SummaryWriter(log_dir) if open_tb else None
        self.buffer = ReplayBuffer()
        self.criterion_cirtic = nn.MSELoss()
    
        if not os.path.exists(save_dir): 
            os.mkdirs(save_dir)
        self.update_time = 0
        
        self.norm_advantage = norm_advantage
        self.norm_critic_loss = norm_critic_loss
        self.clip_grad = clip_grad
        self.open_tb = open_tb
        self.open_tqdm = open_tqdm
        self.verbose = verbose

    @abc.abstractmethod
    def preprocess_obs(self, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, obs, sample=True, mask=None, hidden=None):
        with torch.no_grad():
            action_logits = self.policy.act(obs)
            
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        if mask is not None:
            masked_action_logits = action_logits + mask.log()
            masked_action_probs = F.softmax(masked_action_logits, dim=-1)
            masked_dist = Categorical(masked_action_probs)
        else:
            masked_action_probs = action_probs
            masked_dist = dist

        if sample:
            action = masked_dist.sample()
        else:
            action = masked_action_probs.argmax(-1)
            
        return action.item()

    @abc.abstractmethod
    def estimate_obs(self, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_actions(self, old_observations, old_actions):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, next_obs):
        raise NotImplementedError

    def train(self, mode=True):
        self.policy.train(mode)

    def eval(self):
        self.policy.eval()

    def save(self, fname='model', epoch_idx=0):
        checkpoint_path = os.path.join(self.save_dir, f'{fname}-{epoch_idx}.pkl')
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            }, checkpoint_path)
        print(f'Save checkpoint to {checkpoint_path}\n')

    def load(self, checkpoint_path):
        try:
            print(f'Loading checkpoint from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict']) if self.lr_scheduler is not None else None
            print(f'Load successfully!\n')
        except:
            print(f'Load failed! Initialize parameters randomly\n')



    def estimate_obs(self, obs):
        value = self.policy.estimate(obs)
        return value.squeeze(-1)

    def evaluate_actions(self, old_observations, old_actions):
        actions_logits = self.policy.act(old_observations)
        actions_probs = F.softmax(actions_logits, dim=-1)
        dist = Categorical(actions_probs)
        action_logprobs = dist.log_prob(old_actions)
        dist_entropy = dist.entropy()

        value = self.policy.estimate(old_observations).squeeze(-1)
        return action_logprobs, dist_entropy, value

    def update(self, next_obs):
        old_actions = torch.cat(self.buffer.actions, dim=0)
        old_action_logprobs = torch.cat(self.buffer.action_logprobs, dim=0)
        old_observations = torch.cat(self.buffer.observations, dim=0)
        masks = torch.IntTensor(self.buffer.masks).to(self.device)
        rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        observations = copy.deepcopy(self.buffer.observations)
        observations.append(next_obs)
        observations = torch.cat(observations, dim=0)

        for i in range(self.K_epochs):
            # evaluate actions and observations
            action_logprobs, dist_entropy, values = self.evaluate_actions(old_observations, old_actions)
            values = self.estimate_obs(observations)
            # calculate expected return (Genralized Advantage Estimator)
            returns = torch.zeros_like(rewards).to(self.device)
            last_gae = 0
            for i in reversed(range(self.buffer.size())):
                delta = rewards[i] + self.gamma * (values[i + 1].detach() - values[i].detach()) * masks[i]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae * masks[i]
                returns[i] = last_gae + values[i].detach() * masks[i]
            # calculate advantage
            advantage = returns - values[:-1].detach()
            if self.norm_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)
            # calculate loss = actor_loss + critic_loss + entropy_loss
            # actor_loss = - (advantage * action_logprob)
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()
            # critic_loss = MSE(returns, values)
            benchmark_critic_loss = (returns.std() + 1e-9) if self.norm_critic_loss else 1.
            critic_loss = self.criterion_cirtic(returns, values[:-1]) / benchmark_critic_loss
            # entropy_loss = Entropy(prob)
            entropy_loss = dist_entropy.mean()
            loss = actor_loss + self.coef_critic_loss * critic_loss - self.coef_entropy_loss * entropy_loss
            # update parameters 
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                actor_grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                critic_grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
            if self.open_tb and self.update_time % self.log_interval == 0:
                self.writer.add_scalar('train_lr', self.optimizer.defaults['lr'], self.update_time)
                self.writer.add_scalar('train_loss/loss', loss, self.update_time)
                self.writer.add_scalar('train_loss/actor_loss', actor_loss, self.update_time)
                self.writer.add_scalar('train_loss/critic_loss', critic_loss, self.update_time)
                self.writer.add_scalar('train_loss/entropy_loss', entropy_loss, self.update_time)
                self.writer.add_scalar('train_value/values', values[:-1].mean(), self.update_time)
                self.writer.add_scalar('train_value/returns', returns.mean(), self.update_time)
                self.writer.add_scalar('train_value/rewards', rewards.mean(), self.update_time)
                self.writer.add_scalar('train_value/action_logprobs', action_logprobs.mean(), self.update_time)
                self.writer.add_scalar('train_grad/actor_grad_clipped', actor_grad_clipped, self.update_time)
                self.writer.add_scalar('train_grad/critic_grad_clipped', critic_grad_clipped, self.update_time)

            self.update_time += 1
        
        self.lr_scheduler.step()
        self.buffer.clear()

        return loss.detach()

