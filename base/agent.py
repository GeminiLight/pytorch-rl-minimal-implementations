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
                 obs_dim, 
                 action_dim, 
                 embedding_dim=64,
                 gamma=0.99,
                 gae_lambda=0.95,
                 K_epochs=3,
                 lr_actor=1e-3,
                 lr_critic=3e-3,
                 lr_decay=0.99,
                 max_grad_norm=0.5,
                 eps_clip=0.2,
                 log_dir='logs/algo',
                 save_dir='save/algo',
                 use_cuda=True,
                 norm_reward=True,
                 norm_advantage=True,
                 clip_grad=True,
                 open_tb=True,
                 open_tqdm=False,
                 verbose=True):
        self.device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device.type}\n')
        self.policy = ActorCritic(obs_dim, action_dim, embedding_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
        ])
        self.writer = SummaryWriter(log_dir) if open_tb else None
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: lr_decay ** epoch)
        self.buffer = ReplayBuffer()
        self.criterion_cirtic = nn.MSELoss()
    
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.max_grad_norm = max_grad_norm

        if not os.path.exists(save_dir): 
            os.mkdir(save_dir)
        self.save_dir = save_dir
        self.update_time = 0
        
        self.norm_reward = norm_reward
        self.norm_advantage = norm_advantage
        self.clip_grad = clip_grad
        self.open_tb = open_tb
        self.open_tqdm = open_tqdm
        self.verbose = verbose

    @abc.abstractmethod
    def preprocess_obs(self, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, obs, sample=True):
        raise NotImplementedError

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

    @abc.abstractmethod
    def save(self, fname='model', epoch_idx=0):
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, checkpoint_path):
        raise NotImplementedError