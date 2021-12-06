import os
import gym
import copy
import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter

from base.net import ActorCritic
from base.replay import ReplayBuffer


class A2CAgent:
    """
    An Actor-Critic-based reinforcement learning algorithm, 
    using n-step Temporal Difference Estimator to calculate gradients.
    """
    def __init__(self,
                 obs_dim, 
                 action_dim, 
                 embedding_dim=64,
                 gamma=0.99,
                 lr_actor=1e-2,
                 lr_critic=1e-2,
                 lr_decay=0.99,
                 max_grad_norm=0.5,
                 log_dir='logs/a2c',
                 save_dir='save/a2c',
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
        self.writer = SummaryWriter(log_dir)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: lr_decay ** epoch)
        self.buffer = ReplayBuffer()

        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.criterion_cirtic = nn.MSELoss()

        if not os.path.exists(save_dir): 
            os.mkdir(save_dir)
        self.save_dir = save_dir

        self.norm_reward = norm_reward
        self.norm_advantage = norm_advantage
        self.clip_grad = clip_grad
        self.open_tb = open_tb
        self.open_tqdm = open_tqdm
        self.verbose = verbose  # if opening tqdm, suggest setting verbose == False
        
        self.update_time = 0

    def preprocess_obs(self, obs):
        return torch.FloatTensor(obs).to(self.device).unsqueeze(0)

    def select_action(self, obs, sample=True):
        action_logits = self.policy.act(obs)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        if sample:
            action = dist.sample()
        else:
            action = action_probs.argmax(-1, keepdim=True)
        action_logprob = dist.log_prob(action)
        # collect
        agent.buffer.actions.append(action)
        agent.buffer.action_logprobs.append(action_logprob)
        return action.item()

    def estimate_obs(self, obs):
        value = self.policy.estimate(obs).squeeze(-1)
        return value

    def update(self, next_value):
        action_logprobs = torch.cat(self.buffer.action_logprobs, dim=-1)
        masks = torch.IntTensor(self.buffer.masks).to(self.device)
        rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        self.buffer.values.append(next_value)
        values = torch.cat(self.buffer.values, dim=-1)
        # calculate expected return (n-step Temporal Difference Estimator)
        returns = torch.zeros_like(rewards).to(self.device)
        for i in reversed(range(len(rewards))):
            pre_return = next_value.detach() if i == len(rewards)-1 else returns[i + 1]
            returns[i] = rewards[i] + self.gamma * pre_return * masks[i]
        if self.norm_reward:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        # calculate advantage
        advantage = returns - values[:-1].detach()
        if self.norm_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)
        # calculate loss = actor_loss + critic_loss
        # actor_loss = - (advantage * action_log_prob)
        # critic_loss = MSE(returns, values)
        actor_loss = - (advantage * action_logprobs).mean()
        critic_loss = self.criterion_cirtic(returns, values[:-1])
        loss = actor_loss + 0.5 * critic_loss

        # update parameters
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            actor_grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            critic_grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()

        # log info
        if self.open_tb:
            self.writer.add_scalar('loss/loss', loss, self.update_time)
            self.writer.add_scalar('loss/actor_loss', actor_loss, self.update_time)
            self.writer.add_scalar('loss/critic_loss', critic_loss, self.update_time)
            self.writer.add_scalar('value/values', values[:-1].mean(), self.update_time)
            self.writer.add_scalar('value/returns', returns.mean(), self.update_time)
            self.writer.add_scalar('value/rewards', rewards.mean(), self.update_time)
            self.writer.add_scalar('grad/actor_grad_clipped', actor_grad_clipped, self.update_time)
            self.writer.add_scalar('grad/critic_grad_clipped', critic_grad_clipped, self.update_time)

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


def train(env, agent, batch_size=64, num_epochs=100, start_epoch=0, max_step=200, render=False):
    agent.train()
    cumulative_rewards = []
    pbar = tqdm.tqdm(desc='epoch', total=num_epochs) if agent.open_tqdm else None
    for epoch_idx in range(start_epoch, start_epoch + num_epochs):
        one_epoch_rewards = []
        obs = env.reset()
        for step_idx in range(max_step):
            env.render() if render else None
            action = agent.select_action(agent.preprocess_obs(obs))
            value = agent.estimate_obs(agent.preprocess_obs(obs))
            next_obs, reward, done, info = env.step(action)
            # collect experience
            agent.buffer.values.append(value)
            agent.buffer.rewards.append(reward)
            agent.buffer.masks.append(not done)
            one_epoch_rewards.append(reward)
            # obs transition
            obs = next_obs
            # update model
            if agent.buffer.size() == batch_size:
                next_value = agent.estimate_obs(agent.preprocess_obs(obs))
                agent.update(next_value)
            # episode done
            if done:
                cumulative_rewards.append(sum(one_epoch_rewards))
                print(f'epoch {epoch_idx:3d} | cumulative reward (max): {cumulative_rewards[-1]:4.1f} ' + 
                    f'({max(cumulative_rewards):4.1f})')
                break
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
        one_epoch_rewards = []
        obs = env.reset()
        for step_idx in range(max_step):
            env.render() if render else None
            action = agent.select_action(agent.preprocess_obs(obs), sample=False)
            next_obs, reward, done, info = env.step(action)
            one_epoch_rewards.append(reward)
            # obs transition
            obs = next_obs
            # episode done
            if done:
                cumulative_rewards.append(sum(one_epoch_rewards))
                print(f'epoch {epoch_idx:3d} | cumulative reward (max): {cumulative_rewards[-1]:4.1f} ' + 
                    f'({max(cumulative_rewards):4.1f})')
                break
    env.close()


if __name__ == '__main__':
    # config
    env_name = 'CartPole-v0'
    embedding_dim=64
    num_epochs = 1000
    start_epoch=0
    batch_size = 64
    max_step=200
    
    # initialize
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = A2CAgent(obs_dim=obs_dim, action_dim=action_dim, embedding_dim=embedding_dim)

    # train
    train(env, agent, batch_size=batch_size, num_epochs=num_epochs, start_epoch=start_epoch, 
        max_step=max_step, render=False)

    # test
    checkpoint_path = f'save/a2c/model-{num_epochs-1}.pkl'
    evaluate(env, agent, checkpoint_path, num_epochs=10, max_step=max_step, render=True)
