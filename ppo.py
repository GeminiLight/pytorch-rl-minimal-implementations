import os
import gym
import copy
import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter

from common.net import ActorCritic
from common.replay import ReplayBuffer


class PPOAgent:
    """
    An Actor-Critic-commond reinforcement learning algorithm, 
    using Genralized Advantage Estimator to calculate gradients.
    """
    def __init__(self,
                 obs_dim, 
                 action_dim, 
                 embedding_dim=64,
                 gamma=0.995,
                 gae_lambda=0.95,
                 K_epochs=5,
                 lr_actor=3e-2,
                 lr_critic=3e-2,
                 lr_decay=0.99,
                 max_grad_norm=1.,
                 eps_clip=0.2,
                 coef_critic_loss=0.5,
                 coef_entropy_loss=0.01,
                 log_interval=10,
                 log_dir='logs/ppo',
                 save_dir='save/ppo',
                 norm_advantage=True,
                 norm_critic_loss=False,
                 clip_grad=True,
                 use_cuda=True,
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
        self.log_interval = log_interval

        self.coef_critic_loss = coef_critic_loss
        self.coef_entropy_loss = coef_entropy_loss

        if not os.path.exists(save_dir): 
            os.mkdir(save_dir)
        self.save_dir = save_dir
        
        self.norm_advantage = norm_advantage
        self.norm_critic_loss = norm_critic_loss
        self.clip_grad = clip_grad
        self.open_tb = open_tb
        self.open_tqdm = open_tqdm
        self.verbose = verbose

        self.update_time = 0

    def preprocess_obs(self, obs):
        return torch.FloatTensor(obs).to(self.device).unsqueeze(0)

    def select_action(self, obs, mask=None, sample=True):
        with torch.no_grad():
            action_logits = self.policy.act(obs)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        if sample:
            action = dist.sample()
        else:
            action = action_probs.argmax(-1)
        action_logprob = dist.log_prob(action)
        # collect
        self.buffer.observations.append(obs)
        self.buffer.actions.append(action)
        self.buffer.action_logprobs.append(action_logprob)
        return action.item()

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
        actions = torch.cat(self.buffer.actions, dim=0)
        old_action_logprobs = torch.cat(self.buffer.action_logprobs, dim=0)
        self.buffer.observations.append(next_obs)
        observations = torch.cat(self.buffer.observations, dim=0)
        masks = torch.IntTensor(self.buffer.masks).to(self.device)
        rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)

        for i in range(self.K_epochs):
            # evaluate actions and observations
            action_logprobs, dist_entropy, old_obs_values = self.evaluate_actions(observations[:-1], actions)
            values = self.estimate_obs(observations)
            # calculate expected return (Genralized Advantage Estimator)
            # returns = torch.zeros_like(rewards).to(self.device)
            all_advantage = torch.zeros(self.buffer.size() + 1)
            for i in reversed(range(self.buffer.size())):
                delta = rewards[i] + self.gamma * (values[i+1].detach() * masks[i]) - values[i]
                all_advantage[i] = delta + (self.gamma * self.gae_lambda * all_advantage[i + 1] * masks[i])
            returns = all_advantage[:-1] + values[:-1].squeeze(0)
            
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
        obs, info = env.reset()
        for step_idx in range(max_step):
            env.render() if render else None
            action = agent.select_action(agent.preprocess_obs(obs))
            next_obs, reward, terminated, truncated, info = env.step(action)
            # collect experience
            agent.buffer.rewards.append(reward)
            agent.buffer.masks.append(not (terminated or truncated))
            one_epoch_rewards.append(reward)
            # obs transition
            obs = next_obs
            # update model
            if agent.buffer.size() == batch_size:
                loss = agent.update(agent.preprocess_obs(next_obs))
                pbar.set_postfix(loss=loss.item()) if pbar is not None else None
            # episode done
            if terminated or truncated:
                cumulative_rewards.append(sum(one_epoch_rewards))
                print(f'epoch {epoch_idx:3d} | cumulative reward (max): {cumulative_rewards[-1]:4.1f} ' + 
                    f'({max(cumulative_rewards):4.1f})') if agent.verbose else None
                pbar.set_postfix(reward=cumulative_rewards[-1]) if pbar is not None else None
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
        obs, info = env.reset()
        for step_idx in range(max_step):
            env.render() if render else None
            action = agent.select_action(agent.preprocess_obs(obs), sample=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            one_epoch_rewards.append(reward)
            # obs transition
            obs = next_obs
            # episode done
            if terminated or truncated:
                cumulative_rewards.append(sum(one_epoch_rewards))
                print(f'epoch {epoch_idx:3d} | cumulative reward (max): {cumulative_rewards[-1]:4.1f} ' + 
                    f'({max(cumulative_rewards):4.1f})')
                break
    env.close()


if __name__ == '__main__':
    # config
    env_name = 'CartPole-v0'
    embedding_dim = 64
    num_epochs = 500
    start_epoch = 0
    batch_size = 64
    max_step = 200

    # initialize
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, embedding_dim=embedding_dim)

    # train
    train(env, agent, batch_size=batch_size, num_epochs=num_epochs, start_epoch=start_epoch, 
        max_step=max_step, render=False)

    # test
    checkpoint_path = f'save/ppo/model-{num_epochs-1}.pkl'
    evaluate(env, agent, checkpoint_path, num_epochs=10, max_step=max_step, render=True)
