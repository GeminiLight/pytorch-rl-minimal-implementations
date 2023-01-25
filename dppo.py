import os
import gym
import time
import copy
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data import Batch

from buffer import ReplayBuffer
from common.net import ActorCritic

class commonAgent:
    def __init__(self, 
                 gamma=0.99, 
                 gae_lambda=0.98,
                ):
        self.device = torch.device('cpu')
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer = ReplayBuffer(gamma=gamma, gae_lambda=gae_lambda)

    def preprocess_one_obs(self, obs):
        return torch.FloatTensor(obs).to(self.device).unsqueeze(0)

    def select_action(self, obs, sample=True, mask=None):
        with torch.no_grad():
            action_logits = self.policy.act(obs)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            if sample:
                action = dist.sample()
            else:
                action = action_probs.argmax(-1)
            action_logprob = dist.log_prob(action)
            return action.item(), action_logprob.item()

    def estimate_obs(self, obs):
        with torch.no_grad():
            value = self.policy.estimate(obs)
            return value.squeeze(-1).item()

    def train(self, mode=True):
        self.policy.train(mode=mode)

    def eval(self):
        self.policy.eval()


class Worker(mp.Process, commonAgent):

    def __init__(self, rank, lock, policy, env, config, experience_queue, parameters_dict):
        mp.Process.__init__(self, name=f'worker-{rank:02d}')
        commonAgent.__init__(self, config['gamma'], config['gae_lambda'])
        self.rank = rank
        self.lock = lock
        self.target_step = config['target_step']
        self.policy = copy.deepcopy(policy)
        self.env = copy.deepcopy(env)
        self.parameters_dict = parameters_dict
        self.experience_queue = experience_queue

        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)


    def run(self):
        obs = self.env.reset()
        self._last_obs = obs
        while True:
            # receive parameters
            while self.rank not in self.parameters_dict.keys():
                time.sleep(0.0001)
            parameters = self.parameters_dict[self.rank]
            self.policy.load_state_dict(parameters)
            del self.parameters_dict[self.rank]

            # collect experience
            self.buffer.reset()
            self.explore_env(self.env)
            self.experience_queue.put((self.rank, self.buffer))

    def explore_env(self, env):
        total_step = 0
        episode_steps = 0
        buffer = ReplayBuffer(self.gamma, self.gae_lambda)
        while True:
            total_step += 1
            episode_steps += 1

            # select action
            obs_tensor = self.preprocess_one_obs(self._last_obs)
            action, log_prob = self.select_action(obs_tensor)
            value = self.estimate_obs(obs_tensor)
            # state transition
            next_obs, raward, terminated, truncated, info = env.step(action)
            # collect experience
            buffer.add(copy.deepcopy(self._last_obs), action, raward, terminated or truncated, log_prob, value)
            # episode done
            timeout = (total_step == self.target_step)
            terminal = timeout or terminated or truncated

            if terminal:
                if terminated or truncated:
                    last_value = 0
                    next_obs, info = env.reset()
                elif timeout:
                    obs_tensor = self.preprocess_one_obs(next_obs)
                    last_value = self.estimate_obs(obs_tensor)
                buffer.compute_returns_and_advantages(last_value=last_value)
                self.buffer.merge(buffer)
                # restart
                buffer.reset()
                episode_steps = 0

            self._last_obs = next_obs

            if total_step == self.target_step:
                break


class Master(commonAgent):

    def __init__(self,
                obs_dim, action_dim, embedding_dim,
                 num_workers=4,
                 gamma=0.99,
                 gae_lambda=0.98,
                 eps_clip=0.2,
                 lr_actor=1e-3,
                 lr_critic=1e-3,
                 coef_value_loss=0.5,
                 coef_entropy_loss=0.00,
                 max_grad_norm=0.5,
                 log_dir='logs',
                 save_dir='save/dppo',
                 norm_advantage=True,
                 clip_grad=True,
                 verbose=1,
                 norm_value_loss=True,
                 use_cuda=True,
                 open_tb=True):
        super(Master, self).__init__(gamma, gae_lambda)
        self.num_workers = num_workers
        self.device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.policy = ActorCritic(obs_dim, action_dim, embedding_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
        ])
        if not os.path.exists(save_dir): 
            os.mkdir(save_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.save_dir = save_dir
        self.verbose = verbose

        self.eps_clip = eps_clip
        self.coef_value_loss = coef_value_loss
        self.coef_entropy_loss = coef_entropy_loss
        self.max_grad_norm = max_grad_norm
        self.norm_advantage = norm_advantage
        self.clip_grad = clip_grad
        self.norm_value_loss= norm_value_loss
        self.target_step = 1024
        self.batch_size = 64
        self.repeat_times = 10
        self.log_interval = 10
        self.save_interval = 2
        self.open_tb = open_tb


    def update_workers(self):
        if self.device == torch.device('cuda'): self.policy.cpu()
            
        for p_id in range(self.num_workers):
            self.parameters_dict[p_id] = self.policy.state_dict()
            
        if self.device == torch.device('cuda'): self.policy.cuda()

    def learn(self, env, total_step):
        self.manager = mp.Manager()
        self.experience_queue = self.manager.Queue(self.num_workers)
        self.parameters_dict = self.manager.dict()
        lock = mp.Lock()
        config = {
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'target_step': int(self.target_step / self.num_workers)
        }

        self.policy.cpu().share_memory()
        self.processes = []
        for i in range(self.num_workers):
            p = Worker(i, lock, self.policy, env, config, self.experience_queue, self.parameters_dict)
            self.processes.append(p)
        # self.policy.cuda()

        for i in range(self.num_workers):
            self.processes[i].start()

        self.update_workers()

        self.step_count = 0
        iteration = 0
        self.update_time = 0
        while self.step_count < total_step:

            # one epoch
            self.pull_experience()
            self.update_net()
            self.update_workers()
            iteration += 1

            if iteration % self.save_interval == 0 and not iteration == 0:
                print(f"Saved model at iteration: {iteration}")
                # save model
                filename = self.save_dir + f"/iteration{int(iteration)}.pkl"
                torch.save({'policy':self.policy.state_dict(),
                            'optimizer':self.optimizer.state_dict(),
                            'steps':self.step_count,
                            'iterations': iteration
                            }, filename)

        self.writer.close()
        for i in range(self.num_workers):
            self.processes[i].terminate()
        for i in range(self.num_workers):
            self.processes[i].join()

    def pull_experience(self):
        # observations, actions, returns, advantage
        worker_buffers = [None] * self.num_workers
        for _ in range(self.num_workers):
            rank, buffer = self.experience_queue.get()
            worker_buffers[rank] = buffer
        
        for i in range(self.num_workers):
            self.buffer.merge(worker_buffers[i])

    def preprocess_obs(self, observations):
        
        observations = torch.FloatTensor(np.array(observations)).to(self.device)
        return observations

    def update_net(self):
        losses, policy_losses, value_losses, entropy_losses = [], [], [], []

        buffer_size = self.buffer.size()

        buffer_observations = self.preprocess_obs(self.buffer.observations)
        buffer_actions = torch.tensor(self.buffer.actions).to(self.device)
        buffer_old_log_probs = torch.tensor(self.buffer.log_probs).to(self.device)
        buffer_returns = torch.tensor(self.buffer.returns).to(self.device)
        buffer_advantages = torch.tensor(self.buffer.advantages).to(self.device)
        
        update_times = int(buffer_size / self.batch_size * self.repeat_times)
        # perform mini-batch updates
        for i_update in range(update_times):
            # shuffle individual transitions:
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False, device=self.device)
            observations = buffer_observations[indices]
            actions = buffer_actions[indices]
            returns = buffer_returns[indices]
            advantages = buffer_advantages[indices]
            old_log_probs = buffer_old_log_probs[indices]
            # update policy
            values, log_probs, dist_entropy = self.evaluate_actions(observations, actions)
            # advantages normalization
            if self.norm_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
            # calculate loss = policy_loss + value_loss + entropy_loss
            # policy_loss = - (advantage * action_logprob)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * advantages
            policy_loss = - torch.min(surr1, surr2).mean()
            policy_losses.append(policy_loss.item())
            # value_loss = MSE(returns, values)
            benchmark_value_loss = (returns.std() + 1e-9) if self.norm_value_loss else 1.
            value_loss = F.mse_loss(returns, values) / benchmark_value_loss
            value_losses.append(value_loss.item())
            # entropy_loss = Entropy(prob)
            entropy_loss = dist_entropy.mean()
            entropy_losses.append(entropy_loss.item())
            # loss
            loss = policy_loss + self.coef_value_loss * value_loss + self.coef_entropy_loss * entropy_loss
            losses.append(loss.item())
            # update parameters 
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                grad_clipped = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.update_time += 1

        print(f'loss: {np.mean(losses)}, policy_loss: {np.mean(policy_losses)}, value_loss: {np.mean(value_losses)}, entropy_loss: {np.mean(entropy_losses)}')
        self.step_count += self.buffer.size()
        self.buffer.reset()

        # log
        if self.open_tb and self.update_time % self.log_interval == 0:
            self.writer.add_scalar('train_lr', self.optimizer.defaults['lr'], self.update_time)
            self.writer.add_scalar('train_loss/loss', loss, self.update_time)
            self.writer.add_scalar('train_loss/policy_loss', policy_loss, self.update_time)
            self.writer.add_scalar('train_loss/value_loss', value_loss, self.update_time)
            self.writer.add_scalar('train_loss/entropy_loss', entropy_loss, self.update_time)
            self.writer.add_scalar('train_value/values', values[:-1].mean(), self.update_time)
            self.writer.add_scalar('train_value/returns', returns.mean(), self.update_time)
            self.writer.add_scalar('train_value/action_logprobs', log_probs.mean(), self.update_time)
            self.writer.add_scalar('train_grad/grad_clipped', grad_clipped, self.update_time)

    def estimate_obs(self, obs):
        value = self.policy.estimate(obs)
        return value.squeeze(-1)

    def evaluate_actions(self, old_observations, old_actions):
        values = self.policy.estimate(old_observations).squeeze(-1)
        actions_logits = self.policy.act(old_observations)
        actions_probs = F.softmax(actions_logits, dim=-1)
        dist = Categorical(actions_probs)
        action_logprobs = dist.log_prob(old_actions)
        dist_entropy = dist.entropy()
        return values, action_logprobs, dist_entropy


if __name__ == '__main__':
    # config
    env_name = 'CartPole-v0'
    embedding_dim = 64
    num_epochs = 2000
    start_epoch = 0
    batch_size = 64
    max_step = 200

    # initialize
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Master(obs_dim=obs_dim, action_dim=action_dim, embedding_dim=embedding_dim, num_workers=1)
    s = time.time()
    agent.learn(env, 100000)
    e = time.time()
    print(e - s)

    # train
    # train(env, agent, batch_size=batch_size, num_epochs=num_epochs, start_epoch=start_epoch, 
    #     max_step=max_step, render=False)

    # test
    checkpoint_path = f'save/dppo/iteration8.pkl'
    checkpoint = torch.load(checkpoint_path)

    agent = Master(obs_dim=obs_dim, action_dim=action_dim, embedding_dim=embedding_dim)
    agent.policy.load_state_dict(checkpoint['policy'])

    for i in range(10):

        done = False
        obs, info = env.reset()
        step = 0
        r = 0
        while not done:
            step += 1
            action, _ = agent.select_action(agent.preprocess_one_obs(obs))
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = (terminated or truncated)
            r += reward
                
        print(f"Return: {r}")