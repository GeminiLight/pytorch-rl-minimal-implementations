# PyTorch RL Minimal Implementations

There are implementations of some reinforcement learning algorithms, whose characteristics are as follow:

1. **Less packages-based**:  Only PyTorch and Gym, for building neural networks and testing algorithms' performance respectively,  are necessary to install.
2. **Independent implementation**: All RL algorithms are implemented in separate files, which facilitates to understand their processes and modify them to adapt to other tasks.
3. **Various expansion configurations**: It's convenient to configure various parameters and tools, such as reward normalization, advantage normalization, tensorboard, tqdm and so on.

## RL Algorithms List

| Name       | Type         | Paper                                                        | File                           |
| ---------- | ------------ | ------------------------------------------------------------ | ------------------------------ |
| Q-learning | Value-based  | Watkins et al. [Q-Learning](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf). *Machine Learning*, 1992 | [q_learning.py](q_learning.py) |
| REINFORCE  | Policy-based | Sutton et al. [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf). In *NeurIPS*, 2000. | [reinforce.py](reinforce.py)   |
| DQN        | Value-based  | Mnih et al. [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). In *NeurIPS Deep Learning Workshop*, 2013 | doing                          |
| A2C        | Actor-Critic | Mnih et al. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783). In *ICML*, 2016 | [a2c.py](a2c.py)               |
| A3C        | Actor-Critic | Mnih et al. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783). In *ICML*, 2016 | [a3c.py](a3c.py)               |
| ACKTR      | Actor-Critic | Wu et al. [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://proceedings.neurips.cc/paper/2017/file/361440528766bbaaaa1901845cf4152b-Paper.pdf). In *NeurIPS*, 2017 | doing                          |
| PPO        | Actor-Critic | Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). *arXiv*, 2017. | [ppo.py](ppo.py)               |
| ACER       | Actor-Critic | Wang et al. [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224). In *ICLR*, 2017 | doing                          |

## Quick Start

### Requirements

```shell
pytorch
gym

tensorboard  # for summary writer
tqdm         # for process bar
```

### Abstract Agent

#### Components / Parameters

| Component      | Description                                    |
| -------------- | ---------------------------------------------- |
| policy         | neural network model                           |
| gamma          | discount factor of cumulative reward           |
| lr             | learning rate i.e. `lr_actor`, `lr_critic`     |
| lr_decay       | weight decay to schedule the learning rate     |
| lr_scheduler   | scheduler for the learning rate                |
| writer         | summary writer to record information           |
| buffer         | replay buffer to store historical trajectories |
| use_cuda       | use GPU                                        |
| clip_grad      | gradients clipping                             |
| max_grad_norm  | maximum norm of gradients clipped              |
| norm_reward    | reward scaling                                 |
| norm_advantage | advantage normalization                        |
| open_tb        | open summary writer                            |
| open_tqdm      | open process bar                               |

#### Methods

| Methods          | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| preprocess_obs() | preprocess observation before input into the neural network  |
| select_action()  | use actor network to select an action base on the policy distribution. |
| estimate_obs()   | use critic network to estimate the value of observation      |
| update()         | update the parameter by calculate losses and gradients       |
| train()          | set the neural network to train mode                         |
| eval()           | set the neural network to evaluate mode                      |
| save()           | save the model parameters                                    |
| load()           | load the model parameters                                    |

## Update &  To-do & Limitations

### Update History

- `2021-12-07` `ADD` `ALGO`: A3C
- `2021-12-05` `ADD` `ALGO`: PPO
- `2021-11-28` `ADD` `ALGO`: A2C
- `2021-11-20` `ADD` `ALGO`: Q learning, Reinforce

### To-do List

- [ ] `ADD` `ALGO` DQN, Double DQN, Dueling DQN, DDPG
- [ ] `ADD` `ALGO` DQN, Double DQN, Dueling DQN, DDPG
- [ ] `ADD` `NN` RNN Mode
- [ ] `FIX` `PARAM` The tradeoff between verbose and tqdm

### Current Limitations

- Unsupport `Vectorized environments`
- Unsupport `Continuous action space`
- Unsupport `RNN-based model`
- Unsupport `Imatation learning`

## Reference & Acknowledgements

- [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)
- [Tianshou](https://github.com/thu-ml/tianshou)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [pytorch-A3C](https://github.com/MorvanZhou/pytorch-A3C)
