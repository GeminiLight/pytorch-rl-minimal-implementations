import os
import time
import json
import socket
import argparse
import pprint

parser = argparse.ArgumentParser(description='configuration file')

def str2bool(v):
    return v.lower() in ('true', '1')

### Environment ###
problem_arg = parser.add_argument_group('env')
problem_arg.add_argument('--env_name', type=str, default='CartPole-v0', help='')

### Network ###
model_arg = parser.add_argument_group('model')
model_arg.add_argument('--embedding_dim', type=int, default=64, help='')

### Agent ###
agent_arg = parser.add_argument_group('agent')
# print info
agent_arg.add_argument('--verbose', type=str2bool, default=True, help='')
agent_arg.add_argument('--open_tqdm', type=str2bool, default=True, help='')
agent_arg.add_argument('--open_tb', type=str2bool, default=True, help='')
# cuda and parallelism
agent_arg.add_argument('--use_cuda', type=str2bool, default=True, help='')
agent_arg.add_argument('--allow_parallel', type=str2bool, default=False, help='')
# learning rate
agent_arg.add_argument('--lr_actor', type=float, default=3e-4, help='')
agent_arg.add_argument('--lr_critic', type=float, default=1e-5, help='')
agent_arg.add_argument('--lr_update_step', type=int, default=5000, help='')
agent_arg.add_argument('--lr_decay', type=float, default=0.95, help='')
agent_arg.add_argument('--max_grad_norm', type=float, default=1., help='')
# reinforcement learning
agent_arg.add_argument('--rl_gamma', type=float, default=0.99, help='discounted factor for futher reward')
agent_arg.add_argument('--gae_lambda', type=float, default=0.95, help='generalized advantage estimator')
# log and save
agent_arg.add_argument('--log_dir', type=str, default='logs', help='')
agent_arg.add_argument('--save_dir', type=str, default='save', help='')
agent_arg.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
agent_arg.add_argument('--load_path', type=str, default='save', help='must be assigned!')

### Train ###
train_arg = parser.add_argument_group('train')
train_arg.add_argument('--num_epochs', type=int, default=100, help='')
train_arg.add_argument('--batch_size', type=int, default=64, help='')
train_arg.add_argument('--seed', type=int, default=1234, help='Random seed')

def get_config(args=None):
    config = parser.parse_args(args)
    config.run_time = time.strftime("%Y%m%dT%H%M%S")
    config.host_name = socket.gethostname()
    
    # preprocess
    config.log_dir = os.path.join(
        config.log_dir, 
        f'{config.problem_name}-{config.problem_size}', 
        f'{config.host_name}-{config.run_time}'
    )
    config.save_dir = os.path.join(
        config.save_dir, 
        f'{config.problem_name}-{config.problem_size}', 
        f'{config.host_name}-{config.run_time}'
    )

    if not os.path.exists(config.log_dir): os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir): os.makedirs(config.save_dir)
    return config
    
def save_config(config, fname='args.json'):
    config_path = os.path.join(config.save_dir, fname)
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=True)
    print(f'Save config in {config_path}')

def show_config(config):
    pprint(config)

if __name__ == '__main__':
    pass