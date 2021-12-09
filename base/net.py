from torch import nn


class ActorCritic(nn.Module):

    def __init__(self, input_dim, output_dim, embedding_dim=64):
        super(ActorCritic, self).__init__()
        self.actor = Actor(input_dim, output_dim, embedding_dim)
        self.critic = Critic(input_dim, embedding_dim)

    def act(self, x):
        return self.actor(x)

    def estimate(self, x):
        return self.critic(x)


class Actor(nn.Module):

    def __init__(self, input_dim, output_dim, embedding_dim=64):
        super(Actor, self).__init__()
        self.mlp = MLPNet(input_dim, output_dim, embedding_dim)

    def forward(self, x):
        return self.mlp(x)


class Critic(nn.Module):
    
    def __init__(self, input_dim, embedding_dim=64):
        super(Critic, self).__init__()
        self.mlp = MLPNet(input_dim, 1, embedding_dim)

    def forward(self, x):
        return self.mlp(x)


class MLPNet(nn.Module):
    
    def __init__(self, input_dim, output_dim, embedding_dim=64):
        super(MLPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def layer_norm(net, init_method='orthogonal', weight_std=1.0, bias_const=0.0):
    for name, param in net.named_parameters():
        if name.endswith('weight'):
            nn.init.orthogonal_(param, weight_std)
        elif name.endswith('bias'):
            nn.init.constant_(param, bias_const)
