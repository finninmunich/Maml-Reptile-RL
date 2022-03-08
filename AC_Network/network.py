import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from torch.distributions import MultivariateNormal,Categorical


class ActorCritic(nn.Module):
    def __init__(self, in_dim, out_dim, continuous = True, layer_norm=True):
        super(ActorCritic, self).__init__()

        self.actor_fc1 = nn.Linear(in_dim, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, out_dim)
        self.continuous = continuous
        if self.continuous:
            self.cov_mat = nn.Parameter(torch.diag(torch.full(size=(out_dim,), fill_value=0.5)))
        else:
            self.softmax =nn.Softmax(dim=-1)

        self.critic_fc1 = nn.Linear(in_dim, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if layer_norm:
            self.normalization(self.actor_fc1, std=1.0)
            self.normalization(self.actor_fc2, std=1.0)
            self.normalization(self.actor_fc2, std=1.0)

            self.normalization(self.critic_fc1, std=1.0)
            self.normalization(self.critic_fc2, std=1.0)
            self.normalization(self.critic_fc3, std=1.0)

    @staticmethod
    def normalization(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        if self.continuous:
            action_mean, action_std = self._forward_actor(states)
            critic_value = self._forward_critic(states)
            return action_mean, action_std, critic_value
        else:
            action_prob = self._forward_actor(states)
            critic_value = self._forward_critic(states)
            return action_prob, critic_value

    def _forward_actor(self, states):
        x = F.relu(self.actor_fc1(states))
        x = F.relu(self.actor_fc2(x))
        if self.continuous:
            action_mean = self.actor_fc3(x)
            action_std = self.cov_mat
            return action_mean, action_std
        else:
            action_prob = self.softmax(self.actor_fc3(x))
            return action_prob

    def _forward_critic(self, states):
        x = F.relu(self.critic_fc1(states))
        x = F.relu(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value
    def evaluate(self,states,actions):
        logprobas = self.get_logproba(states,actions)
        values = self._forward_critic(states).flatten()
        loss_entropy = torch.mean(torch.exp(logprobas) * logprobas)
        return logprobas,values,loss_entropy
    def sample_action(self, action_mean, action_std=None):
        if self.continuous:
            dist = MultivariateNormal(action_mean, action_std)
            action = dist.sample()
            logproba = dist.log_prob(action)
            return action, logproba
        else:
            dist = Categorical(action_mean)
            action = dist.sample()
            logproba = dist.log_prob(action)
            return action, logproba

    def get_logproba(self, states, actions):
        if self.continuous:
            action_mean, action_std = self._forward_actor(states)
            dist = MultivariateNormal(action_mean, action_std)
            logproba = dist.log_prob(actions)
        else:
            action_prob = self._forward_actor(states)
            dist = Categorical(action_prob)
            logproba = dist.log_prob(actions)
        return logproba


if __name__ == '__main__':
    test_network = ActorCritic(5, 5)
    x = torch.rand(1, 5)
    action_mean, action_logstd, value = test_network(x)
    action, logproba = test_network.sample_action(action_mean, action_logstd)
    action = action.detach().numpy()
    print(action.shape)
