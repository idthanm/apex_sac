from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.distributions import Normal
import math


cuda_avail = False  # torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")
class QNet(nn.Module):
    def __init__(self, num_states, num_action, num_hidden_cell, NN_type,log_std_min=-20, log_std_max=4):
        super(QNet, self).__init__()
        self.NN_type = NN_type
        if self.NN_type == "CNN":
            self.conv_part = nn.Sequential(
                nn.Conv2d(num_states[-1], 32, kernel_size=4, stride=2, padding=3),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
                nn.ELU(),)
            _conv_out_size = self._get_conv_out_size(num_states)
            self.linear1 = nn.Linear(5*5*32+num_action,  num_hidden_cell, bias=True)
            self.linear2 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)


        if self.NN_type == "mlp":
            self.linear1 = nn.Linear(num_states[-1]+num_action, num_hidden_cell, bias=True)
            self.linear2 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
            self.linear3 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
            self.linear4 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
            self.linear5 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)

        self.mean_layer = nn.Linear(num_hidden_cell, 1, bias=True)
        self.log_std_layer = nn.Linear(num_hidden_cell, 1, bias=True)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.init_weights()

    def _get_conv_out_size(self, num_states):
        out = self.conv_part(torch.zeros(num_states).unsqueeze(0).permute(0,3,1,2))
        return int(np.prod(out.size()))

    def forward(self, state, action):
        if self.NN_type == "CNN":
            x = self.conv_part(state)
            x = x.view(state.size(0),-1)
            x = torch.cat([x, action], 1)
            x = self.linear1(x)
            x = F.elu(x)
            x = self.linear2(x)
            x = F.elu(x)
        elif self.NN_type == "mlp":
            x = torch.cat([state, action], 1)
            x = self.linear1(x)
            x = F.elu(x)
            x = self.linear2(x)
            x = F.elu(x)
            x = self.linear3(x)
            x = F.elu(x)
            x = self.linear4(x)
            x = F.elu(x)
            x = self.linear5(x)
            x = F.elu(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, action, epsilon=1e-6):
        mean, log_std = self.forward(state, action)
        std = log_std.exp()
        normal = Normal(torch.zeros(mean.shape), torch.ones(std.shape))
        z = torch.min(normal.sample(),normal.sample()).to(device)
        #z = normal.sample().to(device)
        q_value = mean + torch.mul(z, std)
        return mean, std, q_value

    def init_weights(self):
        if isinstance(self, nn.Linear):
            weight_shape = list(self.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            self.weight.data.uniform_(-w_bound, w_bound)
            self.bias.data.fill_(0)
        elif isinstance(self, nn.BatchNorm1d):
            self.weight.data.fill_(1)
            self.bias.data.zero_()

class ValueNet(nn.Module):
    def __init__(self, num_states, num_hidden_cell,NN_type):
        super(ValueNet, self).__init__()
        self.NN_type = NN_type

        if self.NN_type == "CNN":
            self.conv_part = nn.Sequential(
                nn.Conv2d(num_states[-1], 32, kernel_size=4, stride=2, padding=3),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
                nn.ELU(),)
            _conv_out_size = self._get_conv_out_size(num_states)
            self.linear1 = nn.Linear(5*5*32,  num_hidden_cell, bias=True)
            self.linear2 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
            self.linear3 = nn.Linear(num_hidden_cell, 1, bias=True)

        if self.NN_type == "mlp":
            self.linear1 = nn.Linear(num_states[-1], num_hidden_cell, bias=True)
            self.linear2 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
            self.linear3 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
            self.linear4 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
            self.linear5 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
            self.linear6 = nn.Linear(num_hidden_cell, 1, bias=True)
        self.init_weights()

    def _get_conv_out_size(self, num_states):
        out = self.conv_part(torch.zeros(num_states).unsqueeze(0).permute(0,3,1,2))
        return int(np.prod(out.size()))


    def forward(self, state):

        if self.NN_type == "CNN":
            x = self.conv_part(state)
            x = x.view(state.size(0),-1)
            x = self.linear1(x)
            x = F.elu(x)
            x = self.linear2(x)
            x = F.elu(x)
            x = self.linear3(x)
        if self.NN_type == "mlp":
            x = state
            x = self.linear1(x)
            x = F.elu(x)
            x = self.linear2(x)
            x = F.elu(x)
            x = self.linear3(x)
            x = F.elu(x)
            x = self.linear4(x)
            x = F.elu(x)
            x = self.linear5(x)
            x = F.elu(x)
            x = self.linear6(x)

        return x

    def init_weights(self):
        if isinstance(self, nn.Linear):
            weight_shape = list(self.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            self.weight.data.uniform_(-w_bound, w_bound)
            self.bias.data.fill_(0)
        elif isinstance(self, nn.BatchNorm1d):
            self.weight.data.fill_(1)
            self.bias.data.zero_()


class PolicyNet(nn.Module):
    def __init__(self, num_states, num_hidden_cell, action_high, action_low,NN_type,log_std_min=-20, log_std_max=2):
        super(PolicyNet, self).__init__()
        self.NN_type = NN_type

        if self.NN_type == "CNN":
            self.conv_part = nn.Sequential(
                nn.Conv2d(num_states[-1], 32, kernel_size=4, stride=2, padding=3),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
                nn.ELU(),)
            _conv_out_size = self._get_conv_out_size(num_states)
            self.linear1 = nn.Linear(5*5*32,  num_hidden_cell, bias=True)
            self.linear2 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
        if self.NN_type == "mlp":
            # self.policy_bn0 = nn.BatchNorm1d(self.num_inputs, momentum=0.99)
            self.linear1 = nn.Linear(num_states[-1], num_hidden_cell, bias=True)
            self.linear2 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
            self.linear3 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
            self.linear4 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
            self.linear5 = nn.Linear(num_hidden_cell, num_hidden_cell, bias=True)
        self.mean_layer = nn.Linear(num_hidden_cell, len(action_high), bias=True)
        self.log_std_layer = nn.Linear(num_hidden_cell, len(action_high), bias=True)
        self.init_weights()

        action_high = torch.tensor(action_high, dtype=torch.float32)
        action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_range = (action_high-action_low)/2
        self.action_bias =  (action_high + action_low)/2
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
    def _get_conv_out_size(self, num_states):

        out = self.conv_part(torch.zeros(num_states).unsqueeze(0).permute(0,3,1,2))

        return int(np.prod(out.size()))


    def forward(self, state):
        if self.NN_type == "CNN":
            x = self.conv_part(state)
            x = x.view(state.size(0),-1)
            x = self.linear1(x)
            x = F.elu(x)
            x = self.linear2(x)
            x = F.elu(x)
        if self.NN_type == "mlp":
            x = self.linear1(state)
            x = F.elu(x)
            x = self.linear2(x)
            x = F.elu(x)
            x = self.linear3(x)
            x = F.elu(x)
            x = self.linear4(x)
            x = F.elu(x)
            x = self.linear5(x)
            x = F.elu(x)

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        #log_std = (self.log_std_max - self.log_std_min)/2 * torch.tanh(log_std) + (self.log_std_max + self.log_std_min)/2
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(torch.zeros(mean.shape), torch.ones(std.shape))
        z = normal.sample().to(device)
        action_0 = mean + torch.mul(z, std)
        action_1 = torch.tanh(action_0)
        action = torch.mul(self.action_range.to(device), action_1) + self.action_bias.to(device)
        log_prob = Normal(mean, std).log_prob(action_0)-torch.log(1. - action_1.pow(2) + epsilon) - torch.log(self.action_range.to(device))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = Normal(mean, std).entropy()

        return action, log_prob, entropy, mean.detach(), std.detach()



    def get_action(self, state, deterministic, epsilon=1e-6):

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(torch.zeros(mean.shape), torch.ones(std.shape))
        z = normal.sample()
        action_0 = mean + torch.mul(z, std)
        action_1 = torch.tanh(action_0)
        action = torch.mul(self.action_range, action_1) + self.action_bias
        log_prob = Normal(mean, std).log_prob(action_0)-torch.log(1. - action_1.pow(2) + epsilon) - torch.log(self.action_range)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action_mean = torch.mul(self.action_range, torch.tanh(mean)) + self.action_bias
        action = action_mean.detach().cpu().numpy() if deterministic else action.detach().cpu().numpy()
        return action, log_prob.detach().item()

    def init_weights(self):
        if isinstance(self, nn.Linear):
            weight_shape = list(self.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            self.weight.data.uniform_(-w_bound, w_bound)
            self.bias.data.fill_(0)
        elif isinstance(self, nn.BatchNorm1d):
            self.weight.data.fill_(1)
            self.bias.data.zero_()

def test():
    mean = torch.tensor([[0,0],[0.5,0.5]], dtype = torch.float32)
    sig = torch.tensor([[1, 1],[2,2]], dtype=torch.float32)
    print(mean.shape)
    bbb = torch.zeros(mean.shape)
    ccc = torch.ones(sig.shape)
    dist = Normal(bbb, ccc).sample()

    pro = Normal(bbb, ccc).log_prob(dist)
    print(pro)

    bb = dist.pow(2)
    print(bb)
    print(bb-1)
    print(bb.sum(-1, keepdim=True))




if __name__ == "__main__":
    test()
