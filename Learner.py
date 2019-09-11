from __future__ import print_function
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time
from Model import QNet, ValueNet, PolicyNet
from utils import *
import torch.nn as nn
from torch.distributions import Normal

from gym.utils import seeding




class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, args):
        self.storage = []
        self.priority_buffer = []
        self.args = args
        self.ptr = 0
        self.cuda_avail = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_avail else "cpu")

    def push(self, data):
        if len(self.storage) == self.args.buffer_size_max:
            self.storage[int(self.ptr)] = data[0:-1]
            self.priority_buffer[int(self.ptr)] = data[-1]
            self.ptr = (self.ptr + 1) % self.args.buffer_size_max
        else:
            self.storage.append(data[0:-1])
            self.priority_buffer.append(data[-1])

    #self.experience_queue.put((self.counter.value, last_state, u, reward, state, micro_step, done))
    def sample(self, batch_size, epsilon = 1e-6):
        if self.args.sample_method == "random":
            ind = np.random.randint(0, len(self.storage), size=batch_size)
            state, a, r, state_next, micro_step, done = [],[],[], [],[],[]
            for i in ind:
                S, A, R, S_N, M_N, D = self.storage[i]
                state.append(np.array(S, copy=False))
                a.append(np.array(A, copy=False))
                r.append(np.array(R, copy=False))
                state_next.append(np.array(S_N, copy=False))
                micro_step.append(np.array(M_N,copy=False))
                done.append(np.array(D, copy=False))
            return np.array(state),  np.array(a),  np.array(r), np.array(state_next), np.array(micro_step), np.array(done)
        elif self.args.sample_method == "priority":
            priority_length = len(self.priority_buffer)
            if priority_length >= self.args.priority_slice_size:
                self.start_point = np.random.randint(0, priority_length-self.args.priority_slice_size)
                slice_length = self.args.priority_slice_size
            else:
                self.start_point = 0
                slice_length = priority_length
            priority_slice = self.priority_buffer[self.start_point:self.start_point+slice_length]
            priority = (np.absolute(np.array(priority_slice)) + epsilon) ** self.args.priority_alpha
            weight = (slice_length * priority) ** (-self.args.priority_beta)
            weight = torch.FloatTensor(weight)
            priority = torch.FloatTensor(priority)
            self.ind = torch.utils.data.sampler.WeightedRandomSampler(weights=priority, num_samples=batch_size, replacement=True)
            state, a, r, state_next, micro_step, done = [], [], [], [], [], []
            self.weight = []
            for i in self.ind:
                S, A, R, S_N, M_N, D = self.storage[self.start_point+i]
                state.append(np.array(S, copy=False))
                a.append(np.array(A, copy=False))
                r.append(np.array(R, copy=False))
                state_next.append(np.array(S_N, copy=False))
                micro_step.append(np.array(M_N, copy=False))
                done.append(np.array(D, copy=False))
                self.weight.append(weight[i].item())
            return np.array(state), np.array(a), np.array(r), np.array(state_next), np.array(micro_step), np.array(done)

    def get_weight(self):
        return torch.FloatTensor(self.weight).to(self.device), self.ind, self.start_point

class Learner():
    def __init__(self, args, shared_queue,shared_value):
        super(Learner,self).__init__()
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.experience_queue = shared_queue[0]
        self.policy_param_queue = shared_queue[1]
        self.q_param_queue = shared_queue[2]
        self.policy_test_queue = shared_queue[3]
        self.stop_sign = shared_value[1]
        self.buffer = Replay_buffer(args)
        self.args =args


        self.cuda_avail = False  # torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_avail else "cpu")

        if self.args.alpha == 'auto':
            self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
            self.target_entropy = (-args.action_dim if args.target_entropy == 'auto' else args.target_entropy)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = self.args.alpha


        self.Q_net1 = QNet(args.state_dim, args.action_dim, args.num_hidden_cell,args.NN_type).to(self.device)
        self.Q_net1.train()
        self.Q_net1_target = QNet(args.state_dim, args.action_dim, args.num_hidden_cell,args.NN_type).to(self.device)
        self.Q_net1_target.train()
        self.Q_net2 = QNet(args.state_dim, args.action_dim, args.num_hidden_cell,args.NN_type).to(self.device)
        self.Q_net2.train()
        self.Q_net2_target = QNet(args.state_dim, args.action_dim, args.num_hidden_cell,args.NN_type).to(self.device)
        self.Q_net2_target.train()
        self.Value_net = ValueNet(args.state_dim, args.num_hidden_cell,args.NN_type).to(self.device)
        self.Value_net.train()
        self.Value_net_target = ValueNet(args.state_dim, args.num_hidden_cell,args.NN_type).to(self.device)
        self.Value_net_target.train()
        self.actor = PolicyNet(args.state_dim, args.num_hidden_cell, args.action_high, args.action_low,args.NN_type).to(self.device)
        if self.args.code_model=="eval":
            self.actor.load_state_dict(torch.load('./data/method_' + str(1) + '/model/policy_' + str(20000) + '.pkl'))
        self.actor_target = PolicyNet(args.state_dim, args.num_hidden_cell, args.action_high, args.action_low,args.NN_type).to(self.device)
        self.actor.train()
        self.actor_target.train()

        self.Q_net1_optimizer = torch.optim.Adam(self.Q_net1.parameters(), lr=args.critic_lr)
        self.Q_net2_optimizer = torch.optim.Adam(self.Q_net2.parameters(), lr=args.critic_lr)
        self.Value_optimizer = torch.optim.Adam(self.Value_net.parameters(), lr=args.critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.Q_net1_target.load_state_dict(self.Q_net1.state_dict())
        self.Q_net2_target.load_state_dict(self.Q_net2.state_dict())

    def get_qloss(self,q, q_std, next_q):
        if self.args.distributional_Q:
            if self.args.sample_method == "random":
                criterion = nn.MSELoss()
                loss = criterion(q, next_q)
                #loss = -Normal(q, q_std).log_prob(next_q).mean()
            elif self.args.sample_method == "priority":
                torch.pow(q.t() - next_q.t(), 2).matmul(self.weight) / self.weight_sum
                #loss = -Normal(q, q_std).log_prob(next_q).matmul(self.weight) / self.weight_sum
        else:
            if self.args.sample_method == "random":
                criterion = nn.MSELoss()
                loss = criterion(q, next_q)
            elif self.args.sample_method == "priority":
                loss = torch.pow(q.t() - next_q.t(), 2).matmul(self.weight) / self.weight_sum

        return loss

    def get_policyloss(self,q, log_prob):
        if self.args.sample_method == "random":
            loss =  (self.alpha * log_prob - q).mean()
        elif self.args.sample_method == "priority":
            loss = (self.alpha * log_prob.t() - q.t()).matmul(self.weight)/self.weight_sum
        return loss

    def update_net(self,loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def send_to_device(self,s, a, r, s_next, micro_step, done, device):
        s = torch.FloatTensor(s).to(device)
        a = torch.FloatTensor(a).to(device)
        r = torch.FloatTensor(r).to(device)
        s_next = torch.FloatTensor(s_next).to(device)
        micro_step = torch.FloatTensor(micro_step).to(device)
        done = torch.FloatTensor(done).to(device)
        return s, a, r, s_next, micro_step, done

    def update_target_net(self, current_net, target_net):
        params_target = get_flat_params_from(target_net)
        params = get_flat_params_from(current_net)
        set_flat_params_to(target_net, (1 - self.args.tau) * params_target + self.args.tau * params)

    def put_parameter(self):
        state_dict = self.actor.state_dict()
        if self.cuda_avail:
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
        while not self.policy_param_queue.full():
            self.policy_param_queue.put(state_dict)

        state_dict = self.Q_net1.state_dict()
        if self.cuda_avail:
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
        while not self.q_param_queue.full():
            self.q_param_queue.put(state_dict)

    def put_parameter_for_test(self):
        state_dict = self.actor.state_dict()
        if self.cuda_avail:
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
        self.policy_test_queue.put((self.iteration, state_dict))


    def run(self):
        time_init = time.time()
        self.iteration = 0

        while not self.stop_sign.value:
            while not self.experience_queue.empty():
                self.buffer.push(self.experience_queue.get())
            if len(self.buffer.storage) <= self.args.initial_buffer_size:
                pass
            else:
                s, a, r, s_next, micro_step, done = self.buffer.sample(self.args.batch_size)
                s, a, r, s_next, micro_step, done = self.send_to_device(s, a, r, s_next, micro_step, done, self.device)
                if self.args.NN_type == "CNN":
                    s = s.permute(0,3,1,2)
                    s_next = s_next.permute(0,3,1,2)

                if self.args.distributional_Q:
                    _, q_std_1, q_1 = self.Q_net1.evaluate(s, a)
                    index_target = 2
                else:
                    q_1, q_std_1, _ = self.Q_net1.evaluate(s, a)
                    index_target = 0

                a_new, log_prob, entropy, a_mean,std = self.actor.evaluate(s)
                a_next, log_prob_next,_ ,_,_= self.actor.evaluate(s_next)


                if self.args.double_Q:
                    q_2,q_std_2,_ = self.Q_net2.evaluate(s, a)
                    target_q_next = torch.min(self.Q_net1_target.evaluate(s_next, a_next)[index_target],self.Q_net2_target.evaluate(s_next, a_next)[index_target]) - self.alpha * log_prob_next
                    q_value_new = torch.min(self.Q_net1(s, a_new)[0], self.Q_net2(s, a_new)[0])
                else:
                    target_q_next = self.Q_net1_target.evaluate(s_next, a_next)[index_target] - self.alpha * log_prob_next
                    q_value_new = self.Q_net1(s, a_new)[0]

                target_q = r + (1-done) * self.args.gamma * target_q_next

                if self.args.sample_method == "priority":
                    self.weight, ind, start_point = self.buffer.get_weight()
                    self.weight_sum = self.weight.sum()
                    TD = q_1.squeeze() - target_q.squeeze()
                    TD_index = 0
                    for i in ind:
                        self.buffer.priority_buffer[start_point+i] = TD[TD_index].item()
                        TD_index+=1
                if self.args.alpha == 'auto':
                    if self.args.sample_method == "random":
                        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                    elif self.args.sample_method == "priority":
                        alpha_loss = -(self.log_alpha * (log_prob.t().matmul(self.weight)/self.weight_sum + self.target_entropy).detach())
                        self.args.priority_beta = min(1, self.args.priority_beta + self.args.priority_beta_incre)
                    self.update_net(alpha_loss, self.alpha_optimizer)
                    self.alpha = self.log_alpha.exp()
                else:
                    self.alpha = self.args.alpha


                q_loss_1 = self.get_qloss(q_1, q_std_1, target_q.detach())
                self.update_net(q_loss_1, self.Q_net1_optimizer)
                self.update_target_net(self.Q_net1, self.Q_net1_target)

                if self.args.double_Q:
                    q_loss_2 = self.get_qloss(q_2, q_std_2, target_q.detach())
                    self.update_net(q_loss_2, self.Q_net2_optimizer)
                    self.update_target_net(self.Q_net2, self.Q_net2_target)

                if self.args.code_model == "train":
                    policy_loss = self.get_policyloss(q_value_new, log_prob)
                    self.update_net(policy_loss, self.actor_optimizer)

                if self.iteration%self.args.put_param_period == 0:
                    self.put_parameter()

                if self.policy_test_queue.empty():
                    self.put_parameter_for_test()

                if self.iteration % self.args.save_model_period == 0:
                    torch.save(self.actor.state_dict(),'./data/method_' + str(self.args.method) + '/model/policy_' + str(self.iteration) + '.pkl')
                    torch.save(self.Q_net1.state_dict(),'./data/method_' + str(self.args.method) + '/model/Q1_' + str(self.iteration) + '.pkl')
                    if self.args.double_Q:
                        torch.save(self.Q_net2.state_dict(),'./data/method_' + str(self.args.method) + '/model/Q2_' + str(self.iteration) + '.pkl')

                self.iteration += 1

                if self.iteration%1000  == 0:
                    print("iteration", self.iteration, "time",time.time() - time_init )
                    print("loss_1", q_loss_1)
                    print("alpha",self.alpha)
                    print("log",q_std_1.t())

def test():
    def fff(x):
        return x,x+1,x+2

    print(fff(1)[0:2])










if __name__ == "__main__":
    test()
