from __future__ import print_function
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time
from Model import  QNet, ValueNet, PolicyNet
import gym
from utils import *


class Actor():
    def __init__(self, args, shared_queue, shared_value, lock, i):
        super(Actor, self).__init__()
        self.agent_id = i
        seed = args.seed + np.int64(self.agent_id)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.experience_queue = shared_queue[0]
        self.policy_param_queue = shared_queue[1]
        self.q_param_queue = shared_queue[2]
        self.counter = shared_value[0]
        self.stop_sign = shared_value[1]
        self.lock = lock
        self.env = gym.make(args.env_name)
        self.args = args
        self.device = torch.device("cpu")
        self.actor = PolicyNet(args.state_dim, args.num_hidden_cell, args.action_high, args.action_low,args.NN_type).to(self.device)
        self.Q_net1 = QNet(args.state_dim, args.action_dim, args.num_hidden_cell, args.NN_type).to(self.device)

    def update_actor_net(self, current_dict, actor_net):
        params_target = get_flat_params_from(actor_net)
        params = get_flat_params_from_dict(current_dict)
        set_flat_params_to(actor_net, (1 - self.args.syn_tau) * params_target + self.args.syn_tau * params)

    def load_param(self):
        if self.policy_param_queue.empty():
            #pass
            #print("agent", self.agent_id, "is waiting param")
            time.sleep(0.5)
            #self.load_param()
        else:
            param = self.policy_param_queue.get()
            if self.args.syn_method == "copy":
                self.actor.load_state_dict(param)
            elif self.args.syn_method == "slow":
                self.update_actor_net(param, self.actor)
        if self.q_param_queue.empty():
            time.sleep(0.5)
            #self.load_param()
        else:
            param = self.q_param_queue.get()
            self.Q_net1.load_state_dict(param)

    def put_data(self):
        if not self.stop_sign.value:
            if self.experience_queue.full():
                #print("agent", self.agent_id, "is waiting queue space")
                time.sleep(0.5)
                self.put_data()
            else:
                self.experience_queue.put((self.last_state, self.last_u, [self.reward], self.state, [self.micro_step], [self.done], self.TD.detach().cpu().numpy().squeeze()))
        else:
            pass

    def run(self):
        time_init = time.time()
        step = 0
        self.micro_step = 0
        while not self.stop_sign.value:
            self.state = self.env.reset()
            self.episode_step = 0
            state_tensor = torch.FloatTensor(self.state.copy()).float().to(self.device)
            if self.args.NN_type == "CNN":
                state_tensor = state_tensor.permute(2, 0, 1)
            self.u, log_prob = self.actor.get_action(state_tensor.unsqueeze(0), False)
            q_1 = self.Q_net1(state_tensor.unsqueeze(0), torch.FloatTensor(self.u).to(self.device))[0]
            self.u = self.u.squeeze(0)
            self.last_state = self.state.copy()
            self.last_u = self.u.copy()
            last_q_1 = q_1

            for i in range(self.args.max_step):
                self.state, self.reward, self.done, _ = self.env.step(self.u)
                state_tensor = torch.FloatTensor(self.state.copy()).float().to(self.device)
                if self.args.NN_type == "CNN":
                    state_tensor = state_tensor.permute(2, 0, 1)
                self.u, log_prob = self.actor.get_action(state_tensor.unsqueeze(0), False)
                q_1 = self.Q_net1(state_tensor.unsqueeze(0), torch.FloatTensor(self.u).to(self.device))[0]
                self.u = self.u.squeeze(0)
                if self.episode_step > 0:
                    self.TD = self.reward + (1 - self.done) * self.args.gamma * q_1 - last_q_1
                    self.put_data()
                self.last_state = self.state.copy()
                self.last_u = self.u.copy()
                last_q_1 = q_1

                with self.lock:
                    self.counter.value += 1

                if self.done == True:
                    break

                if step%self.args.load_param_period == 0:
                    self.load_param()
                step += 1
                self.episode_step += 1

def test():
    u = np.array([[-1,3]])
    print(u.max())


if __name__ == "__main__":
    test()



