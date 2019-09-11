from __future__ import print_function
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time
from Model import PolicyNet,QNet
import gym
import matplotlib.pyplot as plt



class Simulation():
    def __init__(self, args, shared_queue,shared_value):
        super(Simulation, self).__init__()
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.policy_test_queue = shared_queue[3]
        self.stop_sign = shared_value[1]
        self.args = args
        self.env = gym.make(args.env_name)
        self.device = torch.device("cpu")
        self.load_index = 20000


        self.actor = PolicyNet(args.state_dim, args.num_hidden_cell, args.action_high, args.action_low,args.NN_type).to(self.device)
        self.actor.load_state_dict(torch.load('./data/method_' + str(1) + '/model/policy_' + str(self.load_index) + '.pkl'))

        self.Q_net1_m0 = QNet(args.state_dim, args.action_dim, args.num_hidden_cell, args.NN_type).to(self.device)
        self.Q_net1_m0.load_state_dict(torch.load('./data/method_' + str(0) + '/model/Q1_' + str(self.load_index) + '.pkl'))

        self.Q_net1_m1 = QNet(args.state_dim, args.action_dim, args.num_hidden_cell, args.NN_type).to(self.device)
        self.Q_net1_m1.load_state_dict(torch.load('./data/method_' + str(1) + '/model/Q1_' + str(self.load_index) + '.pkl'))
        self.Q_net2_m1 = QNet(args.state_dim, args.action_dim, args.num_hidden_cell, args.NN_type).to(self.device)
        self.Q_net2_m1.load_state_dict(torch.load('./data/method_' + str(1) + '/model/Q2_' + str(self.load_index) + '.pkl'))

        self.Q_net1_m2 = QNet(args.state_dim, args.action_dim, args.num_hidden_cell, args.NN_type).to(self.device)
        self.Q_net1_m2.load_state_dict(torch.load('./data/method_' + str(2) + '/model/Q1_' + str(self.load_index) + '.pkl'))



        self.test_step = 0

        self.save_interval = 10000
        self.iteration = 0
        self.reward_history = []
        self.entropy_history = []
        self.epoch_history =[]
        self.done_history = []
        self.Q_real_history = []
        self.Q_m0_history =[]
        self.Q_m1_history = []
        self.Q_m2_history = []
        self.Q_std_m2_history = []



    def load_param(self):
        if self.policy_test_queue.empty():
            pass
        else:
            self.iteration, param = self.policy_test_queue.get()
            self.actor.load_state_dict(param)

    def run(self):

        step = 0
        while True:
            self.state = self.env.reset()
            self.episode_step = 0
            state_tensor = torch.FloatTensor(self.state.copy()).float().to(self.device)
            if self.args.NN_type == "CNN":
                state_tensor = state_tensor.permute(2, 0, 1)
            self.u, log_prob = self.actor.get_action(state_tensor.unsqueeze(0), False)


            for i in range(self.args.max_step):
                q_m0 = self.Q_net1_m0(state_tensor.unsqueeze(0), torch.FloatTensor(self.u).to(self.device))[0]
                q_m1 = torch.min(
                    self.Q_net1_m1(state_tensor.unsqueeze(0), torch.FloatTensor(self.u).to(self.device))[0],
                    self.Q_net2_m1(state_tensor.unsqueeze(0), torch.FloatTensor(self.u).to(self.device))[0])
                q_m2, q_std, _ = self.Q_net1_m2.evaluate(state_tensor.unsqueeze(0), torch.FloatTensor(self.u).to(self.device))



                self.Q_m0_history.append(q_m0.detach().item())
                self.Q_m1_history.append(q_m1.detach().item())
                self.Q_m2_history.append(q_m2.detach().item())
                self.Q_std_m2_history.append(q_std.detach().item())

                self.u = self.u.squeeze(0)
                self.state, self.reward, self.done, _ = self.env.step(self.u)

                self.reward_history.append(self.reward)
                self.done_history.append(self.done)
                self.entropy_history.append(log_prob)


                if step%10000 >=0 and step%10000 <=9999:
                    self.env.render(mode='human')
                state_tensor = torch.FloatTensor(self.state.copy()).float().to(self.device)
                if self.args.NN_type == "CNN":
                    state_tensor = state_tensor.permute(2, 0, 1)
                self.u, log_prob = self.actor.get_action(state_tensor.unsqueeze(0), False)



                if self.done == True:
                    time.sleep(1)
                    print("!!!!!!!!!!!!!!!")
                    break
                step += 1
                self.episode_step += 1

            if self.done == True:
                break



        for i in range(len(self.Q_m0_history)):
            a = 0
            for j in range(i, len(self.Q_m0_history), 1):
                a += pow(self.args.gamma, j-i)*self.reward_history[j]
            for z in range(i+1, len(self.Q_m0_history), 1):
                a -= self.args.alpha * pow(self.args.gamma, z-i) * self.entropy_history[z]
            self.Q_real_history.append(a)

        print(self.reward_history)
        print(self.entropy_history)
        print(self.Q_m2_history)
        print(self.Q_std_m2_history)

        plt.figure()
        x = np.arange(0,len(self.Q_m0_history),1)
        plt.plot(x, np.array(self.Q_m0_history), 'r', linewidth=2.0)
        plt.plot(x, np.array(self.Q_m1_history), 'g', linewidth=2.0)
        plt.plot(x, np.array(self.Q_m2_history), 'b', linewidth=2.0)


        plt.plot(x, np.array(self.Q_real_history), 'k', linewidth=2.0)

        plt.show()




def test():
    a = np.arange(0,10,1)
    print(a)


if __name__ == "__main__":
    test()



