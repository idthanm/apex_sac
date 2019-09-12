from __future__ import print_function
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time
from Model import  PolicyNet
import gym
from tensorboardX import SummaryWriter
import tensorboard


class Test():
    def __init__(self, args, shared_queue,shared_value):
        super(Test, self).__init__()
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.policy_test_queue = shared_queue[3]
        self.stop_sign = shared_value[1]
        self.args = args
        self.env = gym.make(args.env_name)
        self.device = torch.device("cpu")
        self.actor = PolicyNet(args.state_dim, args.num_hidden_cell, args.action_high, args.action_low,args.NN_type).to(self.device)
        self.test_step = 0
        self.epoch_length = 1000
        self.save_interval = 10000
        self.iteration = 0
        self.reward_history = []
        self.iteration_history = []


    def load_param(self):
        if self.policy_test_queue.empty():
            pass
        else:
            self.iteration, param = self.policy_test_queue.get()
            self.actor.load_state_dict(param)

    def run(self):

        epoch = 0
        step = 0
        epoch_reward = 0
        """
        write_stop = 0
        writer = SummaryWriter(comment="test", log_dir='compare'+str(self.args.method))
        """
        while not self.stop_sign.value:
            self.state = self.env.reset()
            self.episode_step = 0
            self.micro_step = 0
            state_tensor = torch.FloatTensor(self.state.copy()).float().to(self.device)
            if self.args.NN_type == "CNN":
                state_tensor = state_tensor.permute(2, 0, 1)
            self.u, log_prob = self.actor.get_action(state_tensor.unsqueeze(0), False)
            self.u = self.u.squeeze(0)
            accumulate_reward = 0
            for i in range(self.args.max_step):
                self.state, self.reward, self.done, self.load_action = self.env.step(self.u)
                if step%10000 >=0 and step%10000 <=9999:
                    epoch_reward += self.reward / self.epoch_length
                    self.env.render(mode='human')
                state_tensor = torch.FloatTensor(self.state.copy()).float().to(self.device)
                if self.args.NN_type == "CNN":
                    state_tensor = state_tensor.permute(2, 0, 1)
                self.u, log_prob = self.actor.get_action(state_tensor.unsqueeze(0), False)
                self.u = self.u.squeeze(0)
                if self.done == True:
                    time.sleep(1)
                    break

                step += 1
                self.episode_step += 1

                if step%self.epoch_length == 0:
                    self.iteration_history.append(self.iteration)
                    self.reward_history.append(epoch_reward)
                    self.load_param()
                    epoch_reward = 0
                    epoch+=1
                    print(epoch)

                if step % self.save_interval == 0:
                    np.save('./data/method_' + str(self.args.method) + '/result/iteration', np.array(self.iteration_history))
                    np.save('./data/method_' + str(self.args.method) + '/result/reward', np.array(self.reward_history))
                    if self.iteration >= self.args.max_train:
                        self.stop_sign.value = 1
                        break

                """
                if write_stop==0:
                    writer.add_scalar("scalar/test", accumulate_reward, epoch)
                if epoch>=100 and write_stop==0:
                    writer.close()
                    write_stop=1
                    print("!!!!!!!!!!!!!!!!")
                """

def test():
    a = torch.tensor([1,-1,1.])
    print(torch.abs(a))



if __name__ == "__main__":
    test()



