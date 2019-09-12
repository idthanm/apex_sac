from __future__ import print_function
import numpy as np
from Config import GeneralConfig
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import argparse
import random
import os
import time
from Actor import Actor
from Learner import Learner
from Test import Test
from Simulation import Simulation
import gym

torch.multiprocessing.set_start_method('spawn', force=True)
def built_parser(method):
    parser = argparse.ArgumentParser(description='SAC')
    parser.add_argument('--critic_lr' , type=float, default=0.0003,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--actor_lr', type=float, default=0.0003,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--alpha_lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--alpha',  default="auto") #"auto" or 1.
    parser.add_argument('--target_entropy',  default="auto")
    parser.add_argument('--max_step', type=int, default=1000,
                        help='maximum length of an episode (default: 1000000)')
    parser.add_argument('--buffer_size_max', type=int, default=500000, help='replay memory size')
    parser.add_argument('--initial_buffer_size', type=int, default=500,
                        help='Learner waits until replay memory stores this number of transition')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_hidden_cell', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument("--env_name", default="Walker2d-v3")
    #MountainCarContinuous-v0 BipedalWalkerHardcore-v2 Pendulum-v0   LunarLanderContinuous-v2  BipedalWalker-v2  CarRacing-v0
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--state_dim', dest='list', type=int, default=[])
    parser.add_argument('--action_dim', type=int, default=1)
    parser.add_argument('--data_queue_size', type=int, default=100)
    parser.add_argument('--param_queue_size', type=int, default=10)
    parser.add_argument('--action_high', dest='list', type=float, default=[],action="append")
    parser.add_argument('--action_low', dest='list', type=float, default=[],action="append")
    parser.add_argument("--NN_type", default="mlp") # mlp or CNN

    parser.add_argument('--priority_alpha',  type=float, default=0.7)
    parser.add_argument('--priority_beta',  type=float, default=0.4)
    parser.add_argument('--priority_beta_incre', type=float, default=1e-6)
    parser.add_argument('--load_param_period', type=int, default=20)
    parser.add_argument('--put_param_period', type=int, default=20)
    parser.add_argument('--save_model_period', type=int, default=10000)
    parser.add_argument("--syn_tau", type=float, default=0.005)
    parser.add_argument('--priority_slice_size', type=int, default=5000)
    parser.add_argument("--method", type=int, default=method)
    parser.add_argument("--max_train", type=int, default=1000000)

    parser.add_argument("--explore_method", default="parallel")  # "single" or "parallel"
    parser.add_argument("--sample_method", default="random")  # "random or priority"
    parser.add_argument("--syn_method", default="copy")  # "copy" and "slow"
    parser.add_argument("--code_model", default="train") # "train"  "eval" "simu"

    if parser.parse_args().method == 0:
        parser.add_argument("--double_Q", default=False)
        parser.add_argument("--distributional_Q", default=False)
    elif parser.parse_args().method == 1:
        parser.add_argument("--double_Q", default=True)
        parser.add_argument("--distributional_Q", default=False)
    elif parser.parse_args().method == 2:
        parser.add_argument("--double_Q", default=False)
        parser.add_argument("--distributional_Q", default=True)

    return parser.parse_args()



def actor_agent(args, shared_queue, shared_value, lock, i):
    actor = Actor(args, shared_queue, shared_value, lock, i)
    actor.run()

def leaner_agent(args, shared_queue,shared_value):

    leaner = Learner(args, shared_queue,shared_value)
    leaner.run()

def test_agent(args, shared_queue,shared_value):

    test = Test(args, shared_queue,shared_value)
    test.run()

def simu_agent(args, shared_queue,shared_value):

    simu = Simulation(args, shared_queue,shared_value)
    simu.run()


def main(method):
    args = built_parser(method=method)
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    #max_action = float(env.action_space.high[0])
    args.state_dim = state_dim
    args.action_dim = action_dim
    action_high = env.action_space.high
    action_low = env.action_space.low
    args.action_high = action_high.tolist()
    args.action_low = action_low.tolist()
    args.seed = np.random.randint(0,30)
    #time_init = time.time()
    num_cpu = mp.cpu_count()
    if args.explore_method == "single":
        num_actors = 1
    elif args.explore_method == "parallel":
        num_actors = num_cpu - 2

    args.data_queue_size = num_actors * 5
    args.param_queue_size = 10#(num_actors+1) * 3

    experience_queue = Queue(maxsize=args.data_queue_size)
    policy_param_queue = Queue(maxsize=args.param_queue_size)
    policy_test_queue = Queue(maxsize=1)
    q_param_queue = Queue(maxsize=args.param_queue_size)
    shared_queue = [experience_queue, policy_param_queue, q_param_queue,policy_test_queue]
    step_counter = mp.Value('i', 0)
    stop_sign = mp.Value('i', 0)
    shared_value = [step_counter, stop_sign]
    lock = mp.Lock()
    procs=[]
    if args.code_model!="train":
        args.alpha = 0.01
    if args.code_model!="simu":
        procs.append(Process(target=leaner_agent, args=(args, shared_queue,shared_value)))
        for i in range(num_actors):
            procs.append(Process(target=actor_agent, args=(args, shared_queue, shared_value, lock, i)))
        procs.append(Process(target=test_agent, args=(args, shared_queue,shared_value)))
    elif args.code_model=="simu":
        procs.append(Process(target=simu_agent, args=(args, shared_queue, shared_value)))

    for p in procs:
        p.start()
    for p in procs:
        p.join()

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    main(1)





