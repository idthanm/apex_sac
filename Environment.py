import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

import math
import sympy as sym
from sys import exit
import time




class Highway(gym.Env):


    def __init__(self, x_max=200, v_max=10, y_max=3, ratio_max = 5):
        self.lane_width = 3.5
        self.lane_num = 3
        self.veh_num = 60
        self.forward_max = 1000
        self.rear_max = -800
        self.lane_width = 3.5
        self.timestep = 1/20.
        self.v_upbound = 100
        self.v_lowbound = 60
        self.vel_max = 120.
        self.vel_min = 40.

        self.viewer = None

        self.x_max = x_max
        self.v_max = v_max
        self.y_max = y_max
        self.dist_max = 300.
        self.interval_min = 100.
        self.interval_max = 300.

        self.ratio_max = ratio_max
        self.load_action = 1
        self.max_state = np.array([self.lane_num*self.lane_width, math.pi/2, self.vel_max, self.lane_num,
                                self.lane_width]+[self.dist_max]*2*self.lane_num + [self.vel_max]*2*self.lane_num, dtype=np.float32)
        self.min_state = np.array([0, -math.pi / 2, self.vel_min, -1,
                                -self.lane_width] + [-self.dist_max] *2* self.lane_num + [-self.vel_max]*2*self.lane_num, dtype=np.float32)
        max_action = np.ones([7])
        min_action = np.array([0]+[-1]*6)
        self.action_space = spaces.Box(low=min_action , high=max_action, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_state, high=self.max_state, dtype=np.float32)
        self.seed()
        self.plot = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def function_tra(self, x):
        y = self.a4 * pow(x/self.x_max, 4) + self.a3 * pow(x/self.x_max, 3) + self.a2 * pow(x/self.x_max, 2) + self.y0
        return y

    def function_v(self, x):
        v = self.b3 * pow(x/self.x_max, 3) + self.b2 * pow(x/self.x_max, 2) + self.b1 * x/self.x_max + self.v0
        return v

    def check_lane(self, y):
        for i in range(self.lane_num):
            if y>= i*self.lane_width  and y< (i+1)*self.lane_width:
                lane = i
                break
        if y < 0 or y > self.lane_num*self.lane_width:
            lane = -1
        return lane

    def step(self,a,episilon=1e-8):
        #a = x y v (a4） a3 a2  b3 b2  self.car = [x_init, y_init, angle_init, V_init, lane_init]
        if self.load_action == 1:
            self.x_T = a[0]*self.x_max
            self.y_T = a[1]*self.y_max
            self.V_T = a[2]*self.v_max
            self.a4 = a[3]*self.ratio_max
            self.a3 = a[4]*self.ratio_max
            self.b3 = a[5]*self.ratio_max
            self.b2 = a[6]*self.ratio_max
            self.a2 = (self.y_T - self.a4 * pow(self.x_T/self.x_max, 4) - self.a3 * pow(self.x_T/self.x_max, 3)) / (pow(self.x_T/self.x_max, 2)+episilon)
            self.b1 = (self.V_T - self.b3 * pow(self.x_T/self.x_max, 3) - self.b2 * pow(self.x_T/self.x_max, 2)) / ((self.x_T/self.x_max)+episilon)
            self.y0 = self.car[1]
            self.v0 = self.car[3]
            self.x0 = self.car[0]

        for i in range(self.lane_num):
            for j in range(len(self.other_vehilce_run[i])):
                self.other_vehilce_run[i][j][1] += self.other_vehilce_run[i][j][4] * self.timestep / 3.6
                if i == self.car[4] and j == self.car_neighbor[i][1]:
                    goal_velocity = self.car[3]
                elif j == 0:
                    goal_velocity = self.other_vehilce_run[i][j][6]
                else:
                    goal_velocity = self.other_vehilce_run[i][j-1][4]
                a = 0.0881*(goal_velocity - self.other_vehilce_run[i][j][4]) #+ 1.184 * np.random.randn()/5
                self.other_vehilce_run[i][j][4] += 3.6 * a * self.timestep

        self.car[0] += self.car[3] * np.cos(self.car[2]) * self.timestep / 3.6
        self.car[1] = self.function_tra(x=self.car[0]-self.x0)
        x = sym.symbols("x")
        self.car[2] = float(sym.diff(self.function_tra(x),x).subs(x, self.car[0]-self.x0))
        self.car[3] = self.function_v(x=self.car[0]-self.x0)
        self.car[4] = self.check_lane(self.car[1])


        for i in range(self.lane_num):
            if self.other_vehilce_run[i][0][1] - self.car[0] > 1000:
                self.other_vehilce_stop.append(self.other_vehilce_run[i][0])
                self.other_vehilce_run[i].pop(0)
            if self.other_vehilce_run[i][-1][1] - self.car[0] < -800:
                self.other_vehilce_stop.append(self.other_vehilce_run[i][0])
                self.other_vehilce_run[i].pop(-1)
            if self.other_vehilce_run[i][0][1] - self.car[0] < 0:
                self.other_vehilce_run[i].insert(0, self.other_vehilce_stop[0])
                self.other_vehilce_stop.pop(0)
                self.other_vehilce_run[i][0][1] = self.car[0] + np.random.uniform(500, 1000)
                self.other_vehilce_run[i][0][6] = self.other_vehilce_run[i][1][4] + np.random.uniform(-10.0, 10.0)
                self.other_vehilce_run[i][0][4] = self.other_vehilce_run[i][0][6] + np.random.uniform(-10.0, 10.0)
                self.other_vehilce_run[i][0][5] = i
                self.other_vehilce_run[i][0][2] = i * self.lane_width + self.lane_width / 2 + np.random.uniform( -0.1, 0.1)
            elif self.other_vehilce_run[i][0][1] - self.car[0] < 700:
                self.other_vehilce_run[i].insert(0, self.other_vehilce_stop[0])
                self.other_vehilce_stop.pop(0)
                self.other_vehilce_run[i][0][1] = self.other_vehilce_run[i][1][1] + np.random.uniform(self.interval_min, self.interval_max)
                self.other_vehilce_run[i][0][6] = self.other_vehilce_run[i][1][4] + np.random.uniform(-10.0, 10.0)
                self.other_vehilce_run[i][0][4] = self.other_vehilce_run[i][0][6] + np.random.uniform(-10.0, 10.0)
                self.other_vehilce_run[i][0][5] = i
                self.other_vehilce_run[i][0][2] = i * self.lane_width + self.lane_width / 2 + np.random.uniform( -0.1, 0.1)
            if self.other_vehilce_run[i][-1][1] - self.car[0] > 0:
                self.other_vehilce_run[i].append(self.other_vehilce_stop[0])
                self.other_vehilce_stop.pop(0)
                self.other_vehilce_run[i][-1][1] = self.car[0] - np.random.uniform(500, 800)
                self.other_vehilce_run[i][-1][4] = self.other_vehilce_run[i][-2][4] + np.random.uniform(-10.0, 10.0)
                self.other_vehilce_run[i][-1][5] = i
                self.other_vehilce_run[i][-1][2] = i * self.lane_width + self.lane_width / 2 + np.random.uniform(-0.1,0.1)
            elif self.other_vehilce_run[i][-1][1] - self.car[0] > -500:
                self.other_vehilce_run[i].append(self.other_vehilce_stop[0])
                self.other_vehilce_stop.pop(0)
                self.other_vehilce_run[i][-1][1] = self.other_vehilce_run[i][-2][1] - np.random.uniform(self.interval_min, self.interval_max)
                self.other_vehilce_run[i][-1][4] = self.other_vehilce_run[i][-2][4] + np.random.uniform(-10.0, 10.0)
                self.other_vehilce_run[i][-1][5] = i
                self.other_vehilce_run[i][-1][2] = i * self.lane_width + self.lane_width / 2 + np.random.uniform(-0.1,0.1)
            self.other_vehilce_run[i] = sorted(self.other_vehilce_run[i], key=lambda x: -x[1])

        self.neighbor_vehicle()
        state_output=self._get_obs()
        self._get_reward()
        if self.car[3] * np.cos(self.car[2]) * self.timestep / 3.6 > self.x_T - (self.car[0] - self.x0) or self.done == True:
            self.load_action = 1
        else:
            self.load_action = 0
        return state_output, self.reward, self.done, self.load_action

    def vehilcestate(self, ID, x, y, angle, v, lane, exp_v, forward_ID):
        return [ID, x, y, angle, v, lane, exp_v, forward_ID]

    def neighbor_vehicle(self):
        self.car_neighbor = []
        for i in range(self.lane_num):
            for j in range(len(self.other_vehilce_run[i])):
                if self.other_vehilce_run[i][j][1] - self.car[0] >= 0 and self.car[0] - self.other_vehilce_run[i][j + 1][1] > 0:
                    self.car_neighbor.append([j, j + 1])



    def reset(self):
        self.other_vehilce_stop = []
        self.other_vehilce_run = []


        for i in range(self.veh_num):
            self.other_vehilce_stop.append(self.vehilcestate(ID=i, x=0, y=0, angle=0, v=0, lane=0, exp_v=np.random.uniform(65.0, 115.0), forward_ID=0))
        x_init = 0.
        lane_init = np.floor(np.random.uniform(0, self.lane_num))
        y_init = lane_init * self.lane_width + self.lane_width/2 + np.random.uniform(-0.5, 0.5)
        angle_init = np.random.uniform(-math.pi/4, math.pi/4)
        V_init = np.random.uniform(40.0, 120.0)
        self.car = [x_init, y_init, angle_init, V_init, lane_init]
        x_other = []
        for i in range(self.lane_num):
            x_other.append([])
            self.other_vehilce_run.append([])

            if i != self.car[4]:
                zzz = self.forward_max  - np.random.uniform(self.interval_min, self.interval_max)
                while zzz  > self.rear_max and zzz  < self.forward_max:
                    x_other[i].append([zzz, i])
                    zzz -=  np.random.uniform(self.interval_min, self.interval_max)
            if i == self.car[4]:
                zzz = self.forward_max  - np.random.uniform(self.interval_min, self.interval_max)
                while zzz  > 50  and zzz  < self.forward_max:
                    x_other[i].append([zzz, i])
                    zzz -=  np.random.uniform(self.interval_min, self.interval_max)
                zzz =  - np.random.uniform(self.interval_min, self.interval_max)
                while zzz  > self.rear_max:
                    x_other[i].append([zzz, i])
                    zzz -=  np.random.uniform(self.interval_min, self.interval_max)
            for j in range(len(x_other[i])):
                self.other_vehilce_run[i].append(self.other_vehilce_stop[0])
                self.other_vehilce_stop.pop(0)
                self.other_vehilce_run[i][j][1] = x_other[i][j][0]
                self.other_vehilce_run[i][j][5] = x_other[i][j][1]
                self.other_vehilce_run[i][j][2] = x_other[i][j][1]  * self.lane_width + self.lane_width / 2 + np.random.uniform(-0.1, 0.1)
            self.other_vehilce_run[i] = sorted(self.other_vehilce_run[i], key=lambda x: -x[1])
            for j in range(len(self.other_vehilce_run[i])):
                if j == 0:
                    self.other_vehilce_run[i][j][4] = self.other_vehilce_run[i][j][6] + np.random.uniform(-5.0, 5.0)
                else:
                    self.other_vehilce_run[i][j][4] = self.other_vehilce_run[i][j-1][4] + np.random.uniform(-5.0, 5.0)
        self.neighbor_vehicle()
        self.load_action = 1

        return self._get_obs()

    def _get_reward(self):
        for i in range(self.lane_num):
            if i == self.car[4]:
                self.dist_forward = self.state[5 + i * 2]
                self.dist_rear = self.state[5 + i * 2 + 1]
                self.dist_min = np.min(np.absolute(self.state[5 + i * 2 :5 + i * 2+1]))

        if self.state[3] == -1 or self.dist_min < 3 or self.state[2]>=self.vel_max or self.state[2] <= self.vel_min:
            self.reward = -200
            self.reward_s = -100
            self.reward_v = -100
            self.done = True
        else:
            if self.state[2] > self.v_upbound:
                self.reward_v = -0.1 * (self.state[2]-self.v_upbound)
            elif self.state[2] < self.v_lowbound:
                self.reward_v = -0.05 * (self.v_lowbound-self.state[2])
            else:
                self.reward_v = 0.05 * (self.state[2] - self.v_lowbound)
            self.reward_s = 1 * (self.lane_width/2 - np.absolute(self.state[4])) + 1*(1 - np.absolute(self.state[3] - 1.))
            self.reward = self.reward_v + self.reward_s
            self.done = False

    def _get_obs(self):
        state = self.car.copy()[1:len(self.car)]
        if state[2] >= self.vel_max:
            state[2] = self.vel_max
        elif state[2] <= self.vel_min:
            state[2] = self.vel_min

        if state[3] == -1:
            if self.car[1]  <= 0:
                state.append(self.car[1] - self.lane_width/2)
            elif self.car[1] >= self.lane_width * self.lane_num:
                state.append(self.car[1] - (self.lane_width * self.lane_num-self.lane_width / 2))
        else:
            for i in range(self.lane_num):
                bound_1= self.car[1] -  i * self.lane_width
                bound_2 = self.car[1] - (i+1) * self.lane_width
                if bound_1 >=  0  and bound_2 < 0:
                    state.append((bound_1+bound_2)/2)
                    break

        for i in range(self.lane_num):
            index_1 = self.car_neighbor[i][0]
            index_2 = self.car_neighbor[i][1]
            state.append(np.min([self.other_vehilce_run[i][index_1][1]-self.car[0], self.dist_max]))
            state.append(np.max([self.other_vehilce_run[i][index_2][1]-self.car[0], -self.dist_max]))


        for i in range(self.lane_num):
            index_1 = self.car_neighbor[i][0]
            index_2 = self.car_neighbor[i][1]
            state.append(self.other_vehilce_run[i][index_1][4]-state[2])
            state.append(self.other_vehilce_run[i][index_2][4]-state[2])

        self.state = np.array(state, dtype="float32")






        return self.state/self.max_state

    def mypause(self,interval):
        manager = self.plt._pylab_helpers.Gcf.get_active()
        if manager is not None:
            canvas = manager.canvas
            if canvas.figure.stale:
                canvas.draw_idle()
                # plt.show(block=False)
            canvas.start_event_loop(interval)
        else:
            time.sleep(interval)


    def render(self, mode='human'):
        if self.plot == 0:
            from matplotlib import pyplot as plt

            self.plt = plt
            self.plt.ion()
            import matplotlib.gridspec as gridspec
            self.plot = 1
            plt.figure()
            gs = gridspec.GridSpec(2,2)
            self.ax1 = plt.subplot(gs[0,:])
            self.ax2 = plt.subplot(gs[1, 0])
            self.ax3 = plt.subplot(gs[1, 1])


        x = np.linspace(0, self.x_T, 500)
        y=np.zeros(500, dtype="float32")
        v=np.zeros(500, dtype="float32")
        for i in range(len(x)):
            y[i] = self.function_tra(x=x[i])
            v[i] = self.function_v(x=x[i])

        for i in range(self.lane_num):
            for j in range(len(self.other_vehilce_run[i])):
                self.ax1.scatter(self.other_vehilce_run[i][j][1], self.other_vehilce_run[i][j][2], c='b', marker="*")
            self.ax1.hlines(self.lane_width * (i + 1), self.car[0]-800, self.car[0]+1000)
            self.ax1.vlines(self.car[0]-800, 0, self.lane_width * self.lane_num)
            self.ax1.vlines(self.car[0] + 1000, 0, self.lane_width * self.lane_num)
        self.ax1.scatter(self.car[0], self.car[1], c='r', marker="*")
        self.ax1.text(self.car[0]+100, self.car[1], ['dist_f= %.2f' % self.dist_forward])
        self.ax1.text(self.car[0]-100, self.car[1], ['dist_r= %.2f' % self.dist_rear])
        self.ax1.plot(x+self.x0, y, 'g', linewidth=1.5)
        self.ax1.set_xlim((self.car[0]-200, self.car[0]+200))
        self.ax1.set_ylim((0, self.lane_width * self.lane_num ))

        self.ax2.plot(x, y, 'g', linewidth=1.5)
        self.ax2.text(self.car[0]-self.x0, self.car[1],['reward_s= %.2f' % self.reward_s])
        self.ax2.scatter(self.car[0]-self.x0, self.car[1], c='r', marker="*")

        self.ax3.plot(x, v, 'g', linewidth=1.5)
        self.ax3.scatter(self.car[0]-self.x0, self.car[3], c='r', marker="*")
        self.ax3.text(self.car[0]-self.x0, self.car[3], ['reward_v= %.2f' % self.reward_v])
        #plt.draw()
        #self.plt.pause(0.01)
        self.mypause(0.01)
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()



    def close(self):
        from matplotlib import pyplot as plt
        plt.close('all')


def test():
    env = Highway()
    while True:
        state = env.reset()
        print(state)
        last_state = state.copy()
        micro_step = 0
        u = np.array([np.random.uniform(0.5, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                      np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                      np.random.uniform(-1, 1)], dtype="float32")
        for i in range(100000):
            state, reward, done, load_action = env.step(u)
            env.render(mode="human")
            micro_step += 1
            print(i)
            print(micro_step)
            print("done", done)
            print("reward", reward)
            if load_action == 1:
                last_state = state.copy()
                micro_step = 0
                u = np.array([np.random.uniform(0.5, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                              np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                              np.random.uniform(-1, 1)], dtype="float32")
                print("!!!!")
            if done == 1:

                break















if __name__ == "__main__":
    test()




