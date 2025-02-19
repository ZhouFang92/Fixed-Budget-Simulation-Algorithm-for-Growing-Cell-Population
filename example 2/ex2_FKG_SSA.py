import numpy as np
import time
import math
# import pandas as pd
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample
import pickle
from copy import deepcopy



T = 30
# h = 0.0001
# steps = int(T / h) + 1
# p = np.zeros(steps)
num = 50000
N = [x for x in range(0,100)]
lambda_death = 1


k_death_1 = 0.1 # basel death rate
k_death_2 = 0.072 # antigen death rate
k_div = 0.5
k_A = 0.5 # antigen mutation acquisition rate
p_A = 1/3 # the parameter of the geometric distribution
p_IE = 0.0001 # the probability of the immune escape


x1_size = 101
x2_size = 1001
x3_size = 2

# def state_to_index(n):
#     return int(n[0] * x2_size * x3_size + n[1] * x3_size + n[2])
#
# def index_to_state(index):
#     n = np.zeros(3)
#     n[0] = index // (x2_size * x3_size)
#     n[1] = (index % (x2_size * x3_size)) // x3_size
#     n[2] = index % x3_size
#     return n



def lambda_death(n):
    """

    :param n:  a 3-dimension vector indicating the antigen mutations, antigenicity, and immune escape mutations
    :return:
    """
    return k_death_1 + k_death_2 * n[1] * (1-n[2])


def lambda_div(n):
    return k_div

def lambda_double_div(n):
    return 2 * lambda_div(n)


class Reaction:
    def __init__(self, propensity, num_lefts=None, num_rights=None):
        self.propensity = propensity #反应速率函数
        assert len(num_lefts) == len(num_rights)
        self.num_lefts = np.array(num_lefts) # 反应前各个反应物的数目
        self.num_rights = np.array(num_rights) # 反应后各个反应物的数目
        self.num_diff = self.num_rights - self.num_lefts # 改变数


class System:
    def __init__(self, num_elements):
        assert num_elements > 0
        self.num_elements = num_elements # 系统内的反应物的类别数
        self.reactions = [] # 反应集合

    def add_reaction(self, propensity, num_lefts=None, num_rights=None):
        assert len(num_lefts) == self.num_elements
        assert len(num_rights) == self.num_elements
        self.reactions.append(Reaction(propensity, num_lefts, num_rights))

    def evolute(self, final_time, inits = None, steps=1000000):
        self.t = [0]
        if inits is None:
            self.n = [np.zeros(self.num_elements)]
        else:
            self.n = [np.array(inits)]   # 反应物数目，n[0]是初始数目
        for i in range(steps):
            A = np.array([rec.propensity(self.n[-1])
                          for rec in self.reactions])
            A0 = A.sum()
            A = A / A0  # 归一化得到概率分布
            t0 = -np.log(np.random.random()) / A0  # 按概率选择下一个反应发生的间隔
            if self.t[-1] + t0 > final_time:
                self.t.append(final_time)
                self.n.append(self.n[-1])
                break
            self.t.append(self.t[-1] + t0)
            d = np.random.choice(self.reactions, p=A)  # 按概率选择其中一个反应发生
            if d == self.reactions[0]:
                DX1 = np.random.poisson(lam=k_A)
                if DX1 > 0:
                    DX2 = np.random.negative_binomial(n=DX1, p=p_A)
                else:
                    DX2 = 0
                current_state = self.n[-1]
                if current_state[-1] == 0:
                    DX3 = np.random.binomial(n=1, p=p_IE)
                else:
                    DX3 = 0
                self.n.append(current_state + np.array([DX1, DX2, DX3]))

        return deepcopy(self.t), deepcopy(self.n)


def FGK_SSA_one_period(sample_size, time_period, mu_total, init_particles=None, h=0.0001):
    """

    :param sample_size:
    :param time_period:
    :param h:
    :param ini_particles: a dictionary state_list and weight_list
    :return:
    """

    # initial condition
    initial_conditions = []
    if init_particles is None:
        initial_conditions = [np.zeros(3)]*sample_size
    else:
        total_weight = sum(init_particles['weight_list'])
        weights = [w / total_weight for w in init_particles['weight_list']]
        index = systematic_resample(weights)
        for i in index:
            initial_conditions.append(init_particles['state_list'][i])
        mu_total = mu_total


    # evolution of the system
    num_elements = 3
    system = System(num_elements)
    system.add_reaction(lambda_double_div, [0, 0, 0], [0, 0, 0])
    weight_all = []
    state_all = []
    traj_n = []
    traj_t = []
    for i in range(0, sample_size):
        single_t, single_trj = system.evolute(final_time=time_period, inits=initial_conditions[i])
        traj_n.append(single_trj)
        traj_t.append(single_t)

    # compute the weight of each trajectory
    for i in range(len(traj_n)):
        t_grid = traj_t[i]
        n_at_tgrid = traj_n[i]
        w = 1
        for j in range(len(t_grid)-1):
            dt = t_grid[j+1] - t_grid[j]
            integrand = lambda_div(n_at_tgrid[j]) - lambda_death(n_at_tgrid[j])
            w = np.exp(integrand * dt) * w
        weight_all.append(w)
        state_all.append(n_at_tgrid[-1])
    ESS = sum(weight_all)**2 / sum([x**2 for x in weight_all])

    num_traj = len(traj_n)
    print("Sample Size:", num_traj)
    # state_all_array = np.array(state_all)
    # n_max = np.max(state_all_array, axis=0)
    # print(n_max)
    dist = np.zeros([x1_size, x2_size, x3_size])
    for i in range(num_traj):
        n = traj_n[i][-1]
        dist[int(n[0]), int(n[1]), int(n[2])] += weight_all[i]
    dist = dist / num_traj * mu_total

    return dist, ESS, weight_all, state_all


# num_elements = 3
# system = System(num_elements)
# system.add_reaction(lambda_double_div, [0, 0, 0], [0, 0, 0])
# t, state = system.evolute(final_time=50)
#
# print(state[-1][1] / state[-1][0])
# print(state[-1][0])
# print(t[-1])
# print(t)
# print(state)
#
# plt.step(t, state, where='post')
# plt.show()


start_time = time.time()
dist, ESS, weight_all, state_all = FGK_SSA_one_period(sample_size=num, time_period=T, mu_total=10)
print("Time:", time.time()-start_time)
print("ESS:", ESS)
# print("weight_all:", weight_all)
print("Cell Population:", dist.sum())








