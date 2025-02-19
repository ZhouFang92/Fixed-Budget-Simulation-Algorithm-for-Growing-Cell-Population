import numpy as np
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample
import pickle


import os
os.chdir('..')

T = 0.25
# period_for_resampling = 0.25
h = 0.0001
steps = int(T / h) + 1
p = np.zeros(steps)
num = 1000 #16000
N = [x for x in range(0,100)]
lambda_death = 1

deg = 25
def lambda_prod(n):
    alpha = 588
    k1 = 5600
    K1 = 140
    if n[0] == 0:
        return alpha
    return alpha + k1 / (pow((K1 / n[0]), 2) + 1)

def lambda_deg(n):
    return deg*n[0]


def lambda_div(n):
    k2 = 40
    K2 = 16.46
    return k2 / (pow((n[0] / K2), 4) + 1)

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
            A /= A0  # 归一化得到概率分布
            t0 = -np.log(np.random.random()) / A0  # 按概率选择下一个反应发生的间隔
            self.t.append(self.t[-1] + t0)
            d = np.random.choice(self.reactions, p=A)  # 按概率选择其中一个反应发生
            if d in self.reactions[0:2]:
                self.n.append(self.n[-1] + d.num_diff)
            else:
                self.n.append((np.random.binomial(self.n[-1].astype(np.int64, copy=False), 0.5)))
            if self.t[-1] > final_time:
                break

def lambda_in(n):
    return 10 if n == 0 else 0

def lambda_div1(n, nt):
    k2 = 40
    K2 = 16.46
    return (math.factorial(n) / (math.factorial(nt) * math.factorial(n-nt))) * pow(0.5, n) * k2 / (pow((n / K2), 4) + 1)

def div(p):
	return [sum(lambda_div1(int(n), nt) for nt in N[:(int(n) + 1)])-lambda_death for n in p]

start_time = time.time()

def trajectory_on_the_grid(t, n, T, h):
    steps = int(T / h) + 1
    tgrid = np.linspace(0, T, steps)
    n_at_tgrid = []
    i_current = 0
    for time_point in tgrid:
        while t[i_current+1] < time_point: # find the first time point that is greater than the current time point
            i_current += 1
        n_at_tgrid.append(n[i_current])
    return tgrid, n_at_tgrid

def FGK_SSA_one_period(sample_size, time_period, h, mu_total, init_particles=None):
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
        initial_conditions = [np.zeros(1)]*sample_size
    else:
        init_particles['state_list'].append(np.zeros(1))
        total_weight = sum(init_particles['weight_list'])
        weights = [w / total_weight * mu_total for w in init_particles['weight_list']]
        weights.append(lambda_in(0)/sample_size)
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        index = systematic_resample(weights)
        for i in index:
            initial_conditions.append(init_particles['state_list'][i])
        mu_total = mu_total + lambda_in(0)/sample_size


    # evolution of the system
    num_elements = 1
    system = System(num_elements)
    system.add_reaction(lambda_prod, [0], [1])
    system.add_reaction(lambda_deg, [1], [0])
    system.add_reaction(lambda_double_div, [1], [1])
    weight_all = []
    pro_num = []
    traj_n = []
    traj_t = []
    for i in range(0, sample_size):
        system.evolute(final_time=time_period, inits=initial_conditions[i])
        t_grid, n_at_tgrid = trajectory_on_the_grid(system.t, system.n, time_period, h)
        traj_n.append(n_at_tgrid)
        traj_t.append(t_grid)

    # compute the probability at zero
    steps = int(time_period / h) + 1
    p = np.zeros(steps)
    for i in range(len(traj_n)):
        t_grid = traj_t[i]
        n_at_tgrid = traj_n[i]
        for j in range(len(t_grid)):
            if n_at_tgrid[j] == 0:
                p[j] += 1 / sample_size

    # compute the weight of each trajectory
    for i in range(len(traj_n)):
        t_grid = traj_t[i]
        n_at_tgrid = traj_n[i]
        step = len(t_grid)
        w = np.zeros(step)
        w[0] = 1
        division_rate = div(n_at_tgrid)
        for j in range(0, step - 1):
            w[j + 1] = w[j] + h * (division_rate[j] * w[j])
            if n_at_tgrid[j] == 0:
                w[j + 1] = w[j + 1] + h * lambda_in(n_at_tgrid[j]) / (mu_total * p[j])
        weight_all.append(w[step-1])
        pro_num.append(n_at_tgrid[-1])
    ESS = sum(weight_all)**2 / sum([x**2 for x in weight_all])

    num_traj = len(traj_n)
    print("Sample Size:", num_traj)
    weight_pro_dist = []
    weight_pro_dist = np.zeros(int(max(pro_num)) + 1)
    for i in range(len(pro_num)):
        weight_pro_dist[int(pro_num[i])] += weight_all[i]
    weight_pro_dist = weight_pro_dist / num_traj * mu_total

    return weight_pro_dist, ESS, weight_all, pro_num





# test the algorithm on the whole time period

start_time = time.time()
ESS_list = []
resampling_time = [0, 0.05, 0.10, 0.15, 0.2, 0.25]
# resampling_time = [0, 0.25]
for i in range(len(resampling_time)-1):
    if i == 0:
        weight_pro_dist, ESS, weight_all, pro_num = FGK_SSA_one_period(sample_size=num, time_period=resampling_time[1], h=h, mu_total=10)
    else:
        init_particles = {'state_list': pro_num, 'weight_list': weight_all}
        mu_total = sum(weight_pro_dist)
        print("mu_total:", mu_total)
        weight_pro_dist, ESS, weight_all, pro_num = FGK_SSA_one_period(sample_size=num, time_period=resampling_time[i+1]-resampling_time[i], h=h, mu_total=mu_total, init_particles=init_particles)
    ESS_list.append(ESS)
    print("ESS:", ESS)
# weight_pro_dist, ESS, weight_all, pro_num = FGK_SSA_one_period(sample_size=num, time_period=T, h=h, mu_total=10)
dist_1000 = pd.DataFrame({'pro_num':[x for x in range(int(max(pro_num)+1))],'weights':weight_pro_dist},index = None)
end_time = time.time()
print("Time used:", end_time - start_time)


# ODE_pro = np.loadtxt('trajectories/protein_ODE.txt')
# # err = sum([abs(x-y)**2 for x,y in zip(ODE_pro,dist_1000.weights[:50])])/50
# err = sum([abs(x-y)**2 for x,y in zip(ODE_pro,dist_1000.weights[:50])]) / sum([x**2 for x in ODE_pro])
# print(err)
# plt.bar(dist_1000.pro_num,dist_1000.weights, color = "indianred",width = 1,label="FKG–SSA",alpha=0.5)
# plt.plot([x for x in range(0,51)],ODE_pro,color="steelblue",label='ODE')
# plt.show()

data_to_save = {
    'time': resampling_time,
    'ESS': ESS_list,
    'CPU_time': end_time - start_time,
    # 'error': err,
    'sample_size': num,
    'dist': dist_1000
}

print("the algorithm finished")

# save the result
# with open(f'ex1_resampling_{num}_samples.pkl', 'wb') as f:
#     pickle.dump(data_to_save, f)



