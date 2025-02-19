import numpy as np
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import os
os.chdir('..')

T = 0.25
h = 0.0001
steps = int(T / h) + 1
num = 1000
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

weight_all = []
pro_num = []
traj_n = []
traj_t = []
for i in range(0, num):
    #filename = f"test_react_{i}.csv"
    num_elements = 1
    system = System(num_elements)
    system.add_reaction(lambda_prod, [0], [1])
    system.add_reaction(lambda_deg, [1], [0])
    system.add_reaction(lambda_double_div, [1], [1])
    system.evolute(final_time=T)
    mn = [int(i) for i in system.n]
    mt = [float(i) for i in system.t]
    traj_n.append(mn)
    traj_t.append(mt)
# plt.step(traj_t[0],traj_n[0])
# print(traj_t[0])
# plt.show()

steps = int(T / h) + 1
p = np.zeros(steps)
for j in range(len(traj_n)):
    mt = traj_t[j]
    mn = traj_n[j]
    t_grid, n_at_tgrid = trajectory_on_the_grid(mt, mn, T, h)  # struct the trajectory on the grid
    for i in range(len(t_grid)):
        if n_at_tgrid[i] == 0:
            p[i] += 1 / num


for i in range(len(traj_n)):
    mt = traj_t[i]
    mn = traj_n[i]
    t_grid, n_at_tgrid = trajectory_on_the_grid(mt, mn, T, h)  # struct the trajectory on the grid
    step = len(t_grid)
    w = np.zeros(step)
    w[0] = 1
    division_rate = div(n_at_tgrid)
    for j in range(0, step - 1):
        w[j + 1] = w[j] + h * (division_rate[j] * w[j])
        if n_at_tgrid[j] == 0:
            w[j + 1] = w[j + 1] + h * lambda_in(n_at_tgrid[j]) / (10 * p[j])
    weight_all.append(w[step-1])
    pro_num.append(n_at_tgrid[-1])
ESS = sum(weight_all)**2 / sum([x**2 for x in weight_all])
print("ESS:", ESS)

#protein = list(zip(pro_num,weight_all))
#sorted_protein = list(sorted(protein,key=(lambda x:x[0]),reverse=False))
#dist = pd.DataFrame({'pro_num':pro_num,'weights':weight_all},index = None)

num_traj = len(traj_n)
print("Sample Size:", num_traj)
weight_pro_dist = []
weight_pro_dist = np.zeros(max(pro_num)+1)
for i in range(len(pro_num)):
    weight_pro_dist[pro_num[i]] += weight_all[i]
weight_pro_dist = weight_pro_dist / num_traj * 10

# for pro in range(0,max(pro_num)+1):
#     weights_pro = []
#     for i in range(0, len(pro_num)):
#         if pro_num[i] == pro:
#             weights_pro.append(weight_all[i])
#     weight_pro_dist.append(sum(weights_pro) / num_traj * 10)
dist_1000 = pd.DataFrame({'pro_num':[x for x in range(max(pro_num)+1)],'weights':weight_pro_dist},index = None)
np.savetxt('dist_1000.csv', dist_1000)

end_time = time.time()
cpu_time = [end_time - start_time]
print(f"CPU time: {cpu_time} seconds")

# ODE_pro = np.loadtxt('trajectories/protein_ODE.txt')
# # err = sum([abs(x-y)**2 for x,y in zip(ODE_pro,dist_1000.weights[:50])])/50
# err = sum([abs(x-y)**2 for x,y in zip(ODE_pro,dist_1000.weights[:50])]) / sum([x**2 for x in ODE_pro])
# print(err)
# plt.bar(dist_1000.pro_num,dist_1000.weights, color = "indianred",width = 1,label="FKG–SSA",alpha=0.5)
# plt.plot([x for x in range(0,51)],ODE_pro,color="steelblue",label='ODE')
# plt.show()


data_to_save = {
    'ESS': ESS,
    'CPU_time': end_time - start_time,
    # 'error': err,
    'sample_size': num,
    'dist': dist_1000
}

with open(f'ex1_FKG-SSA_{num}_samples.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)