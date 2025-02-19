from scipy.integrate import odeint
import numpy as np
from scipy.stats import poisson, nbinom
import time
import math
# import pandas as pd
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample
import pickle
from copy import deepcopy
from tqdm import tqdm

T = 30
# resampling_time = [t for t in range(0, T+1, 2)]

k_death_1 = 0.1 # basel death rate
k_death_2 = 0.072 # antigen death rate
k_div = 0.5
k_A = 0.5 # antigen mutation acquisition rate
p_A = 1/3 # the parameter of the geometric distribution
p_IE = 0.0001 # the probability of the immune escape

x1_size = 51
x2_size = 201
x3_size = 2


def lambda_death(n):
    """

    :param n:  a 3-dimension vector indicating the antigen mutations, antigenicity, and immune escape mutations
    :return:
    """
    return k_death_1 + k_death_2 * n[1] * (1-n[2])


def lambda_div(n):
    return k_div


def state_to_index(n):
    return int(n[0] * x2_size * x3_size + n[1] * x3_size + n[2])

def index_to_state(index):
    n = np.zeros(3)
    n[0] = index // (x2_size * x3_size)
    n[1] = (index % (x2_size * x3_size)) // x3_size
    n[2] = index % x3_size
    return n


# def vector_field(n, t):
#     """
#     The vector field of the ODE system
#     :param n: the mean copy numbers in each state
#     :param t: time
#     :return: the derivative of the mean population dynamics
#     """
#     state_size = x1_size * x2_size * x3_size
#     dndt = np.zeros(state_size)
#     for state_index in range(state_size):
#         current_state = index_to_state(state_index)
#         dndt[state_index] -= (lambda_death(current_state)+lambda_div(current_state)) * n[state_index] # outflow
#         division_rate = lambda_div(current_state) * n[state_index] # division rate
#         if current_state[-1] == 0:
#             immune_escape_rate = p_IE
#         else:
#             immune_escape_rate = 0
#         # antigen mutation
#         for DX1 in range(0, int(x1_size-current_state[0])):
#             for DX2 in range(0, int(x2_size-current_state[1])):
#                 for DX3 in range(0, int(x3_size-current_state[2])):
#                     if DX1 == 0 & DX2 == 0:
#                         inflow_rate = division_rate*(1-immune_escape_rate)*poisson.pmf(DX1, k_A)
#                     elif DX1 == 0 & DX2 != 0:
#                         inflow_rate = 0
#                     elif DX3 == 0:
#                         inflow_rate = division_rate*(1-immune_escape_rate)*poisson.pmf(DX1, k_A)*nbinom.pmf(DX2, DX1, p_A)
#                     else:
#                         inflow_rate = division_rate*immune_escape_rate*poisson.pmf(DX1, k_A)*nbinom.pmf(DX2, DX1, p_A)
#                     inflow_rate = inflow_rate * 2 # double rate due to two daughter cells
#                     new_state = deepcopy(current_state)
#                     new_state[0] += DX1
#                     new_state[1] += DX2
#                     new_state[2] += DX3
#                     new_state_index = state_to_index(new_state)
#                     dndt[new_state_index] += inflow_rate
#     return dndt

# comstruct A matrix in the mean population dyanmics

def A_martrix_construction():
    state_size = x1_size * x2_size * x3_size
    A = np.zeros((state_size, state_size))
    for state_index in tqdm(range(state_size)):
        current_state = index_to_state(state_index)
        A[state_index, state_index] -= (lambda_death(current_state)+lambda_div(current_state))
        division_rate = lambda_div(current_state)
        if current_state[-1] == 0:
            immune_escape_rate = p_IE
        else:
            immune_escape_rate = 0
        sum = 0
        for DX1 in range(0, int(x1_size-current_state[0])):
            for DX2 in range(0, int(x2_size-current_state[1])):
                for DX3 in range(0, int(x3_size-current_state[2])):
                    # print(DX1, DX2, DX3)
                    if DX1 == 0 and DX2 == 0 and DX3 == 0:
                        inflow_rate = division_rate*(1-immune_escape_rate)*poisson.pmf(DX1, k_A)
                        # print("step 1")
                    elif DX1 == 0 and DX2 == 0 and DX3 != 0:
                        inflow_rate = division_rate*immune_escape_rate*poisson.pmf(DX1, k_A)
                        # print("step 2")
                    elif DX1 == 0 and DX2 != 0:
                        inflow_rate = 0
                        # print("step 3")
                    elif DX3 == 0:
                        inflow_rate = division_rate*(1-immune_escape_rate)*poisson.pmf(DX1, k_A)*nbinom.pmf(DX2, DX1, p_A)
                        # print("step 4")
                    else:
                        inflow_rate = division_rate*immune_escape_rate*poisson.pmf(DX1, k_A)*nbinom.pmf(DX2, DX1, p_A)
                    inflow_rate = inflow_rate * 2 # double rate due to two daughter cells
                    # test whether the inflow rate is nan
                    # if math.isnan(inflow_rate):
                    #     inflow_rate = 0
                    new_state = deepcopy(current_state)
                    new_state[0] += DX1
                    new_state[1] += DX2
                    new_state[2] += DX3
                    new_state_index = state_to_index(new_state)
                    A[new_state_index, state_index] += inflow_rate
                    sum = sum + inflow_rate
                    # print(inflow_rate)
    return A


class FSP_system:
    def __init__(self, A):
        self.A = A # the A matrix for the mean population dynamics

    def vector_field(self, n, t):
        return np.dot(self.A, n)





# start_time = time.time()
# n0 = np.zeros(x1_size * x2_size * x3_size)
# n0[0] = 10
# t = np.linspace(0, T, 2)
# track1 = odeint(vector_field, n0, t)
# final_solution = track1[-1, :]
# end_time = time.time()
# print("Population size:", sum(final_solution))
# print("CPU time:", end_time - start_time, "seconds")

start_time = time.time()
A = A_martrix_construction()
end_time = time.time()
print("CPU time for constructing A matrix:", end_time - start_time, "seconds")
data_to_save = {"A": A, "CPU time": end_time - start_time}
with open("A_matrix.pkl", "wb") as f:
    pickle.dump(data_to_save, f)


start_time = time.time()
n0 = np.zeros(x1_size * x2_size * x3_size)
n0[0] = 10
t = np.linspace(0, T, 40)
fsp = FSP_system(A)
track1 = odeint(fsp.vector_field, n0, t)
# print(A)
# z = A.dot(n0.T)
# print(z)
# ones = np.ones(x1_size * x2_size * x3_size)
# print(ones.dot(A))
end_time = time.time()
print("CPU time for solve the ode:", end_time - start_time, "seconds")

data_to_save = {"solution": track1, "CPU time": end_time - start_time}
with open("FSP_solution.pkl", "wb") as f:
    pickle.dump(data_to_save, f)




