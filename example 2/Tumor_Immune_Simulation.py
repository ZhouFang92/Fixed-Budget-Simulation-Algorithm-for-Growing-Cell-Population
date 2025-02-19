import numpy as np
from scipy.special import comb
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import pandas as pd

b = 0.5 
mut_rate = 0.5 
db = 0.1 
cutoff = 0 
escape_rate = 1e-4 
s = -0.08 
mutations = {}
sumA = {}

def sel_strength(N, E):
    return s

def birth_rate(cells, N, E):
    return b


def total_death_rate(cells, N, E):
    ss = sel_strength(N, E)
    return sum([cell.death_rate(ss) for cell in cells]) / len(cells)


class Cell:
    def __init__(self, mutation=[], escape_type=None):
        self.mutation = mutation
        self.escape_type = escape_type

    def sumA(self):
        mutation = self.mutation
        suma = sum([mutations[i][0] for i in mutation])
        return suma

    def num(self):
        mutation = self.mutation
        num = len([mutations[i][0] for i in mutation])
        return num

    def escape(self):
        if self.escape_type == 2:
            return 1
        return 0

    def death_rate(self, s):
        sumA = self.sumA()

        if sumA < cutoff or self.escape_type == 2:
            return db
        else:return (1 + s * sumA) * (db - 1) + 1

class Reaction:
    def __init__(self, rate=0., num_lefts=None, num_rights=None, index=None):
        self.rate = rate
        assert len(num_lefts) == len(num_rights)
        self.num_lefts = np.array(num_lefts)
        self.num_rights = np.array(num_rights)
        self.num_diff = self.num_rights - self.num_lefts
        self.index = index

    def combine(self, n, s):
        return np.prod(comb(n, s))

    def propensity(self, n, cells, N, E):
        return self.rate(cells, N, E) * self.combine(n, self.num_lefts)


class System:
    def __init__(self, num_elements, inits=None, N=10, E=0, max_t=10000):
        assert num_elements > 0
        self.num_elements = num_elements
        self.reactions = []
        self.N = N
        self.E = E
        self.log = {'N': [N], 'E': [E]}
        self.max_t = max_t
        if inits is None:
            self.n = [np.ones(self.num_elements)]
            self.cells = [Cell()]

        else:
            self.n = [np.array(inits)]
            self.cells = [Cell() for _ in range(int(self.n[0][0]))]

    def add_reaction(self, rate=0., num_lefts=None, num_rights=None, index=None):
        assert len(num_lefts) == self.num_elements
        assert len(num_rights) == self.num_elements
        self.reactions.append(Reaction(rate, num_lefts, num_rights, index=index))

    def evolute(self, steps):

        self.t = [0]

        def mutation(cell):
            new_mut_num = round( (np.random.poisson(mut_rate)))
            new_mut_id = [max(mutations.keys(), default=0) + i + 1 for i in range(new_mut_num)]
            anti_val = [(np.random.geometric(1/3)-1) for _ in range(new_mut_num)]
            for i, new_mut in enumerate(new_mut_id):
                mutations.update({new_mut: [anti_val[i], 0]})
            cell.mutation += new_mut_id
            return cell, new_mut_id

        def proliferate():
            divide_cell = np.random.choice(self.cells)
            dau1, new_mut1 = mutation(deepcopy(divide_cell))
            dau2, new_mut2 = mutation(deepcopy(divide_cell))

            for mut in new_mut1 + new_mut2:
                mutations[mut][1] += 1
            for mut in divide_cell.mutation:
                mutations[mut][1] += 1
            if np.random.rand() < escape_rate:
                dau1.escape_type = 2
            if np.random.rand() < escape_rate:
                dau2.escape_type = 2
            if divide_cell.escape_type == 2:
                dau1.escape_type = 2
                dau2.escape_type = 2
            self.cells.remove(divide_cell)
            self.cells.append(dau1)
            self.cells.append(dau2)
            if divide_cell.sumA() < cutoff:
                self.N -= 1

                if dau1.sumA() < cutoff:
                    self.N += 1
                else:
                    self.E += 1
                if dau2.sumA() < cutoff:
                    self.N += 1
                else:
                    self.E += 1
            else:
                self.E -= 1
                if dau1.sumA() < cutoff:
                    self.N += 1
                else:
                    self.E += 1
                if dau2.sumA() < cutoff:
                    self.N += 1
                else:
                    self.E += 1

        def death():
            deathp = np.array([cell.death_rate(s) for cell in self.cells])
            deathp = deathp / sum(deathp)
            death_cell = np.random.choice(self.cells, p=deathp)
            if death_cell.sumA() < cutoff:
                self.N -= 1
            else:
                self.E -= 1
            for mut in death_cell.mutation:
                mutations[mut][1] -= 1
            self.cells.remove(death_cell)

        for i in range(steps):
            A = np.array([rec.propensity(self.n[-1], self.cells, self.N, self.E)
                          for rec in self.reactions])
            A0 = A.sum()
            A /= A0
            t0 = -np.log(np.random.random()) / A0
            if self.t[-1] + t0 > self.max_t:
                self.t.append(self.max_t)
                self.n.append(self.n[-1])
                break
            self.t.append(self.t[-1] + t0)
            react = np.random.choice(self.reactions, p=A)
            self.n.append(self.n[-1] + react.num_diff)
            switch = {0: proliferate, 1: death}
            switch.get(react.index)()
            self.log['N'].append(self.N)
            self.log['E'].append(self.E)

start_time = time.time()
for i in range(800,801):
    filename1 = f"output_pt_t30_{i}.csv"
    filename2 = f"output_mut_t30_{i}.csv"
    filename3 = f"output_mutlist_t30_{i}.csv"
    filename4 = f"output_CPU_t30_{i}.csv"
    system = System(1, np.array([10]), max_t=30)
    system.add_reaction(birth_rate, [1], [2], index=0)
    system.add_reaction(total_death_rate, [1], [0], index=1)
    system.evolute(500000000)
    pop_size = np.array(system.n).reshape(1, -1)[0]
    sum_anti = np.array([i.sumA() for i in system.cells])
    num = np.array([i.num() for i in system.cells])
    escape = np.array([i.escape() for i in system.cells])

    pop = pd.DataFrame(pop_size, columns=['pop'])
    t = pd.DataFrame(system.t, columns=['time'])
    sum_anti = pd.DataFrame(sum_anti, columns=['sum_anti'])
    num = pd.DataFrame(num, columns=['mut_num'])
    escape = pd.DataFrame(escape, columns=['escape'])
    data_pt = pd.concat([pop, t], axis = 1, ignore_index=False)
    data_mut = pd.concat([sum_anti, num, escape], axis = 1, ignore_index=False)
    data_pt.to_csv(filename1, index=False)
    data_mut.to_csv(filename2, index=False)
    mutation_num = np.array([0, 0])
    for i in mutations:
        mutation_num = np.vstack((mutation_num, [mutations[i][0], mutations[i][1]]))
    mutation_num = mutation_num[1:, :]
    np.savetxt(filename3, mutation_num)
    end_time = time.time()
    cpu_time = [end_time - start_time]
    np.savetxt(filename4, cpu_time, delimiter=',')