import numpy as np
from scipy.special import comb
from copy import deepcopy
import matplotlib.pyplot as plt
from random import choice
import time
from scipy.integrate import odeint

db = 1
cutoff = 1
proteins = {}
sumA = {} #number of proteins in a cell
pro_num = {}
max_pro_num = 50
deg = 25
lambda_in = 10

def total_product_rate(cells):
    return sum([cell.lambda_prod() for cell in cells]) #/ len(cells)

def total_deg_rate(cells):
    return sum([cell.lambda_deg() for cell in cells]) #/ len(cells)

def total_influx_rate(cells):
    return lambda_in

def total_birth_rate(cells):
    return sum([cell.lambda_div() for cell in cells]) #/ len(cells)

def total_death_rate(cells):
    return sum([cell.death_rate() for cell in cells]) #/ len(cells)


class Cell:
    def __init__(self, protein=[]):
        #Cell.protein is a list of protein indices. The length indicate the number of proteins in the cell
        self.protein = protein

    def sumA(self):
        protein = self.protein
        #suma = len([proteins[i][0] for i in protein])
        suma = len(protein)
        return suma

    def lambda_prod(self):
        alpha = 588
        k1 = 5600
        K1 = 140
        n = self.sumA()
        return alpha + k1 / (pow((K1 / n), 2) + 1) if n>0 else alpha

    def lambda_deg(self):
        n = self.sumA()
        return deg * n

    def lambda_div(self):
        k2 = 40
        K2 = 16.46
        n = self.sumA()
        return k2 / (pow((n / K2), 4) + 1)

    def death_rate(self):
        return db

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

    def propensity(self, n, cells):
        return self.rate(cells)


class System:
    def __init__(self, num_elements, inits=None, max_t=3000):
        assert num_elements > 0
        self.num_elements = num_elements
        self.reactions = []
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

        def protein(cell):
            new_mut_num = 1
            new_mut_id = [max(proteins.keys(), default=0) + i + 1 for i in range(new_mut_num)]
            anti_val = [0 for _ in range(new_mut_num)]
            for i, new_mut in enumerate(new_mut_id):
                proteins.update({new_mut: [anti_val[i], 0]})
            cell.protein += new_mut_id
            return cell, new_mut_id

        def production():
            prodp = np.array([cell.lambda_prod() for cell in self.cells])
            prodp = prodp / sum(prodp)
            old_cell = np.random.choice(self.cells,p = prodp)
            #protein(old_cell)
            new_cell, new_pro = protein(deepcopy(old_cell))
            self.cells.remove(old_cell)
            self.cells.append(new_cell)

        def degregate():
            degp = np.array([cell.lambda_deg() for cell in self.cells])
            degp = degp / sum(degp)
            deg_cell = np.random.choice(self.cells, p=degp)
            random_pro = choice(deg_cell.protein)
            deg_cell.protein.remove(random_pro)

        def proliferate():
            divp = np.array([cell.lambda_div() for cell in self.cells])
            divp = divp / sum(divp)
            divide_cell = np.random.choice(self.cells,p=divp)
            pro_num = len(divide_cell.protein)
            bnd = np.random.binomial(pro_num, 0.5)
            dau1 = Cell()
            dau2 = Cell()
            dau1.protein = divide_cell.protein[:bnd]
            dau2.protein = divide_cell.protein[bnd:]
            self.cells.remove(divide_cell)
            self.cells.append(dau1)
            self.cells.append(dau2)

        def death():
            deathp = np.array([cell.death_rate() for cell in self.cells])
            deathp = deathp / sum(deathp)
            death_cell = np.random.choice(self.cells, p=deathp)
            #for pro in death_cell.protein:
                #proteins.pop(pro)
            self.cells.remove(death_cell)

        def influx():
            influx_cell = Cell()
            influx_cell.protein =[]
            self.cells.append(influx_cell)

        for i in range(steps):
            A = np.array([rec.propensity(self.n[-1], self.cells)
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
            switch = {0: proliferate, 1: death, 2: production, 3: degregate, 4: influx}
            switch.get(react.index)()



start_time = time.time()
system = System(1, np.array([10]),max_t=0.25)
system.add_reaction(total_birth_rate, [1], [2], index=0)
system.add_reaction(total_death_rate, [1], [0], index=1)
system.add_reaction(total_product_rate, [1], [1], index=2)
system.add_reaction(total_deg_rate, [1], [1], index=3)
system.add_reaction(total_influx_rate, [0], [1], index=4)
system.evolute(1000000000)
pop_size = np.array(system.n).reshape(1, -1)[0]
sumA = np.array([i.sumA() for i in system.cells])
np.savetxt('protein_test.txt', sumA, delimiter=',')
#np.savetxt('time.txt', system.t, delimiter=',')
#np.savetxt('population.txt', pop_size, fmt='%d', delimiter=',')
#plt.plot(system.t,pop_size)
plt.hist(sumA,bins=30,alpha=0.5)
plt.show()
end_time = time.time()
cpu_time = [end_time - start_time]
print(system.n[-1])
#print(f"CPU time: {cpu_time} seconds")
#np.savetxt('CPU_time_0.txt', cpu_time,delimiter=',')
#print(np.loadtxt('CPU_time_0.txt'))