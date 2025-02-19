from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
import os

t = np.arange(0, 0.25, 0.00001)#t=0.145: 1000, t=0.23: 5000, t=0.1833: 3000, t=0.169: 2000
deg = 25
lambda_death = 1
lambda_in = 10
alpha = 588
def lambda_prod(n):
        alpha = 588
        k1 = 5600
        K1 = 140
        return alpha + k1/(pow((K1/n),2)+1) if n>0 else alpha

def lambda_deg(n):
    return deg*n

def lambda_div(n,nt,ntt):
        k2 = 40
        K2 = 16.46
        return (math.factorial(n)/(math.factorial(nt)*math.factorial(ntt)))*pow(0.5,n)*k2/(pow((n/K2),4)+1)



def tumor1(w, t):
	(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,
     r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,r32,r33,r34,r35,r36,r37,r38,r39,r40,
     r41,r42,r43,r44,r45,r46,r47,r48,r49,r50) = w
	N = [x for x in range(0,51)]
	return ([-alpha*r0+lambda_deg(1)*r1-lambda_deg(0)*r0-r0*lambda_div(0,0,0)
                     +sum(np.multiply([lambda_div(nt,0,nt-0) for nt in N],w)) + sum(np.multiply([lambda_div(nt,nt-0,0) for nt in N],w))
                         -lambda_death*r0+lambda_in] # r0
                    + [lambda_prod(n - 1) * w[n - 1] - lambda_prod(n) * w[n] + lambda_deg(n + 1) * w[n + 1] - lambda_deg(n) *w[n]
                    - w[n] * sum(lambda_div(n, nt, n - nt) for nt in N[:(n + 1)])
                    + sum(np.multiply([lambda_div(nt, n, nt - n) for nt in N[-(51 - n):]], w[-(51 - n):]))
                    + sum(np.multiply([lambda_div(nt, nt - n, n) for nt in N[-(51 - n):]], w[-(51 - n):]))
                    - lambda_death * w[n] for n in [x for x in range(1,50)]] # r1-r49
                    + [lambda_prod(49) * r49 - lambda_prod(50) * r50 + 0 - lambda_deg(50) * r50 - r50 * sum(
                         lambda_div(50, nt, 50 - nt) for nt in N[:51])
                     + sum(np.multiply([lambda_div(nt, 50, nt - 50) for nt in N[-1:]], w[-1:])) + sum(
                         np.multiply([lambda_div(nt, nt - 50, 50) for nt in N[-1:]], w[-1:]))
                     - lambda_death * r50] #r50
            )
start_time = time.time()
track1 = odeint(tumor1, [10]+[0*i for i in range(0,50)], t)
c1 =track1[-1,:]
plt.plot([x for x in range(0,51)],c1,color="steelblue",label='ODE')
c2 = sum(track1[:,n] for n in range(0,51))
#plt.plot(t,c2,color="steelblue",linewidth=3,label="ODE")
#protein = np.loadtxt('protein_test4.txt')
#plt.hist(protein,bins=65, alpha=0.5,color="steelblue",label="Simulation")
#dist = pd.read_csv('dist_all_1000everypro.csv')
#n = dist.n
#nt = dist.nt
#plt.bar(n,nt,color = "indianred",width = 1,label="FKG–SSA",alpha=0.5)
#plt.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
#font={'family':'Arial','size':  17}
#plt.tick_params(labelsize=15)
#plt.title(r"",fontsize=22,font=font)
#plt.xlabel('Number of proteins',fontsize=22,font=font)
#plt.ylabel(r'Number of cells',fontsize=22,font=font)
#plt.legend(fontsize=14)
#plt.show()
np.savetxt('protein_ODE.txt', c1, fmt='%d', delimiter=',')
end_time = time.time()
cpu_time = [end_time - start_time]
print(f"CPU time: {cpu_time} seconds")

# print(c2[-1])
# print(c1)
# print([int(i) for i in c1])

