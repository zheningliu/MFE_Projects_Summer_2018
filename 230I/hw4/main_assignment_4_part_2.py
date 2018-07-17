# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 19:09:17 2018

@author: tomzhang19901030
"""
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from Fixed_Income_Toolbox import Vasicek_Model as VM
from Fixed_Income_Toolbox import Bond as Bond
from scipy.optimize import minimize, fsolve
from Fixed_Income_Toolbox import ZeroCouponBond as ZCB

# Question 3
# Notice that the model is symmetric around x and y, which means that kappa, mu 
#abd sigma for x and y re interchangable, indicating that the results are not unique.

# Question 4
# Set total length to be estimated as T, dt as incrementation
T = 20
dt= 0.1
N = int(T/dt)
market = pd.DataFrame({'Maturity':np.arange(dt,T+dt,dt)})

Q4tree = np.zeros([3,N])
Q4price= np.zeros([3,N])
Q4tree[2,:] = 0.1
Q4tree[1,:] = 0.05
DF = np.zeros(3)
DF[0] = 1
DF[1] = np.exp(-dt*0.05)
DF[2] = np.exp(-dt*0.1)
probmat = np.array([[1-0.1*dt, 0.1*dt,0],[0,1-0.2*dt, 0.2*dt],[0.5*dt, 0, 1-0.5*dt]])

for i in range(0,N):
    j = N-i-1
    price = DF*100
    for k in range(0,j):
        newprice = np.zeros([3,1])
        newprice = np.matmul(probmat,price)*DF
        price = newprice
        
    Q4price[0,j] = price[0]
    Q4price[1,j] = price[1]
    Q4price[2,j] = price[2]
        


bond = ZCB(100, market['Maturity'])


market['r0 = 0%'] = bond.get_spot(Q4price[0,:].T,t=0, k=0)
market['r0 = 5%'] = bond.get_spot(Q4price[1,:].T,t=0, k=0)
market['r0 = 10%'] = bond.get_spot(Q4price[2,:].T,t=0, k=0)


market.plot(x=['Maturity'],y = ['r0 = 0%','r0 = 5%','r0 = 10%'])
plt.savefig('Q4TermStructure.pdf')
T = 3
dt= 0.001
N = int(T/dt)
market = pd.DataFrame({'Maturity':np.arange(0,T+dt,dt)})
probmat = np.array([[1-0.1*dt, 0.1*dt,0],[0,1-0.2*dt, 0.2*dt],[0.5*dt, 0, 1-0.5*dt]])
Q4tree2 = np.zeros([3,N])
Q4price2= np.zeros([3,N])
Q4tree2[2,:] = 0.1
Q4tree2[1,:] = 0.05
DF2 = np.zeros(3)
DF[0] = 1
DF[1] = np.exp(-dt*0.05)
DF[2] = np.exp(-dt*0.1)

price = np.zeros([3,N])
price[:,-1] = DF*104
for i in range(1,N):
    j = N-i-1
    price[:,j] = np.matmul(probmat,price[:,j+1])*DF
    if np.mod(market.loc[j,'Maturity'],1)==0.0 and market.loc[j,'Maturity']!= 0:
        price[:,j] = price[:,j] + DF*4
    if market.loc[j,'Maturity'] == 0.5:
        price[:,j] = np.maximum(price[:,j] -100,0)
        







