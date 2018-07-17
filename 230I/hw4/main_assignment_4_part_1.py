# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 20:27:49 2018

@author: tomzhang19901030
"""

import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from Fixed_Income_Toolbox import Vasicek_Model as VM
from Fixed_Income_Toolbox import Bond as Bond
from scipy.optimize import minimize, fsolve
# Question 0: Import Data from CSV: 
df = pd.read_excel('HW4_data.xls')

def CMT2r0(CMT,T, Model):
    def func1(r):
        S = 0
        iend = int(2*T+1)
        for i in range(1,iend):
            S = S + np.exp(-Model.get_r(r,i/2)*i/2)
        Est_CMT = 2*(1-np.exp(-Model.get_r(r,T)*T))/S
        return Est_CMT - CMT
    def func2(r):
        rc = Model.get_r(r,T)
        Est_CMT = (np.exp(rc/2)-1)*2 
        return Est_CMT - CMT
    if T >=1:
        res = fsolve(func1, 0)
    else:
        res = fsolve(func2, 0) 
    return res

data = df.iloc[:, 9:].values/100

def CMT_Error(param,*args): 
    ''' Calcualte CMT Error with the current set of parameters'''
    # Estimate r0
    #data = args
    kappa = param[0]
    mu = param[1]
    r5   = np.zeros(len(data))
    Model = VM(kappa=kappa, mu=mu, sigma = 0.0221)
    for i in range(0,len(data)):
        r5[i]   = CMT2r0(data[i,3],5,Model)
    
    T = [0.25,2,3,5,7,10]
    est_data = np.zeros([len(data),len(T)])
    for i in range(0,len(est_data)):
        r0 = r5[i]
        for j in range(0,len(T)):
            est_data[i,j] = Model.Get_CMT(r0,T[j])
    error = np.sum((data-est_data)**2)
    return error

param = [0.04,0.17]
estimation = 0
if estimation == 1:
    res = minimize(CMT_Error, param , args = (data), method = 'nelder-mead', options = {'maxiter': 10000, 'xtol': 0.0001, 'disp': False})


kappa = res.x[0]
mu    = res.x[1]
Model = VM(kappa,mu,0.0221)

# REport kappa, mu, and estimated rt
T = [0.25,2,3,5,7,10]
rt   = np.zeros(data.shape[0])
Est_CMT= np.zeros(data.shape)
for i in range(0,data.shape[0]):
    rt[i] = CMT2r0(data[i,3],T[3],Model)
    for j in range(0,data.shape[1]):
        Est_CMT[i,j] = Model.Get_CMT(rt[i],T[j])


est_kappa = res.x[0]
est_mu    = res.x[1] 
est_r0    = rt[0]
print(['Question 1 part a:'])
print(['Estimated Kappa is: ',est_kappa])
print(['Estimated Mu is: ',est_mu])
print(['Estimated r0 is: ',est_r0])

print(['Question 1 part b:'])
X = np.arange(0,len(data),1)
plt.figure(1)
plt.plot(X,Est_CMT[:,0],'b')
plt.plot(X,data[:,0],'r')
plt.title('CMT(0.25) Observed vs Estimated')
plt.legend(['Estimated','Observed'])
plt.savefig('CMT025_Obs_Fit.pdf')

plt.figure(2)
plt.plot(X,Est_CMT[:,5],'b')
plt.plot(X,data[:,5],'r')
plt.title('CMT(10) Observed vs Estimated')
plt.legend(['Estimated','Observed'])
plt.savefig('CMT10_Obs_Fit.pdf')

plt.figure(3)
plt.plot(rt)
plt.title('Estimated rt from CMT(5)')
plt.savefig('Fitted_rt.pdf')

rtmean = np.mean(rt)
rtstd  = np.std(rt)
drtstd = np.std(np.diff(rt))


# Question 2 
Model_Q2 = VM(kappa = 0.5 ,mu = 0.06, sigma = 0.01)

r  = 0.04
T1 = 2
T2 = 5 
Z1 = Model_Q2.get_Z(r,T1)
Z2 = Model_Q2.get_Z(r,T2)
f  = np.log(Z1/Z2)/(T2-T1)


Z_2_5 = Model_Q2.get_Z(Model_Q2.Get_Er(r,2),3,t=2)
r_imp = -np.log(Z_2_5)/3
Model_Q2.Get_Esigma(r,2)
print(['Question 2 part a:'])
print(['Forward for 3 year loan in 2 year is: ',f])
print(['Implied yield for the future 3 year zero:', r_imp])

# Part c
r0 = 0.04
r4 = Model_Q2.Get_Er(r0,4)
Q2bond   = Bond(facevalue = 100, T = 6, k = 1, coupon = 0.06)


def func3(r):
    '''Define function to calculate r* '''
    Market = pd.DataFrame({'Maturity':np.arange(1,7,1)})
    Spot   = np.zeros(len(Market))
    for i in range(0,len(Spot)):
        Spot[i] = Model_Q2.get_r(r,Market.loc[i,'Maturity'])
    Market['Spot'] = Spot
    price = Q2bond.get_price(Market,cont = 1)
    return price - 100

r_star = fsolve(func3, 0)

Market = pd.DataFrame({'Maturity':np.arange(1,7,1)})
Spot   = np.zeros(len(Market))
DF     = np.zeros(len(Market))
for i in range(0,len(Spot)):
    Spot[i] = Model_Q2.get_r(r_star,Market.loc[i,'Maturity'])
    DF[i]   = Model_Q2.get_Z(r_star,Market.loc[i,'Maturity'])
    Market['Spot'] = Spot 
    Market['DF2']  = DF
print(Q2bond.get_price(Market,cont =1))
Market['Discount Function'] = np.exp(-Market['Spot']*Market['Maturity'])

strikes = Q2bond.cashflow*Market['Discount Function']

Call = 0
for i in range(0,len(strikes)):
    Call = Call + Model_Q2.Get_Call_Price(r0 = 0.04, t = 0, TO = 4,TB = i+5, K = strikes[i]/Q2bond.cashflow[i])*Q2bond.cashflow[i]
    print(Q2bond.cashflow[i])
print(Call)



dr = 0.001 
Market2 = pd.DataFrame({'Maturity':np.arange(1,11,1)})
Market3 = pd.DataFrame({'Maturity':np.arange(1,11,1)})
Spot2   = np.zeros(len(Market2))
DF2     = np.zeros(len(Market2))
Spot3   = np.zeros(len(Market3))
DF3     = np.zeros(len(Market3))

for i in range(0,len(Spot2)):
    Spot2[i] = Model_Q2.get_r(0.04+dr,Market2.loc[i,'Maturity'])
    Spot3[i] = Model_Q2.get_r(0.04-dr,Market3.loc[i,'Maturity'])
    DF2[i]   = Model_Q2.get_Z(0.04+dr,Market2.loc[i,'Maturity'])
    DF3[i]   = Model_Q2.get_Z(0.04-dr,Market3.loc[i,'Maturity'])
    Market2['Spot'] = Spot2
    Market2['DF2']  = DF2
    Market3['Spot'] = Spot3
    Market3['DF2']  = DF3

PriceUp = Q2bond.get_price(Market2,cont =1)
PriceDn = Q2bond.get_price(Market3,cont =1)

CallUp = 0
for i in range(0,len(strikes)):
    CallUp = CallUp + Model_Q2.Get_Call_Price(r0 = 0.04+dr, t = 0, TO = 4,TB = i+5, K = strikes[i]/Q2bond.cashflow[i])*Q2bond.cashflow[i]

CallDn = 0
for i in range(0,len(strikes)):
    CallDn = CallDn + Model_Q2.Get_Call_Price(r0 = 0.04-dr, t = 0, TO = 4,TB = i+5, K = strikes[i]/Q2bond.cashflow[i])*Q2bond.cashflow[i]
    #print(Q2bond.cashflow[i])

Delta = (CallUp-CallDn)/(PriceUp-PriceDn)

print(Delta)


Bdelta = (PriceUp-PriceDn)/(2*dr)
Odelta =  (CallUp-CallDn)/(2*dr)













