# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:07:47 2018

@author: tomzhang19901030
"""

import numpy as np
import os
import pandas as pd
from Fixed_Income_Toolbox import Simple_Calculation as SC
from Fixed_Income_Toolbox import Term_Structure_Estimations as TSE
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
"""
Question 1
"""
Q1_a = 20000*SC.disc_comp_ret(0.06,1,8)
Q1_b = 20000*SC.disc_comp_ret(0.06,12,8)
print('Q1')
print(['Annual Disctete Compounding',Q1_a])
print(['Monthly Discrete Compounding',Q1_b])

"""
Question 2
"""
r = 0.055
k = 12 
wealth = 80000*SC.disc_comp_ret(r,k,20)
for i in range(0,240):
    wealth = wealth + 3000*SC.disc_comp_ret(r,k,20-(i+1)/12)
print('Q2')
print(['Total Wealth',wealth])

"""
Question 4
"""
print('Q4')
f_0_1 = ((1+0.05)**1)/(1)-1
f_1_2 = ((1+0.06)**2)/(1+0.05)-1
f_2_3 = ((1+0.07)**3)/((1+0.06)**2)-1
print(['f_0_1',f_0_1])
print(['f_1_2',f_1_2])
print(['f_2_3',f_2_3])

borrow = 1000/(1+0.05)
print(['Borrow this amount for 3 years and invest in 1 year',borrow])

"""
Question 5
"""
r_1 = 0.11
r_2 = (1.13*(r_1+1))**(1/2)-1
r_3 = (1.17*(r_2+1)*(r_2+1))**(1/3)-1
print(['r_1',r_1])
print(['r_2',r_2])
print(['r_3',r_3])

"""
Question 6
"""

P = 10000*(1-32*0.052/360)
print(['The price of the T Bill is:',P])

"""
Question 7
"""

df = pd.read_excel('HW1_data.xls')
df = SC.getspot(df,2,'Spot')
df = SC.getforward(df,k=2,deltaT = 0.25, T = 0.25,name = 'Forward',sourse = 'Spot')
df = SC.getPar(df,k=2,CouponFreq = 0.5, deltaT = 0.25,name = 'Par Yield')
df = SC.getforward(df,k=2,deltaT = 0.25, T = 5,   name = 'Par 5Y Forward',sourse = 'Par Yield')

df.plot(y=['Spot','Forward','Par Yield','Par 5Y Forward'],style=['-','x','-','x'])
df.plot(y=['Spot','Par Yield','Par 5Y Forward'],style=['-','-','x'])


"""
Question 8 a 
"""
df = SC.getdiscountfunction(df,face = 100)
df.plot(x=['Maturity'],y=['Discount Function'],style=['-'])

new_df = pd.DataFrame(df['Maturity'].apply(lambda x: x**i) for i in range(1,6)).T
new_df.columns = ['M1','M2','M3','M4','M5']
new_df['logZ'] = np.log(df['Discount Function'])
rls = sm.ols(formula = "logZ ~ %s + 0" % "+".join("M%s" % i for i in range(1,6)), data=new_df).fit()

y_hat = rls.predict()

plt.figure()
plt.plot(df['Maturity'], np.exp(y_hat), 'y-', df['Maturity'], df['Discount Function'], 'g-')
plt.show()

print(rls.params)

"""
Question 8 b 
"""

SA = np.array(range(1,61))/2

X =np.column_stack((SA,SA**2,SA**3,SA**4,SA**5))
beta = rls.params
estDF = np.exp(np.matmul(beta,X.T))
est_df = pd.DataFrame(estDF)
est_df.columns = ['Estimated DF']
est_df['Maturity'] = SA
est_df['Estimated Price'] = est_df['Estimated DF']*100
est_df = SC.getspot(est_df,k=2,name = 'Estimated Spot',sourse = 'Estimated Price',face =100)
est_df.plot(y = ['Estimated Spot'], x = ['Maturity'])

"""
Question 8 c 
"""


est_df = SC.getPar(est_df,k=2,CouponFreq = 0.5, deltaT = 0.5, name = 'Estimated Par Yield', sourse = 'Estimated Price', face = 100)
est_df.plot(y= ['Estimated Par Yield'],x = ['Maturity'])

"""
Question 8 d 
"""

SA2 = np.array(range(1,121))/2

X2 =np.column_stack((SA2,SA2**2,SA2**3,SA2**4,SA2**5))
estDF2 = np.exp(np.matmul(beta,X2.T))
est_df2 = pd.DataFrame(estDF2)
est_df2.columns = ['Estimated DF2']
est_df2['Maturity'] = SA2
est_df2['Price'] = est_df2['Estimated DF2']*100
est_df2 = SC.getspot(est_df2,k=2,name = 'Spot', face = 100)
est_df2 = SC.getforward(est_df2, k =2, deltaT = 0.5, T = 0.5, name = '6m Forward', sourse = 'Spot',face = 100)

est_df2.plot(y=['Spot','6m Forward'],x= ['Maturity'])

"""
Question 9 b
"""

Q9PMs = TSE.Nelson_Siegel_Estimation(df,k=2,face = 100, variation = False)

SA3 = np.array(range(1,61))/2
Est_SpotQ9 = Q9PMs[0] + \
Q9PMs[1]*(1-np.exp(-SA3/Q9PMs[3]))/SA3*Q9PMs[3]+\
Q9PMs[2]*((1-np.exp(-SA3/Q9PMs[3]))/SA3*Q9PMs[3] 
- np.exp(-SA3/Q9PMs[3]))

Q9df = pd.DataFrame(SA3)
Q9df.columns = ['Maturity']
Q9df['Est Spot'] = Est_SpotQ9

Q9df.plot(y = ['Est Spot'], x = ['Maturity'])

"""
Question 9 c
"""
Q9df = SC.getprice(Q9df,k=2,name = 'Est Price',sourse = 'Est Spot',face = 100)
Q9df = SC.getPar(Q9df,k=2,name = 'Est Par Yield', sourse = 'Est Price')

Q9df.plot(y= ['Est Par Yield'],x = ['Maturity'])

"""
Question 9 d
"""

Est_SpotQ9d = Q9PMs[0] + \
Q9PMs[1]*(1-np.exp(-SA2/Q9PMs[3]))/SA2*Q9PMs[3]+\
Q9PMs[2]*((1-np.exp(-SA2/Q9PMs[3]))/SA2*Q9PMs[3] 
- np.exp(-SA2/Q9PMs[3]))

Q9dfd = pd.DataFrame(SA2)
Q9dfd.columns = ['Maturity']
Q9dfd['Est Spot'] = Est_SpotQ9d

Q9dfd = SC.getforward(Q9dfd, k =2, deltaT = 0.5, T = 0.5, name = '6m Forward', sourse = 'Est Spot',face = 100)

Q9dfd.plot(y=['Est Spot','6m Forward'],x= ['Maturity'])



"""
Question 10 b
"""

Q10PMs = TSE.Svensson_Estimation(df,k=2,face = 100)

Est_SpotQ10 = Q10PMs[0] + \
Q10PMs[1]*(1-np.exp(-SA3/Q10PMs[3]))/SA3*Q10PMs[3]+\
Q10PMs[2]*((1-np.exp(-SA3/Q10PMs[3]))/SA3*Q10PMs[3] 
- np.exp(-SA3/Q10PMs[3])) + \
Q10PMs[4]*((1-np.exp(-SA3/Q10PMs[5]))/SA3*Q10PMs[5] 
- np.exp(-SA3/Q10PMs[5]))

Q10df = pd.DataFrame(SA3)
Q10df.columns = ['Maturity']
Q10df['Est Spot'] = Est_SpotQ10

Q10df.plot(y = ['Est Spot'], x = ['Maturity'])

"""
Question 10 c
"""
Q10df = SC.getprice(Q10df,k=2,name = 'Est Price',sourse = 'Est Spot',face = 100)
Q10df = SC.getPar(Q10df,k=2,name = 'Est Par Yield', sourse = 'Est Price')

Q10df.plot(y= ['Est Par Yield'],x = ['Maturity'])

"""
Question 10 d
"""

Est_SpotQ10d = Q10PMs[0] + \
Q10PMs[1]*(1-np.exp(-SA2/Q10PMs[3]))/SA2*Q10PMs[3]+\
Q10PMs[2]*((1-np.exp(-SA2/Q10PMs[3]))/SA2*Q10PMs[3] 
- np.exp(-SA2/Q10PMs[3])) + \
Q10PMs[4]*((1-np.exp(-SA2/Q10PMs[5]))/SA2*Q10PMs[5] 
- np.exp(-SA2/Q10PMs[5]))

Q10dfd = pd.DataFrame(SA2)
Q10dfd.columns = ['Maturity']
Q10dfd['Est Spot'] = Est_SpotQ10d

Q10dfd = SC.getforward(Q10dfd, k =2, deltaT = 0.5, T = 0.5, name = '6m Forward', sourse = 'Est Spot',face = 100)

Q10dfd.plot(y=['Est Spot','6m Forward'],x= ['Maturity'])





