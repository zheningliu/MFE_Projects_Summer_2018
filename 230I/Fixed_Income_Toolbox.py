# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:06:22 2018

@author: tomzhang19901030
"""

import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
class Simple_Calculation:
    # Price must be Zero Coupon Bond, must have maturity
    def cont_comp_ret(r=0,T=0):
        return np.exp(r*T)
    
    def disc_comp_ret(r=0,k=1,T=0):
        return (1+r/k)**(k*T)
    
    def getdiscountfunction (df,face = 100):
        df['Discount Function'] = df['Price']/face
        return df
    
    def getspot(df,k=1,name = 'Spot',sourse = 'Price',face = 100):
        df[name] = ((face/df[sourse])**(1/(k*df['Maturity']))-1)*k
        return df
    
    def getprice(df,k=1,name = 'Price', sourse = 'Spot',face = 100):
        df[name] = face/(1+df[sourse]/k)**(df['Maturity']*k)
        return df
        
    def getforward(df,k=1,deltaT = 0.5,T = 0.5,name = 'Forward',sourse = 'Spot',face = 100):
        SFT = int(T/deltaT)
        df[name] = ((((1+df[sourse]/k)**(k*df['Maturity'])*(1+ \
          df[sourse].shift(SFT)/k)**(-k*df['Maturity'].shift(SFT)))**\
    (1/(k*(df['Maturity']-df['Maturity'].shift(SFT))))-1)*k).shift(-SFT)
        return df
    
    def getPar(df,k=1,CouponFreq = 0.5, deltaT = 0.5,name = 'Par Yield', sourse = 'Price', face = 100):
        DFT = int(CouponFreq/deltaT)
        for i in range(0,len(df)):
            if df.loc[i,'Maturity'] <= CouponFreq:
                df.loc[i,'Interm'] = df.loc[i,'Maturity']*df.loc[i,sourse]/face
            else:
                 df.loc[i,'Interm'] = df.loc[i-DFT,'Interm'] + df.loc[i,sourse]/(k*face)
        df[name] = (1-df[sourse]/face)/df['Interm']
        return df

class Term_Structure_Estimations:
    def Nelson_Siegel_Estimation(df,k=1,face = 100, variation = False):
        def NSEerror(params):
            if variation == False:
                Est_Spot = params[0] + \
                params[1]*(1-np.exp(-df['Maturity']/params[3]))/df['Maturity']*params[3]+\
                params[2]*((1-np.exp(-df['Maturity']/params[3]))/df['Maturity']*params[3] 
                - np.exp(-df['Maturity']/params[3]))
            else:
                Est_Spot= params[0] + \
                params[1]*(1-np.exp(-df['Maturity']/params[3]))/df['Maturity']*params[3]+\
                params[2]*((1-np.exp(-df['Maturity']/params[4]))/df['Maturity']*params[4] 
                - np.exp(-df['Maturity']/params[4]))
            Est_Price = face/(1+Est_Spot/k)**(k*df['Maturity'])
            Price_Diff = Est_Price - df['Price']
            error = np.dot(Price_Diff**2,1/df['Maturity'])
            return error 
        if variation == True:
            params = np.array([0.5, 0.1, 0.3, 1.0,0.2])
        else:
            params = np.array([0, 0, 0, 1])
        res = minimize(NSEerror, params, method='nelder-mead',options={'xtol': 0.00001, 'disp': True})
        params = res.x    
            
        return params
    
    def Svensson_Estimation(df,k=1,face = 100):
        def SSEerror(params):
            Est_Spot = params[0] + \
            params[1]*(1-np.exp(-df['Maturity']/params[3]))/df['Maturity']*params[3]+\
            params[2]*((1-np.exp(-df['Maturity']/params[3]))/df['Maturity']*params[3] 
            - np.exp(-df['Maturity']/params[3])) +\
            params[4]*((1-np.exp(-df['Maturity']/params[5]))/df['Maturity']*params[5] 
            - np.exp(-df['Maturity']/params[5])) 

            Est_Price = face/(1+Est_Spot/k)**(k*df['Maturity'])
            Price_Diff = Est_Price - df['Price']
            error = np.dot(Price_Diff**2,1/df['Maturity'])
            return error 
        
        params = np.array([0, 0, 0, 1,0,2])

        res = minimize(SSEerror, params, method='nelder-mead',options={'xtol': 0.0001, 'disp': True})
        params = res.x    
            
        return params

        
    
