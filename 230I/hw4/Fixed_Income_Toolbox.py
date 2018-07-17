# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:06:22 2018

@author: Tom Zhang & Nini Liu
"""

import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize, fsolve
from sympy.solvers import solve
from sympy import Symbol
from scipy.stats import norm

class ZeroCouponBond:
    """Represents properties of zero coupon bonds"""
    def __init__(self, fv, maturity):
        """Initialize bond features"""
        self.fv = fv
        self.maturity = maturity


    def cont_ret(self, rate, T, t = 0):
        """Returns continuous compounded return over time period T1 to T2"""
        return np.exp(rate * (T - t))


    def disc_ret(self, rate, k, T, t = 0):
        """Returns discrete compounded return over time period T1 to T2"""
        return np.power(1 + rate / k, k * (T - t))


    def future_value(self, X, spot, T, t = 0, k = 1):
        """Returns future value of a bond. Default to annually compounded"""
        if k == 0:
            return X * self.cont_ret(spot, T, t)
        else:
            return X * self.disc_ret(spot, k, T, t)


    def present_value(self, C, spot, T, t = 0, k = 1):
        """Returns present value of a bond. Default to annually compounded"""
        if k == 0:
            return C / self.cont_ret(spot, T, t)
        else:
            return C / self.disc_ret(spot, k, T, t)
    

    def get_discount_function(self, spot, T, t = 0, k = 1):
        """Returns discount rate of a bond"""
        return self.present_value(1, spot, T, t, k)
    

    def get_spot(self, price, t = 0, k = 1):
        """Returns spot rate of a bond given bond price"""
        if k == 0:
            return  np.log(self.fv/price) / (self.maturity - t)
        else:
            return (np.power(self.fv/price, 1 / (k * (self.maturity - t))) - 1) * k
    

    def get_price(self, spot, t = 0, k = 1):
        """Returns bond price given spot rate"""
        if k == 0:
            return self.fv * np.exp(- spot * (self.maturity - t))
        else:
            return self.fv / np.power(1 + spot / k, (self.maturity - t) * k)
        

    def get_forward(self, spot1, spot2, T1, T2, t = 0, k = 1):
        """Returns forward rate over time period T1 to T2"""
        ret_T1 = self.disc_ret(spot1, k, T1, t)
        ret_T2 = self.disc_ret(spot2, k, T2, t)
        return (np.power(ret_T2 / ret_T1, 1 / (k * (T2 - T1))) - 1) * k

    
    def get_par_yield(self, spot, t = 0, k = 1, coupon_k = 1, fp = 1):
        """Returns par yield curve of a bond"""
        coupon_freq = 1 / coupon_k
        try:
            T = [item for item in self.maturity]
        except TypeError:
            T = [self.maturity]
            spot = [spot]
        df = pd.DataFrame(columns = ['T'])
        df['T'] = T
        df['n_c'] = np.floor(df['T'] / coupon_freq)
        t_c = np.linspace(0, len(df) * coupon_freq, len(df)+1)[1:]
        interp_spot = np.interp(t_c.tolist(), T, spot)
        z = self.get_discount_function(pd.Series(interp_spot), pd.Series(t_c), t, k)
        df['accrued'] = df['T'] % coupon_freq / coupon_freq
        df['m_z'] = self.get_discount_function(spot, self.maturity, t, k)
        df['par'] = df.apply(lambda row: k * (1 - row['m_z']) /
                        (sum(z[:int(row['n_c'])]) + row['m_z'] * row['accrued']), axis=1)
        return df['par']
    
    def get_par_yield2(self,market, k =2, CouponFreq = 0.5, DeltaT = 0.5,facevalue = 100):
        DFT = int(CouponFreq/DeltaT)
        for i in range(0,len(market)):
            if market.loc[i,'Maturity'] <= CouponFreq:
                market.loc[i,'Interm'] = market.loc[i,'Maturity']*market.loc[i,'Price']/facevalue
            else:
                market.loc[i,'Interm'] = market.loc[i-DFT,'Interm'] +market.loc[i,'Price']/(k*facevalue)
        market['Par Yield2'] = (1-market['Price']/facevalue)/market['Interm']
        del market['Interm']
        return market['Par Yield2']

    def get_payoff_tree(spot_dict, cf_dict):
        sorted_lv = sorted(spot_dict.keys(), reverse=True)
        last_period = sorted_lv[0]
        payoff_dict = {}
        for lv in sorted_lv:
            if lv == last_period:
                payoff_dict[lv] = np.divide(cf_dict[lv], [s + 1 for s in spot_dict[lv]]).tolist()
            else:
                payoff_dict[lv] = np.zeros(lv).tolist()
                for i in range(lv):
                    payoff_dict[lv][i] = (0.5 * (payoff_dict[lv+1][i] + payoff_dict[lv+1][i+1]) + cf_dict[lv][i]) / (1 + spot_dict[lv][i])
        return payoff_dict

    def nelson_siegel_estimation(self, true_price, k = 1, variation = False):
        """Returns nelson siegel term structure estimation"""
        def function(params):
            exp_t1 = np.exp(- self.maturity / params[3])
            common_term = params[0] + params[1] * (1 - exp_t1) / self.maturity * params[3]
            if not variation:
                diff_term = params[2] * ((1 - exp_t1) / self.maturity * params[3] - exp_t1)
            else:
                exp_t2 = np.exp(- self.maturity / params[4])
                diff_term = params[2] * ((1 - exp_t2) / self.maturity * params[4] - exp_t2)
            est_spot = common_term + diff_term
            return self.get_price(est_spot, k = k)

        def NSEerror(params):
            est_price= function(params)
            error = np.dot(np.power(est_price - true_price, 2), 1 / self.maturity)
            return error 

        params = np.array([0.01, 0.01, 0.01, 1.0, 0.2]) if variation else np.array([0, 0, 0, 1])
        res = minimize(NSEerror, params, method = 'nelder-mead', options = {'maxiter': 10000, 'xtol': 0.00001, 'disp': True})
        print(res.x)
        return function(res.x)
    

    def svensson_estimation(self, true_price, k = 1):
        """Returns svensson term structure estimation"""
        def function(params):
            exp_t1 = np.exp(- self.maturity / params[4])
            exp_t2 = np.exp(- self.maturity / params[5])
            est_spot = params[0] + params[1] * (1 - exp_t1) / self.maturity * params[4] + \
                        params[2] * ((1 - exp_t1) / self.maturity * params[4] - exp_t1) + \
                        params[3] * ((1 - exp_t2) / self.maturity * params[5] - exp_t2)
            return self.get_price(est_spot, k = k)

        def SSEerror(params):
            est_price = function(params)
            error = np.dot(np.power(est_price - true_price, 2),1 / self.maturity)
            return error

        params = np.array([0, 0, 0, 1, 0, 2])
        res = minimize(SSEerror, params, method = 'nelder-mead', options = {'maxiter': 10000, 'xtol': 0.0001, 'disp': True})
        return function(res.x)
    
    def get_est_exp_discount_function(self,params):
        """ Returns Estimated Discount function from estimated parameters """
        params = params[0:5]
        df = pd.DataFrame(self.maturity.apply(lambda x: x ** i) for i in range(1, 6)).T
        df.columns = ['M1', 'M2', 'M3', 'M4', 'M5']
        return np.exp(df.dot(params))
    
    def get_Mac_duration(T = 1, k = 1 ):
        return 0
        
class Bond:
    """Represents properties a bond"""
    def __init__(self, facevalue = 100, T = 1, k = 2, coupon = 0):
        """Initialize bond features"""
        self.facevalue = facevalue
        self.T = T
        self.k = k
        self.coupon = coupon
        self.cashflow = np.ones(int(T*k))*(coupon/k)*facevalue
        self.cashflow[-1] = facevalue *(1+coupon/k)
        self.time  = np.arange(1/k,(T*k+1)/k,1/k)
        self.N = self.time.shape[0]
        
    def get_Mac_duration(self, market):
        cashsum = 0
        price = self.get_price(market)
        y = self.get_yield(price)
        for i in range(0,self.N):
            cashsum = cashsum + self.cashflow[i] * self.time[i]/(1+y/self.k)**(self.k*self.time[i])
        return cashsum/price
    
    def get_Mod_duration(self,market):
        price = self.get_price(market)
        Mac = self.get_Mac_duration(market)
        y = self.get_yield(price)
        return Mac/(1+y/self.k)
    
    def get_Dol_duration(self,market,price = 100):
        price = self.get_price(market)
        return self.get_Mod_duration(market)*price/100
    
    def get_DV01(self,market):
        return self.get_Dol_duration(market)/100
    
    def get_convexity(self,market):
        price = self.get_price(market)
        y = self.get_yield(price)
        cov = 0
        for i in range(1,self.N):
            t = i
            cov = cov + 1/(1+y/self.k)**2  *self.facevalue/price * t/self.k * (t+1)/self.k * self.coupon/self.k / (1+y/self.k)**t
        t = self.N
        cov = cov + 1/(1+y/self.k)**2  *self.facevalue/price * t/self.k * (t+1)/self.k * (self.coupon/self.k+1) / (1+y/self.k)**t
        return cov

    def get_price(self,market,cont = 0):
        pricesum = 0
        for i in range(0,self.N):
            spot = market.loc[i,'Spot']
            if cont == 0:
                pricesum = pricesum + self.cashflow[i] /(1+spot/self.k)**((market.loc[i,'Maturity'])*self.k)
            else:
                pricesum = pricesum + self.cashflow[i]*np.exp(-spot*market.loc[i,'Maturity'])
        return pricesum
    
    def get_yield(self,price=100):
        def f(y):
            f = 0
            for i in range(1,self.N):
                f = f + self.coupon/self.k/(1+y/self.k)**i
            f = f + (self.coupon/self.k + 1)/(1+y/self.k)**self.N - price/self.facevalue
            return f
 
        def df(y):
            df = 0 
            for i in range(1,self.N):
                df = df - i/self.k * self.coupon/self.k/(1+y/self.k)**(i+1)
            df = df - self.N/self.k * (self.coupon/self.k + 1)/((1+y/self.k)**(self.N+1))
            return df
        
        def dy(f,y):
            return abs(0-f(y))
        
        def newtons_method(f,df,y0,e):
            delta = dy(f,y0)
            while delta > e:
                y0 = y0 - f(y0)/df(y0)
                delta = dy(f,y0)
            return y0
        
        return newtons_method(f, df, 0, 1e-5)
        
class Hull_and_White_Model:
    """Represents properties a HW Model"""
    def __init__(self, market, sigma = 0.01, kappa = 0.1, T = 5, k = 2):
        self.sigma = sigma
        self.T = T
        self.k = k
        self.kappa = kappa
        self.market= market
        self.N   = int(self.T*self.k) 
    

    
    def Hull_and_White_calibration(self):
        dt  = 1/self.k
        var = (self.sigma**2)/(2*self.kappa)*(1-np.exp(-2*self.kappa*dt))
        M   = np.exp(-self.kappa *dt) -1
        dr  = np.sqrt(3*var)
        jmax= int(max(1,np.ceil(-0.184/M)))
        jmin= -jmax
        self.M = M
        self.dr=dr
        self.jmax = max(1,np.ceil(-0.184/M))
        self.var = var
        prob_df = pd.DataFrame({'j':np.arange(jmax, -jmax-1, -1)})
        prob_df['q_u'] = 1/6 + ((prob_df['j']*M)**2 + prob_df['j']*M)/2
        prob_df['q_m'] = 2/3 -  (prob_df['j']*M)**2
        prob_df['q_d'] = 1/6 + ((prob_df['j']*M)**2 - prob_df['j']*M)/2
        # Type B
        prob_df.iloc[-1,1] =  1/6 + ((prob_df.iloc[-1,0]*M)**2 -   prob_df.iloc[-1,0]*M)/2
        prob_df.iloc[-1,2] = -1/3 -  (prob_df.iloc[-1,0]*M)**2 + 2*prob_df.iloc[-1,0]*M
        prob_df.iloc[-1,3] =  7/6 + ((prob_df.iloc[-1,0]*M)**2 - 3*prob_df.iloc[-1,0]*M)/2
        # Type C
        prob_df.iloc[ 0,1] =  7/6 + ((prob_df.iloc[ 0,0]*M)**2 + 3*prob_df.iloc[ 0,0]*M)/2
        prob_df.iloc[ 0,2] = -1/3 -  (prob_df.iloc[ 0,0]*M)**2 - 2*prob_df.iloc[ 0,0]*M
        prob_df.iloc[ 0,3] =  1/6 + ((prob_df.iloc[ 0,0]*M)**2 +   prob_df.iloc[ 0,0]*M)/2
        
        market_df = pd.DataFrame({'Maturity': np.arange(1/self.k, self.T+1/self.k, 1/self.k)})
        for i in range(0,self.N):
            market_df.loc[i,'Discount Function'] =self.market.loc[self.market['Maturity']==market_df.loc[i,'Maturity'],'Discount Function'].iloc[0]
        
        r0 = -np.log(market_df.loc[0,'Discount Function'])*self.k
        H_W_Tree = np.zeros([jmax-jmin+1,self.N])
        H_W_Tree[5,0] = r0
        
        def DF_Estimation(self,DF,Tree,i,prob_df,jmax):
            def func(m, *args):
                DF,Tree,i,prob_df,jmax = args
                est_DF = np.zeros([2*jmax+1,i+1])
                begin = max(0,jmax - i)
                end   = min(2*jmax,jmax+i)
                est_DF[begin:end+1,-1] = np.exp(-(Tree[begin:end+1,i]+m)/self.k)
                while i > 0:
                    begin = max(1,jmax - i)
                    end   = min(2*jmax,jmax+i+1)
                    est_DF[begin:end,i-1] = est_DF[begin-1:end-1,i]*prob_df.iloc[begin:end,1] + est_DF[begin:end,i]*prob_df.iloc[begin:end,2] + est_DF[begin+1:end+1,i]*prob_df.iloc[begin:end,3]
                    est_DF[begin:end,i-1] = est_DF[begin:end,i-1] * np.exp(-(Tree[begin:end,i-1])/self.k)
                    if i > jmax:
                        est_DF[ 0,i-1] = est_DF[ 0,i]*prob_df.iloc[ 0,1] + est_DF[ 1,i]*prob_df.iloc[ 0,2]+est_DF[ 2,i]*prob_df.iloc[ 0,3]
                        est_DF[ 0,i-1] = est_DF[ 0,i-1] * np.exp(-(Tree[0,i-1])/self.k)
                        est_DF[-1,i-1] = est_DF[-1,i]*prob_df.iloc[-1,1] + est_DF[-2,i]*prob_df.iloc[-1,2]+est_DF[-3,i]*prob_df.iloc[-1,3]
                        est_DF[-1,i-1] = est_DF[-1,i-1] * np.exp(-(Tree[-1,i-1])/self.k)
                    i = i -1
                error = DF-est_DF[jmax,0]
                return error
            m = fsolve(func, 0, args=(DF,Tree,i,prob_df,jmax))
            return m
        
        for i in range(1,self.N):
            H_W_Tree[:,i] =  H_W_Tree[:,i-1]
            if i <=jmax:
                H_W_Tree[jmax-i,i] = H_W_Tree[jmax-i+1,i]+dr
                H_W_Tree[jmax+i,i] = H_W_Tree[jmax+i-1,i]-dr
            
            DF = market_df.loc[i,'Discount Function']
            m = DF_Estimation(self,DF,H_W_Tree,i,prob_df,jmax)
            if i <= jmax:
                H_W_Tree[jmax-i:jmax+i+1,i] = H_W_Tree[jmax-i:jmax+i+1,i] + m
            else:
                H_W_Tree[:,i] = H_W_Tree[:,i] + m

        return H_W_Tree, prob_df
    
class Vasicek_Model:
    '''Initialize Parameters for Vasicek Model'''
    def __init__(self, kappa = 0.1, mu = 0.06, sigma = 0.02):
        self.kappa = kappa
        self.mu    = mu
        self.sigma = sigma

    def Get_Esigma(self,r,t):
        Esigma = self.sigma**2/(2*self.kappa)*(1-np.exp(-2*self.kappa*(t)))
        return Esigma
    
    def get_r(self,r,T):
        '''Returns the r(t,t+T) for given r0 and T'''
        S1 = (self.sigma**2/(2*self.kappa**2))
        B  = (1-np.exp(-self.kappa*T))/self.kappa
        A  = (B-T)*(self.mu - S1) - (self.sigma*B)**2/(4*self.kappa)
        if T == 0:
            return r
        return -A/T + B/T*r
        
    def get_Z(self,r,T,t=0):
        '''Returns the Z(t,t+T) for given r0 and T'''
        S1 = (self.sigma**2/(2*self.kappa**2))
        B  = (1-np.exp(-self.kappa*T))/self.kappa
        A  = (B-T)*(self.mu - S1) - (self.sigma*B)**2/(4*self.kappa)
        St = self.Get_Esigma(r,t) 
        return np.exp(A - B*r +0.5*B**2*St)  

    def Get_CMT(self,r,T):
        '''Calculates CMT for given r and T'''
        S = 0
        iend = int(2*T+1)
        if T >= 1:
            for i in range(1,iend):
                #S = S + 1/(2*(1+self.get_r(r,i/2)/2)**i)
                S = S + np.exp(-self.get_r(r,i/2)*i/2)
            Est_CMT = 2*(1- np.exp(-self.get_r(r,T)*T))/S
            return Est_CMT
        else:
            Est_CMT = (np.exp(self.get_r(r,T)/2)-1)*2
            return Est_CMT
    
    def Get_Er(self,r,t):
        Er = r * np.exp(-self.kappa*t) + self.mu*(1-np.exp(-self.kappa*t))
        return Er
    

    
    def Get_Call_Price(self,r0,t,TO,TB,K):
        EXP1   = np.exp(-2*self.kappa*(TO-t))
        EXP2   = np.exp(-self.kappa*(TB-TO))
        sig    = self.sigma
        kap    = self.kappa
        szsqr  = sig**2*(1-EXP1)/(2*kap)*(1-EXP2)**2/(kap**2)
        rt     = self.Get_Er(r0,t)
        #rt     = r0
        Z1     = self.get_Z(rt,TO-t)
        Z2     = self.get_Z(rt,TB-t)
        if szsqr == 0:
            h = math.inf*np.sign(Z2-K)
        else:
            h  = np.log(Z2/(K*Z1))/np.sqrt(szsqr) + np.sqrt(szsqr)/2
        C      = Z2*norm.cdf(h) - K*Z1*norm.cdf(h-np.sqrt(szsqr))
        return C
        

        
         
       
        
        
        
        
        
        