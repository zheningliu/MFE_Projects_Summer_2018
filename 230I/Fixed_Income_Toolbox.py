# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:06:22 2018

@author: Tom Zhang & Nini Liu
"""

import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize


class ZeroCouponBond:
    """Represents properties of zero coupon bonds"""
    def __init__(self, fv, maturity):
        """Initialize bond features"""
        self.fv = fv
        self.maturity = maturity


    def __cont_ret(self, rate, T, t = 0):
        """Returns continuous compounded return over time period T1 to T2"""
        return np.exp(rate * (T - t))


    def __disc_ret(self, rate, k, T, t = 0):
        """Returns discrete compounded return over time period T1 to T2"""
        return np.power(1 + rate / k, k * (T - t))


    def __future_value(self, X, spot, T, t = 0, k = 1):
        """Returns future value of a bond. Default to annually compounded"""
        if k == 0:
            return X * self.__cont_ret(spot, T, t)
        else:
            return X * self.__disc_ret(spot, k, T, t)


    def __present_value(self, C, spot, T, t = 0, k = 1):
        """Returns present value of a bond. Default to annually compounded"""
        if k == 0:
            return C / self.__cont_ret(spot, T, t)
        else:
            return C / self.__disc_ret(spot, k, T, t)
    

    def get_discount_function(self, spot, T, t = 0, k = 1):
        """Returns discount rate of a bond"""
        return self.__present_value(1, spot, T, t, k)
    

    def get_spot(self, price, t = 0, k = 1):
        """Returns spot rate of a bond given bond price"""
        if k == 0:
            return - np.log(self.fv/price) / (self.maturity - t)
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
        ret_T1 = self.__disc_ret(spot1, k, T1, t)
        ret_T2 = self.__disc_ret(spot2, k, T2, t)
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
