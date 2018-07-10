import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import ipdb
from scipy import stats,integrate

# initialization
St = 2775
K = 2775
delta_t = 1
sigma = 0.15
r = 0.0278
y = 0.0189
t1 = 2
t2 = 1
seq = [10**3, 10**4, 10**5, 10**6]

def main():
    # generate random number
    rand_dict = bare_bone_mc_normal(seq)

    # pseudo generate ST using the random series
    ST_dict = {key:simulate_price(rand_dict[key]) for key in rand_dict}

    # ST statistics
    x_stats = sample_stats(ST_dict)

    # transformation of the random series to match moments
    ST_p = {}
    for key in ST_dict:
        args = (rand_dict[key], St, delta_t, sigma, r, y, x_stats.loc['mean', key], x_stats.loc['log variance', key])
        rlt = fsolve(transformation, [0.1, 0.9], args=args)
        print("Sequence {%s} generated result of: %s" % (key, rlt))
        xp = rlt[0] + rlt[1] * rand_dict[key]
        ST_p[key] = simulate_price(xp)

    # risk neutral put price
    t_bb1 = time.time()
    ST_dict = {key:simulate_price(rand_dict[key]) for key in rand_dict}  # repeat to be counted into run time
    Pt = {key:risk_neutral_put(ST_dict[key])[3] for key in ST_dict}
    t_bb = time.time() - t_bb1
    Pt_p = {key:risk_neutral_put(ST_dict[key])[3] for key in ST_dict}

    # bare bone price stats
    Pt_stats = sample_stats(Pt, output=['mean', 'std dev'])
    Ptp_stats = sample_stats(Pt_p, output=['mean', 'std dev'])
    print(Pt_stats, '\n', Ptp_stats)

    # antithetic method
    t_anti1 = time.time()
    P_anti = {key:antithetic(rand_dict[key])[1] for key in rand_dict}
    t_anti = time.time() - t_anti1

    # antithetic price stats
    Panti_stats = sample_stats(P_anti, output=['mean', 'std dev'])

    # ratio test between mc methods
    std_ratio = Pt_stats.loc['std dev', :] / Panti_stats.loc['std dev', :]
    eff_ratio = {key:efficiency_ratio(Panti_stats.loc['std dev', key]**2, Pt_stats.loc['std dev', key]**2, t1, t2) \
                for key in rand_dict}
    eff_real = {key:efficiency_ratio(Panti_stats.loc['std dev', key]**2, Pt_stats.loc['std dev', key]**2, t_anti, t_bb) for key in rand_dict}
    print(t_anti, t_bb, eff_real)

    # plot delta
    epsilon = [0.1, 0.01, 0.001]
    method = ['p', 'm', 'c']
    sample_size = 1000
    S0 = np.linspace(0.9 * St, 1.1 * St, sample_size)
    args = (K, delta_t, r, y, sigma, rand_dict)
    plot_sensitivity(S0, epsilon, method, 'delta', *args, anti=False)

    # plot finite difference gamma
    plot_sensitivity(S0, epsilon, method, 'gamma', *args, anti=False)

    # plot antithetic sensitivities
    plot_sensitivity(S0, epsilon, method, 'delta', *args, anti=True)
    plot_sensitivity(S0, epsilon, method, 'gamma', *args, anti=True)

    # likelihood greeks plot
    greeks_ls = {'Delta':lr_delta, 'Gamma':lr_gamma, 'Rho':lr_rho, 'Vega':lr_vega}
    for key in greeks_ls:
        lr_ls = []
        err_ls = []
        bs_ls = []
        for S in S0:
            sum_payoff, std_payoff = greeks_ls[key](rand_dict[1000], risk_neutral_put, S)
            lr_ls.append(sum_payoff)
            err_ls.append(std_payoff)
            bs_ls.append(bs_put_greeks(S, K, delta_t, r, y, sigma)[key])
        plt.plot(S0, lr_ls, 'b-', S0, bs_ls, 'r-', S0, err_ls, 'g-')
        plt.title("%s Likelihood Ratio vs. Black Scholes" % key)
        plt.legend(['Likelihood Ratio', 'Black Scholes', 'Standard Error'])
        plt.show()

    # part 2a
    SQP_MC_res = {key:SQP_MC(ST[key]) for key in ST} # Store MC result
    SQP_MC_0 = {key:SQP_MC_res[key][0] for key in ST}
    SQP_err = {key:SQP_MC_res[key][1] for key in ST}
    SQP_payoff = {key:SQP_MC_res[key][2] for key in ST}

    # part 2b
    alpha = .5
    y_eff = r - alpha*((r-y)+(alpha-1)*.5*(sigma**2))
    # Transform to power option 
    SQP_BS = np.sqrt(k)*BS("BSP",K=np.sqrt(k),S=np.sqrt(St),r=r,y=y_eff,sigma=abs(alpha)*sigma,T=delta_t)
    t_stat = {key:(SQP_MC_0[key]-SQP_BS)/SQP_err[key] for key in rand_dict}
    print(SQP_BS)
    print(t_stat)

    # part 2c
    print(BL_payoff())

    # part 2d
    BSP_MC_res = {key:risk_neutral_put(ST[key]) for key in ST} # Store MC result
    BSP_MC_0 = {key:BSP_MC_res[key][0] for key in ST}
    BSP_err = {key:BSP_MC_res[key][1] for key in ST}
    BSP_payoff = {key:BSP_MC_res[key][2] for key in ST}
    BSP_MC_0
    mc_cov = {key:np.cov(np.stack((SQP_payoff[key],BSP_payoff[key]))) for key in SQP_payoff}
    mc_corr = {key:np.corrcoef(np.stack((SQP_payoff[key],BSP_payoff[key]))) for key in SQP_payoff}
    print("cov:",mc_cov)
    print("corr:",mc_corr)

    # part 2e
    cvar_MC_res = {key:cvar_MC(ST[key],.5) for key in ST} # Store MC result
    cvar_MC_0 = {key:cvar_MC_res[key][0] for key in ST}
    cvar_err = {key:cvar_MC_res[key][1] for key in ST}
    cvar_payoff = {key:cvar_MC_res[key][2] for key in ST}
    # compare error
    err_diff = {key:SQP_err[key]-cvar_err[key] for key in SQP_err}
    # efficiency
    cvar_eff = {key:efficiency_ratio(BSP_err[key]**2, cvar_err[key]**2,1,2) for key in cvar_err}
    print(cvar_MC_0)
    print(err_diff)
    print(cvar_eff)

    # part 2f&i
    epsilon = [0.1, 0.01, 0.001]
    method = ['c']
    sample_size = 1000
    S0 = np.linspace(0.9 * St, 1.1 * St, sample_size)
    delta_bs = []
    gamma_bs = []
    delta = {}
    for me in method:
        me_df = pd.DataFrame(columns=epsilon)
        for S in S0:
             # finite difference
            for e in epsilon:
                 me_df.loc[S, e] = finite_diff_delta(rand_dict[1000],SQP_MC, S, e, me)
                # bs delta and gamma
            delta, gamma = bs_delta_gamma(S, K, delta_t, r, sigma)
            delta_bs.append(delta)
            gamma_bs.append(gamma)
        me_df.loc[:, 'bs'] = delta_bs
        delta["delta_{0}".format(me)] = me_df.sort_index()
        delta["delta_{0}".format(me)].plot(title="Delta using method %s" % me)
        plt.show()
    print(finite_diff_delta(rand_dict[1000],SQP_MC, S, .1, 'c'))


def bare_bone_mc_normal(size_seq):
    '''Generate a dictionary of random normal series with various length sequence'''
    rand_dict = {}
    for n in size_seq:
        np.random.seed(123)
        rand_dict[n] = np.random.normal(size=n)
    return rand_dict


def simulate_price(x, St=2775, delta_t=1, sigma=0.15, r=0.0278, y=0.0189):
    '''Simulate price using random number series x'''
    return St * np.exp((r - y - sigma**2 / 2) * delta_t + sigma * np.sqrt(delta_t) * x)


def sample_stats(sample_dict, max_min=False, output=[]):
    '''Compute sample statistics'''
    stats_df = pd.DataFrame(columns=sample_dict.keys())
    for key in sample_dict:
        stats_df.loc['mean', key] = np.mean(sample_dict[key])
        stats_df.loc['std dev', key] = np.std(sample_dict[key])
        stats_df.loc['log mean', key] = np.mean(np.log(sample_dict[key]))
        stats_df.loc['log variance', key] = np.var(np.log(sample_dict[key]))
        if max_min:
            stats_df.loc['max', key] = max(sample_dict[key])
            stats_df.loc['min', key] = min(sample_dict[key])
    if not output:
        return stats_df
    else:
        return stats_df.loc[output, :]


def transformation(params, *args):
    '''Moment match algorithm to determine parameters for a tranform series'''
    x, St, delta_t, sigma, r, y, mean, var = args
    xp = params[0] + params[1] * x
    ST_p = simulate_price(xp)
    return [ST_p.mean() - mean, np.log(ST_p).var() - var]


def risk_neutral_put(ST, St=2775, K=2775, delta_t=1, r=0.0278):
    '''Computes risk neural put option price given ST'''
    payoff = [max(s, 0) for s in K - ST]
    payoff_disc = np.multiply(np.exp(- r * delta_t), [max(s, 0) for s in K - ST])
    price = np.exp(- r * delta_t) * np.mean(payoff)
    error = np.exp(- r * delta_t) * np.std(payoff) / np.sqrt(len(ST))
    return price, error, payoff, payoff_disc


def antithetic(x, St=2775):
    '''Antithetic method for variance reduction'''
    x_anti = np.multiply(x, -1)
    ST = simulate_price(x, St)
    ST_anti = simulate_price(x_anti, St)
    PT = 0.5 * np.add(risk_neutral_put(ST)[3], risk_neutral_put(ST_anti)[3])
    return np.mean(PT), PT


def efficiency_ratio(var1, var2, t1, t2):
    '''Ratio tests between Monte Carlo methods'''
    return var1 * t1 / (var2 * t2)


def finite_diff_delta(x, option_payoff, St, epsilon, method, anti=False):
    '''
    x: (list) a list of random variables
    option_payoff: (function) pass in a payoff function, i.e. bs put
    St: (float) S0 value
    epsilon: (float) a number of bump size
    method: (string) delta plus, minus or center
    anti: (boolean) antithetic variates method

    Return: (float) delta of some asset prices using finite difference.
    '''
    ST = simulate_price(x, St)
    Pt = option_payoff(ST, St)[0] if not anti else antithetic(x, St)[0]
    ST_delta = epsilon * St
    if method == 'p':
        St2 = St * (1 + epsilon)
        ST2 = simulate_price(x, St2)
        Pt2 = option_payoff(ST2, St)[0] if not anti else antithetic(x, St2)[0]
    elif method == 'm':
        St2 = St * (1 - epsilon)
        ST2 = simulate_price(x, St2)
        Pt2 = option_payoff(ST2, St2)[0] if not anti else antithetic(x, St2)[0]
        ST_delta *= -1
    elif method == 'c':
        St1 = St * (1 - epsilon)
        St2 = St * (1 + epsilon)
        ST1 = simulate_price(x, St1)
        ST2 = simulate_price(x, St2)
        Pt = option_payoff(ST1, St1)[0] if not anti else antithetic(x, St1)[0]
        Pt2 = option_payoff(ST2, St2)[0] if not anti else antithetic(x, St2)[0]
        ST_delta *= 2
    return (Pt2 - Pt) / ST_delta


def finite_diff_gamma(x, St, epsilon, method='c', anti=False):
    '''Returns gamma of some asset prices using finite difference'''
    ST = simulate_price(x, St)
    ST1 = simulate_price(x, St * (1 - epsilon))
    ST2 = simulate_price(x, St * (1 + epsilon))
    Pt, Pt1, Pt2 = [risk_neutral_put(s)[0] for s in [ST, ST1, ST2]] if not anti \
        else [antithetic(x, s)[0] for s in [St1, St2, St]]
    ST_delta = epsilon * St
    return (Pt1 + Pt2 - 2 * Pt) / (ST_delta ** 2)


def bs_put_greeks(S, K, T, r, y, sigma):
    '''Returns BS greeks associated with a put option'''
    rls = {}
    d1 = (np.log(float(S) / K) + (r - y + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    rls['Delta'] = norm.cdf(d1) - 1
    rls['Gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    rls['Vega'] = S * norm.pdf(d1) * np.sqrt(T)
    rls['Rho'] = K * np.sqrt(T) * np.exp(- r * T) * norm.cdf(-1 * d2)
    return rls


def plot_sensitivity(S0, epsilon, method, greek, *args, anti=False):
    '''Plot greeks using finite differencing'''
    (K, delta_t, r, y, sigma, rand_dict) = args
    if greek == 'delta':
        delta = {}
        for me in method:
            delta_bs = []
            me_df = pd.DataFrame(columns=epsilon)
            for S in S0:
                # finite difference
                for e in epsilon:
                    me_df.loc[S, e] = finite_diff_delta(rand_dict[1000], risk_neutral_put, S, e, me, anti)
                # bs delta
                delta_bs.append(bs_put_greeks(S, K, delta_t, r, y, sigma)['Delta'])
            me_df.loc[:, 'bs'] = delta_bs
            delta["delta_{0}".format(me)] = me_df.sort_index()
            delta["delta_{0}".format(me)].plot(title="Delta using method %s" % me)
            plt.show()
    elif greek == 'gamma':
        gamma_bs = []
        gamma_df = pd.DataFrame(columns=epsilon)
        for S in S0:
            # finite difference
            for e in epsilon:
                gamma_df.loc[S, e] = finite_diff_gamma(rand_dict[1000], S, e)
            # bs gamma
            gamma_bs.append(bs_put_greeks(S, K, delta_t, r, y, sigma)['Gamma'])
        gamma_df.loc[:, 'bs'] = gamma_bs
        gamma_df = gamma_df.sort_index()
        gamma_df.plot(title="Gamma function")
        plt.show()
    return None


def lr_delta(x, option_payoff, St, delta_t=1, sigma=0.15, r=0.0278, y=0.0189):
    '''Returns rho of asset prices uding likelihood ratio method'''
    ST = simulate_price(x, St)
    payoff = np.multiply(x, option_payoff(ST, St)[2])
    sum_payoff = np.exp(- r * delta_t) * np.sum(payoff) / (np.sqrt(delta_t) * St * sigma * len(x))
    std_payoff = np.exp(- r * delta_t) * np.std(payoff) / (np.sqrt(delta_t) * St * sigma)
    return sum_payoff, std_payoff


def lr_gamma(x, option_payoff, St, delta_t=1, sigma=0.15, r=0.0278, y=0.0189):
    '''Returns rho of asset prices uding likelihood ratio method'''
    ST = simulate_price(x, St)
    term = np.subtract((np.power(x,2) - 1) / (sigma**2 * np.sqrt(delta_t) * St**2), x / (sigma * np.sqrt(delta_t) * St**2))
    payoff = np.multiply(term, option_payoff(ST, St)[2])
    sum_payoff = np.exp(- r * delta_t) * np.sum(payoff) / len(x)
    std_payoff = np.exp(- r * delta_t) * np.std(payoff)
    return sum_payoff, std_payoff


def lr_rho(x, option_payoff, St, delta_t=1, sigma=0.15, r=0.0278, y=0.0189):
    '''Returns rho of asset prices uding likelihood ratio method'''
    ST = simulate_price(x, St)
    payoff = np.multiply(x, option_payoff(ST, St)[2])
    sum_payoff = -1 * np.sqrt(delta_t) * np.exp(-1 * r * delta_t) * np.sum(payoff) / (sigma * len(x))
    std_payoff = np.sqrt(delta_t) * np.exp(-1 * r * delta_t) * np.std(payoff) / (sigma)
    return sum_payoff, std_payoff


def lr_vega(x, option_payoff, St, delta_t=1, sigma=0.15, r=0.0278, y=0.0189):
    '''Returns rho of asset prices uding likelihood ratio method'''
    ST = simulate_price(x, St)
    term = np.subtract((np.power(x,2) - 1) / sigma, x * np.sqrt(delta_t))
    payoff = np.multiply(term, option_payoff(ST, St)[2])
    sum_payoff = np.exp(-1 * r * delta_t) * np.sum(payoff)
     / len(x)
    std_payoff = np.exp(-1 * r * delta_t) * np.std(payoff)
    return sum_payoff, std_payoff


def SQP_MC(S_T, K=2775,delta_t=1, r=0.0278):
    "Simulate price of Sqrt Put option given terminal stock price"
    payoff = list(map(lambda x:np.sqrt(K)*(np.sqrt(K)-np.sqrt(x)) if K>x else 0,S_T))
    price=np.exp(-r*delta_t)*np.mean(payoff)
    error=np.exp(-r*delta_t)*np.std(payoff)/np.sqrt(len(S_T))
    return price, error, payoff


def BS(security,K,S,r,y,sigma,T=1,t=0):
    d1=((np.log(S/K)+(r-y)*(T-t))/(sigma*np.sqrt(T-t)))+sigma*np.sqrt(T-t)/2
    d2=((np.log(S/K)+(r-y)*(T-t))/(sigma*np.sqrt(T-t)))-sigma*np.sqrt(T-t)/2
    AONC = S*np.exp(-y*(T-t))*stats.norm.cdf(d1)
    CONC = np.exp(-r*(T-t))*stats.norm.cdf(d2)
    AONP = S*np.exp(-y*(T-t))*stats.norm.cdf(-d1)
    CONP = np.exp(-r*(T-t))*stats.norm.cdf(-d2)
    BSC = AONC - K*CONC
    BSP = K*CONP - AONP 
    if security == "BSC":
        return BSC
    elif security == "BSP": 
        return BSP
    elif security == "CONC": 
        return CONC
    elif security == "CONP":
        return CONP
    elif security == "AONC":
        return AONC
    elif security == "AONP":
        return AONP
    elif security == "d1":
        return d1
    elif security == "d2":
        return d2


def BL_payoff(S=2775, Kstar = 2775, sigma=0.15, r=0.0278, y=0.0189,T=1, t=0):
    def integrand(K,*args):
        d1=((np.log(S/K)+(r-y)*(T-t))/(sigma*np.sqrt(T-t)))+sigma*np.sqrt(T-t)/2
        d2=((np.log(S/K)+(r-y)*(T-t))/(sigma*np.sqrt(T-t)))-sigma*np.sqrt(T-t)/2
        AONP = S*np.exp(-y*(T-t))*stats.norm.cdf(-d1)
        CONP = np.exp(-r*(T-t))*stats.norm.cdf(-d2)
        BSP = K*CONP - AONP 
        F_dd = .25*np.sqrt(K)*(Kstar**(-1.5))
        func = BSP*F_dd
        return func
    args= (S,Kstar,sigma,r,y,T,t)
    result = integrate.quad(integrand, 0, Kstar, args=args)
    return result


def cvar_MC(S_T, alpha, K=2775,delta_t=1, r=0.0278):
    payoff = list(map(lambda x:(np.sqrt(K) - alpha*(np.sqrt(K)+np.sqrt(x)))*(np.sqrt(K)-np.sqrt(x)) if K>x else 0,S_T))
    price=np.exp(-r*delta_t)*np.mean(payoff)
    error=np.exp(-r*delta_t)*np.std(payoff)/np.sqrt(len(S_T))
    return price, error, payoff


if __name__ == "__main__":
    main()


######## Part 2 ###########
# a)
rand_dict = bare_bone_mc_normal(seq)
ST = {key:simulate_price(rand_dict[key]) for key in rand_dict}

def SQP_MC(S_T,St=2775,K=2775,delta_t=1, r=0.0278):
    "Simulate price of Sqrt Put option given terminal stock price"
    payoff = list(map(lambda x:np.sqrt(K)*(np.sqrt(K)-np.sqrt(x)) if K>x else 0,S_T))
    price=np.exp(-r*delta_t)*np.mean(payoff)
    error=np.exp(-r*delta_t)*np.std(payoff)/np.sqrt(len(S_T))
    return price, error, payoff

SQP_MC_res = {key:SQP_MC(ST[key]) for key in ST} # Store MC result
SQP_MC_0 = {key:SQP_MC_res[key][0] for key in ST}
SQP_err = {key:SQP_MC_res[key][1] for key in ST}
SQP_payoff = {key:SQP_MC_res[key][2] for key in ST}
print("SQP price:",SQP_MC_0)
print("std:",SQP_err)

# b)
def BS(security,K,S,r,y,sigma,T=1,t=0):
    d1=((np.log(S/K)+(r-y)*(T-t))/(sigma*np.sqrt(T-t)))+sigma*np.sqrt(T-t)/2
    d2=((np.log(S/K)+(r-y)*(T-t))/(sigma*np.sqrt(T-t)))-sigma*np.sqrt(T-t)/2
    AONC = S*np.exp(-y*(T-t))*stats.norm.cdf(d1)
    CONC = np.exp(-r*(T-t))*stats.norm.cdf(d2)
    AONP = S*np.exp(-y*(T-t))*stats.norm.cdf(-d1)
    CONP = np.exp(-r*(T-t))*stats.norm.cdf(-d2)
    BSC = AONC - K*CONC
    BSP = K*CONP - AONP 
    if security == "BSC":
        return BSC
    elif security == "BSP": 
        return BSP
    elif security == "CONC": 
        return CONC
    elif security == "CONP":
        return CONP
    elif security == "AONC":
        return AONC
    elif security == "AONP":
        return AONP
    elif security == "d1":
        return d1
    elif security == "d2":
        return d2

alpha = .5
y_eff = r - alpha*((r-y)+(alpha-1)*.5*(sigma**2))
# Transform to power option 
SQP_BS = np.sqrt(k)*BS("BSP",K=np.sqrt(k),S=np.sqrt(St),r=r,y=y_eff,sigma=abs(alpha)*sigma,T=delta_t)
t_stat = {key:(SQP_MC_0[key]-SQP_BS)/SQP_err[key] for key in rand_dict}
print(SQP_BS)
print(t_stat)

# c)
def BL_payoff(S=2775, Kstar = 2775, sigma=0.15, r=0.0278, y=0.0189,T=1, t=0):
    def integrand(K,*args):
        d1=((np.log(S/K)+(r-y)*(T-t))/(sigma*np.sqrt(T-t)))+sigma*np.sqrt(T-t)/2
        d2=((np.log(S/K)+(r-y)*(T-t))/(sigma*np.sqrt(T-t)))-sigma*np.sqrt(T-t)/2
        AONP = S*np.exp(-y*(T-t))*stats.norm.cdf(-d1)
        CONP = np.exp(-r*(T-t))*stats.norm.cdf(-d2)
        BSP = K*CONP - AONP 
        F_dd = .25*np.sqrt(K)*(Kstar**(-1.5))
        func = BSP*F_dd
        return func
    args= (S,Kstar,sigma,r,y,T,t)
    result = integrate.quad(integrand, 0, Kstar, args=args)
    return result

BL_payoff()[0]+.5*BS("BSP",K,St,r,y,sigma,delta_t) 

# d)
BSP_MC_res = {key:risk_neutral_put(ST[key]) for key in ST} # Store MC result
BSP_MC_0 = {key:BSP_MC_res[key][0] for key in ST}
BSP_err = {key:BSP_MC_res[key][1] for key in ST}
BSP_payoff = {key:BSP_MC_res[key][2] for key in ST}

mc_cov = {key:np.cov(np.stack((SQP_payoff[key],BSP_payoff[key]))) for key in SQP_payoff}
mc_corr = {key:np.corrcoef(np.stack((SQP_payoff[key],BSP_payoff[key]))) for key in SQP_payoff}
print("cov:",mc_cov)
print("corr:",mc_corr)

# e)
def cvar_MC(S_T, alpha, K=2775,delta_t=1, r=0.0278):
    payoff = list(map(lambda x:(np.sqrt(K) - alpha*(np.sqrt(K)+np.sqrt(x)))*(np.sqrt(K)-np.sqrt(x)) if K>x else 0,S_T))
    price=np.exp(-r*delta_t)*np.mean(payoff)
    error=np.exp(-r*delta_t)*np.std(payoff)/np.sqrt(len(S_T))
    return price, error, payoff

cvar_MC_res = {key:cvar_MC(ST[key],.5) for key in ST} # Store MC result
cvar_MC_0 = {key:cvar_MC_res[key][0] for key in ST}
cvar_err = {key:cvar_MC_res[key][1] for key in ST}
cvar_payoff = {key:cvar_MC_res[key][2] for key in ST}
# compare error
err_diff = {key:SQP_err[key]-cvar_err[key] for key in SQP_err}
# efficiency
cvar_eff = {key:efficiency_ratio(SQP_err[key]**2, cvar_err[key]**2,1,2) for key in cvar_err}
print('price:',cvar_MC_0)
print('std:',cvar_err)
print('efficiency ratio:',cvar_eff)

# f)
# i)
# Numerical derivative of closed form SQP
def SQP_greeks(S, K, T, r, y, sigma):
    rls = {}
    SQP_epsilon = 0.01
    SQP0 = np.sqrt(K)*BS("BSP",np.sqrt(K),np.sqrt(S),r,y,abs(alpha)*sigma,T)
    SQP1 = np.sqrt(K)*BS("BSP",np.sqrt(K),np.sqrt(S-SQP_epsilon),r,y,abs(alpha)*sigma,T)
    SQP2 = np.sqrt(K)*BS("BSP",np.sqrt(K),np.sqrt(S+SQP_epsilon),r,y,abs(alpha)*sigma,T)
    SQP_sigma1 = np.sqrt(K)*BS("BSP",np.sqrt(K),np.sqrt(S),r,y,abs(alpha)*(sigma-SQP_epsilon),T)
    SQP_sigma2 = np.sqrt(K)*BS("BSP",np.sqrt(K),np.sqrt(S),r,y,abs(alpha)*(sigma+SQP_epsilon),T)
    y_eff1 = r - alpha*((r-(y-SQP_epsilon))+(alpha-1)*.5*(sigma**2))
    y_eff2 = r - alpha*((r-(y+SQP_epsilon))+(alpha-1)*.5*(sigma**2))
    SQP_y1 = np.sqrt(K)*BS("BSP",np.sqrt(K),np.sqrt(S),r,y_eff1,abs(alpha)*sigma,T)
    SQP_y2 = np.sqrt(K)*BS("BSP",np.sqrt(K),np.sqrt(S),r,y_eff2,abs(alpha)*sigma,T)
    rls['Delta'] = (SQP2 - SQP1)/(2*SQP_epsilon)
    rls['Gamma'] = (SQP1+SQP2-2*SQP0)/(SQP_epsilon**2)
    rls['Vega'] = (SQP_sigma2 - SQP_sigma1)/(2*SQP_epsilon)
    rls['Rho'] = (SQP_y2 - SQP_y1)/(2*SQP_epsilon)
    return rls

# plot delta
epsilon = [0.1, 0.01, 0.001]
method = ['c']
sample_size = 1000
S0 = np.linspace(0.9 * St, 1.1 * St, sample_size)
delta_bs = []
gamma_bs = []
delta = {}
for me in method:
    me_df = pd.DataFrame(columns=epsilon)
    for S in S0:
         # finite difference
        for e in epsilon:
             me_df.loc[S, e] = finite_diff_delta(rand_dict[1000],SQP_MC, S, e, me)
        # bs delta and gamma
        delta_bs.append(SQP_greeks(S,K,delta_t,r,y_eff,sigma)['Delta'])
    me_df.loc[:, 'bs'] = delta_bs
    delta["delta_{0}".format(me)] = me_df.sort_index()
    delta["delta_{0}".format(me)].plot(title="Sqrt Put BBMC Delta")
    plt.show()

# ii) Plot Gamma
gamma_bs = []
gamma_df = pd.DataFrame(columns=epsilon)
for S in S0:
    # finite difference
    for e in epsilon:
        gamma_df.loc[S, e] = finite_diff_gamma2(rand_dict[1000],SQP_MC, S, e)
    # bs gamma
    gamma_bs.append(SQP_greeks(S,K,delta_t,r,y_eff,sigma)['Gamma'])
gamma_df.loc[:, 'bs'] = gamma_bs
gamma_df = gamma_df.sort_index()
gamma_df.plot(title="Sqrt Put BBMC Gamma")
plt.show()

# iii)
greeks_ls = {'Delta':lr_delta, 'Gamma':lr_gamma, 'Rho':lr_rho, 'Vega':lr_vega}
for key in greeks_ls:
    lr_ls = []
    bs_ls = []
    for S in S0:
        sum_payoff, std_payoff = greeks_ls[key](rand_dict[1000],SQP_MC, S)
        lr_ls.append(sum_payoff)
        bs_ls.append(SQP_greeks(S,K,delta_t,r,y_eff,sigma)[key])
    plt.plot(S0, lr_ls, 'b-', S0, bs_ls, 'r-')
    plt.title("%s Likelihood Ratio vs. Closed Form" % key)
    plt.legend(['Likelihood Ratio', 'Closed Form'])
    plt.show()

# Standard Error
for key in greeks_ls:
    err_ls = []
    for S in S0:
        sum_payoff, std_payoff = greeks_ls[key](rand_dict[1000],SQP_MC, S)
        err_ls.append(std_payoff)
    plt.plot(S0, err_ls, 'g-')
    plt.title("%s Likelihood Ratio vs. Closed Form" % key)
    plt.legend(['Likelihood Ratio', 'Closed Form'])
    plt.show()

# g)
def cvar_price(S_T, St=2775, alpha=0.5,K=2775,delta_t=1, r=0.0278,y=0.0189,sigma=0.15):
    bsp = BS("BSP",K,St,r,y,sigma,delta_t)
    payoff = list(map(lambda x:(np.sqrt(K) - alpha*(np.sqrt(K)+np.sqrt(x)))*(np.sqrt(K)-np.sqrt(x)) if K>x else 0,S_T))
    price=np.exp(-r*delta_t)*np.mean(payoff) + alpha*bsp
    error=np.exp(-r*delta_t)*np.std(payoff)/np.sqrt(len(S_T))
    return price, error, payoff

#cvar_price_res = {key:cvar_price(ST[key]) for key in ST} # Store MC result
cvar_price_0 = {key:cvar_price_res[key][0] for key in ST}
cvar_price_err = {key:cvar_price_res[key][1] for key in ST}
print("cvar_price:", cvar_price_0)

# i)
# plot delta
epsilon = [0.1, 0.01, 0.001]
method = ['c']
sample_size = 1000
S0 = np.linspace(0.9 * St, 1.1 * St, sample_size)
delta_bs = []
gamma_bs = []
delta = {}
for me in method:
    me_df = pd.DataFrame(columns=epsilon)
    for S in S0:
         # finite difference
        for e in epsilon:
             me_df.loc[S, e] = finite_diff_delta(rand_dict[1000],cvar_price, S, e, me)
        # bs delta and gamma
        delta_bs.append(SQP_greeks(S,K,delta_t,r,y_eff,sigma)['Delta'])
    me_df.loc[:, 'bs'] = delta_bs
    delta["delta_{0}".format(me)] = me_df.sort_index()
    delta["delta_{0}".format(me)].plot(title="Sqrt Put Control Variate Delta")
    plt.show()

def finite_diff_gamma2(x,option_payoff, St, epsilon, method='c', anti=False):
    '''Returns gamma of some asset prices using finite difference'''
    ST = simulate_price(x, St)
    ST1 = simulate_price(x, St * (1 - epsilon))
    ST2 = simulate_price(x, St * (1 + epsilon))
    Pt = option_payoff(ST, St)[0] if not anti else antithetic(x, St)[0]
    Pt1 = option_payoff(ST1, St * (1 - epsilon))[0] if not anti else antithetic(x, St * (1 - epsilon))[0]
    Pt2 = option_payoff(ST2, St * (1 + epsilon))[0] if not anti else antithetic(x, St * (1 + epsilon))[0]
    ST_delta = epsilon * St
    return (Pt1 + Pt2 - 2 * Pt) / (ST_delta ** 2)

# ii) Plot Gamma
gamma_bs = []
gamma_df = pd.DataFrame(columns=epsilon)
for S in S0:
    # finite difference
    for e in epsilon:
        gamma_df.loc[S, e] = finite_diff_gamma2(rand_dict[1000],cvar_price, S, e)
    # bs gamma
    gamma_bs.append(SQP_greeks(S,K,delta_t,r,y_eff,sigma)['Gamma'])
gamma_df.loc[:, 'bs'] = gamma_bs
gamma_df = gamma_df.sort_index()
gamma_df.plot(title="Sqrt Put Control Variate Gamma")
plt.show()

# iii)
greeks_ls = {'Delta':lr_delta, 'Gamma':lr_gamma, 'Rho':lr_rho, 'Vega':lr_vega}
for key in greeks_ls:
    lr_ls = []
    bs_ls = []
    for S in S0:
        sum_payoff = greeks_ls[key](rand_dict[1000],cvar_price, S)[0]+.5*bs_put_greeks(S, K, delta_t, r, y, sigma)[key]
        lr_ls.append(sum_payoff)
        bs_ls.append(SQP_greeks(S,K,delta_t,r,y_eff,sigma)[key])
    plt.plot(S0, lr_ls, 'b-', S0, bs_ls, 'r-')
    plt.title("%s Likelihood Ratio vs. Closed Form" % key)
    plt.legend(['Likelihood Ratio', 'Closed Form'])
    plt.show()

# Standard Error
for key in greeks_ls:
    err_ls = []
    for S in S0:
        std_payoff = greeks_ls[key](rand_dict[1000],SQP_MC, S)[1]
        err_ls.append(std_payoff)
    plt.plot(S0, err_ls, 'g-')
    plt.title("%s Likelihood Ratio vs. Closed Form" % key)
    plt.legend(['Likelihood Ratio', 'Closed Form'])
    plt.show()