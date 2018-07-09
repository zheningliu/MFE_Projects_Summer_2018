import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import time
import matplotlib.pyplot as plt

# part a
St = 2775
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
    ST = {key:simulate_price(rand_dict[key]) for key in rand_dict}

    # ST statistics
    x_stats = sample_stats(ST)

    # transformation of the random series to match moments
    ST_p = {}
    for key in ST:
        args = (rand_dict[key], St, delta_t, sigma, r, y, x_stats.loc['mean', key], x_stats.loc['log variance', key])
        rlt = fsolve(transformation, [0.1, 0.9], args=args)
        print("Sequence {%s} generated result of: %s" % (key, rlt))
        xp = rlt[0] + rlt[1] * rand_dict[key]
        ST_p[key] = simulate_price(xp)

    # risk neutral put price
    t_bb1 = time.time()
    ST = {key:simulate_price(rand_dict[key]) for key in rand_dict}  # repeat to be counted into run time
    Pt = {key:risk_neutral_put(ST[key]) for key in ST}
    t_bb = time.time() - t_bb1
    Pt_p = {key:risk_neutral_put(ST[key]) for key in ST}

    # bare bone price stats
    Pt_stats = sample_stats(Pt, output=['mean', 'std dev'])
    Ptp_stats = sample_stats(Pt_p, output=['mean', 'std dev'])
    print(Pt_stats, '\n', Ptp_stats)

    # antithetic method
    t_anti1 = time.time()
    P_anti = {key:antithetic(rand_dict[key]) for key in rand_dict}
    t_anti = time.time() - t_anti1

    # antithetic price stats
    Panti_stats = sample_stats(P_anti, output=['mean', 'std dev'])

    # ratio test between mc methods
    std_ratio = Pt_stats.loc['std dev', :] / Panti_stats.loc['std dev', :]
    eff_ratio = {key:efficiency_ratio(Panti_stats.loc['std dev', key]**2, Pt_stats.loc['std dev', key]**2, t1, t2) \
                for key in rand_dict}
    eff_real = {key:efficiency_ratio(Panti_stats.loc['std dev', key]**2, Pt_stats.loc['std dev', key]**2, t_anti, t_bb) for key in rand_dict}
    print(t_anti, t_bb, eff_real)

    # plot finite difference delta
    epsilon = [0.1, 0.01, 0.001]
    method = ['p', 'm', 'c']
    delta = {}
    for me in method:
        delta["delta_{0}".format(me)] = pd.DataFrame({e:finite_diff_delta(rand_dict[1000], St, e, me) for e in epsilon})
        delta["delta_{0}".format(me)] = delta["delta_{0}".format(me)].set_index(ST_dict[1000]).sort_index()
        delta["delta_{0}".format(me)].plot(title="Delta using method %s" % me, xlim=(0.9 * St, 1.1 * St))

    # plot bs delta
    sample_size = 1000
    S0 = np.linspace(0.9 * St, 1.1 * St, sample_size)
    delta_bs = []
    gamma_bs = []
    for S in S0:
        delta, gamma = bs_delta_gamma(S, K, delta_t, r, sigma)
        delta_bs.append(delta)
        gamma_bs.append(gamma)
    plt.plot(S0, delta_bs)

    # plot finite difference gamma
    gamma = pd.DataFrame({e:finite_diff_gamma(ST_dict[1000], e) for e in epsilon})
    gamma = gamma.set_index(ST_dict[1000]).sort_index()
    gamma.plot(title="Gamma function")

    # BS gamma
    plt.plot(S0, gamma_bs)


def bare_bone_mc_normal(size_seq):
    '''Generate a dictionary of random normal series with various length sequence'''
    rand_dict = {}
    for n in seq:
        np.random.seed(123)
        rand_dict[n] = np.random.normal(size=n)
    return rand_dict


def simulate_price(x, St=2775, delta_t=1, sigma=0.15, r=0.0278, y=0.0189):
    '''Simulate price using random number series x'''
    return St * np.exp((r - y - sigma**2 / 2) * delta_t + sigma * np.sqrt(delta_t) * x)


def sample_stats(sample_dict, max_min=False, output=[]):
    '''Compute sample statistics'''
    stats_df = pd.DataFrame(columns=seq)
    for key in sample_dict:
        stats_df.loc['mean', key] = sample_dict[key].mean()
        stats_df.loc['std dev', key] = sample_dict[key].std()
        stats_df.loc['log mean', key] = np.log(sample_dict[key]).mean()
        stats_df.loc['log variance', key] = np.log(sample_dict[key]).var()
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


def risk_neutral_put(ST, K=2775, delta_t=1, r=0.0278):
    '''Computes risk neural put option price given ST'''
    return np.multiply(np.exp(- r * delta_t), [max(s, 0) for s in K - ST])


def antithetic(x):
    '''Antithetic method for variance reduction'''
    x_anti = np.multiply(x, -1)
    ST = simulate_price(x)
    ST_anti = simulate_price(x_anti)
    PT = 0.5 * np.add(risk_neutral_put(ST), risk_neutral_put(ST_anti))
    return PT


def efficiency_ratio(var1, var2, t1, t2):
    '''Ratio tests between Monte Carlo methods'''
    return var1 * t1 / (var2 * t2)


def finite_diff_delta(x, St, epsilon, method, ST_anti=np.array([])):
    '''Returns delta of some asset prices using finite difference'''
    ST = simulate_price(x, St)
    Pt = risk_neutral_put(ST) if ST_anti.size==0 else antithetic(x, St)
    ST_delta = epsilon * St
    if method == 'p':
        St2 = St * (1 + epsilon)
        ST2 = simulate_price(x, St2)
        Pt2 = risk_neutral_put(ST2) if ST_anti.size==0 else antithetic(x, St2)
    elif method == 'm':
        St2 = St * (1 - epsilon)
        ST2 = simulate_price(x, St2)
        Pt2 = risk_neutral_put(ST2) if ST_anti.size==0 else antithetic(x, St2)
    elif method == 'c':
        St1 = St * (1 - epsilon)
        St2 = St * (1 + epsilon)
        ST1 = simulate_price(x, St1)
        ST2 = simulate_price(x, St2)
        Pt = risk_neutral_put(ST1) if ST_anti.size==0 else antithetic(x, St1)
        Pt2 = risk_neutral_put(ST2) if ST_anti.size==0 else antithetic(x, St2)
        ST_delta *= 2
    raise ValueError
    return (Pt2 - Pt) / ST_delta


def finite_diff_gamma(ST, epsilon, method='c'):
    '''Returns gamma of some asset prices using finite difference'''
    ST_delta = ST * epsilon
    ST1 = ST - ST_delta
    ST2 = ST + ST_delta
    Pt, Pt1, Pt2 = [risk_neutral_put(s) for s in [ST, ST1, ST2]]
    return (Pt1 + Pt2 - 2 * Pt) / (ST_delta ** 2)


def bs_delta_gamma(S, K, T, r, sigma):
    '''Returns BS greeks associated with a put option'''
    d1 = (np.log(float(S) / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    Delta = norm.cdf(d1) - 1
    Gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return Delta, Gamma


if __name__ == "__main__":
    main()



