import numpy as np
import pandas as pd
from scipy.optimize import fsolve

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
    Pt = {key:risk_neutral_put(rand_dict[key]) for key in ST}
    Pt_p = {key:risk_neutral_put(rand_dict[key]) for key in ST}

    # bare bone price stats
    Pt_stats = sample_stats(Pt, output=['mean', 'std dev'])
    Ptp_stats = sample_stats(Pt_p, output=['mean', 'std dev'])
    print(Pt_stats, '\n', Ptp_stats)

    # antithetic method
    P_anti = {key:antithetic(rand_dict[key]) for key in rand_dict}

    # antithetic price stats
    Panti_stats = sample_stats(P_anti, output=['mean', 'std dev'])

    # ratio test between mc methods
    std_ratio = Pt_stats.loc['std dev', :] / Panti_stats.loc['std dev', :]
    eff_ratio = {key:efficiency_ratio(Panti_stats.loc['std dev', key]**2, Pt_stats.loc['std dev', key]**2, t1, t2) \
                for key in rand_dict}


def bare_bone_mc_normal(size_seq):
    '''Generate a dictionary of random normal series with various length sequence'''
    rand_dict = {}
    for n in seq:
        np.random.seed(123)
        rand_dict[n] = np.random.normal(size=n)
    return rand_dict


def simulate_price(x, St=2775, delta_t=1, sigma=0.15, r=0.0278, y=0.0189):
    '''Simulate price using random number series x'''
    return St * np.exp((r - y - sigma**2 / 2) * delta_t + sigma * delta_t * x)


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


def risk_neutral_put(x, K=2775, delta_t=1, r=0.0278):
    '''Computes risk neural put option price given ST'''
    ST = simulate_price(x)
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


if __name__ == "__main__":
    main()



