from math import log,sqrt,exp
from scipy import stats
import random 
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, kstest, norm
from scipy.special import gamma
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# 1
# a)
def BS(security,K,S,r,y,sigma,T,t=0):
    d1=((log(S/K)+(r-y)*(T-t))/(sigma*sqrt(T-t)))+sigma*sqrt(T-t)/2
    d2=((log(S/K)+(r-y)*(T-t))/(sigma*sqrt(T-t)))-sigma*sqrt(T-t)/2
    AONC = S*exp(-y*(T-t))*stats.norm.cdf(d1)
    CONC = exp(-r*(T-t))*stats.norm.cdf(d2)
    AONP = S*exp(-y*(T-t))*stats.norm.cdf(-d1)
    CONP = exp(-r*(T-t))*stats.norm.cdf(-d2)
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
    
k = 2775
s0 = 2775
R = 0.0278
Y=0.0189
sigma_atm = 0.15
T=1
CONP = BS("CONP",K=k,S=s0,r=R,y=Y,sigma=sigma_atm,T=T)
AONP = BS("AONP",K=k,S=s0,r=R,y=Y,sigma=sigma_atm,T=T)
BSP = BS("BSP",K=k,S=s0,r=R,y=Y,sigma=sigma_atm,T=T)
print("CONP:",CONP,"ANOP:",AONP,"BSP:",BSP)

# b)
def sigma(a,b,S,K):
    return a+b*log(S/K)
sigma_imp = sigma(.15,.375,s0,k)
d1 = BS("d1",K=k,S=s0,r=R,y=Y,sigma=sigma_imp,T=T)
d2 = BS("d2",K=k,S=s0,r=R,y=Y,sigma=sigma_imp,T=T)
BLbsp = BS("BSP",K=k,S=s0,r=R,y=Y,sigma=sigma_imp,T=T)
BLconp = exp(-R)*(stats.norm.cdf(-d2) -beta*stats.norm.pdf(-d2));
BLAONP = s0*BLconp -BLbsp
print("BLconp:",BLconp,"BSP:",BLbsp,"BLAONP:",BLAONP)

# c)
k1,k2,k3,k4,k5=2775,2800,2750,2780,2770 
bsp_2775 = BS("BSP",K=k1,S=s0,r=R,y=Y,sigma=sigma(.15,.375,s0,k1),T=T)
bsp_2800 = BS("BSP",K=k2,S=s0,r=R,y=Y,sigma=sigma(.15,.375,s0,k2),T=T)
bsp_2750 = BS("BSP",K=k3,S=s0,r=R,y=Y,sigma=sigma(.15,.375,s0,k3),T=T)
bsp_2780 = BS("BSP",K=k4,S=s0,r=R,y=Y,sigma=sigma(.15,.375,s0,k4),T=T)
bsp_2770 = BS("BSP",K=k5,S=s0,r=R,y=Y,sigma=sigma(.15,.375,s0,k5),T=T)
suprep1 = (bsp_2800-bsp_2775)/25
subrep1 = (bsp_2775-bsp_2750)/25
suprep2 = (bsp_2780-bsp_2775)/5
subrep2 = (bsp_2775-bsp_2770)/5
print("suprep1:",suprep1,"subrep1:",subrep1)
print("suprep2:",suprep2,"subrep2:",subrep2)

# 2
seq = [10**3, 10**4, 10**5, 10**6]
rand_dict = {}
for n in seq:
    np.random.seed(123)
    rand_dict[n] = np.random.uniform(size=n)

# compute sample statistics
def sample_stats(sample_dict, max_min=False):
    stats_df = pd.DataFrame(columns=seq)
    for key in sample_dict:
        stats_df.loc['mean', key] = sample_dict[key].mean()
        stats_df.loc['variance', key] = sample_dict[key].var()
        stats_df.loc['std dev', key] = sample_dict[key].std()
        stats_df.loc['skew', key] = skew(sample_dict[key])
        stats_df.loc['kurtosis', key] = kurtosis(sample_dict[key])
        if max_min:
            stats_df.loc['max', key] = max(sample_dict[key])
            stats_df.loc['min', key] = min(sample_dict[key])

table1 = sample_stats(rand_dict)

# histogram
for key in rand_dict:
    plt.hist(rand_dict[key], bins=10, normed=True)
    plt.title("Histogram of sequence {%s}" % key)
    plt.show()
    return stats_df

# k-s test
for key in rand_dict:
    ks_rlt = kstest(rand_dict[key], 'uniform')
    print("Sequence {%s} has a K-S statistics value: %s"
          % (key, ks_rlt[0]))

# test autocorrelation
for key in rand_dict:
    plot_pacf(rand_dict[key], lags=20, title="Partial Autocorrelation of {%s}" % key)
    plot_pacf(np.power(rand_dict[key], 2), lags=20, title="Partial Autocorrelation of {%s}^2" % key)

# two-dimentional scatter plot
lags = [1, 2, 3]
for lag in lags:
    for key in rand_dict:
        plt.scatter(rand_dict[key][:-lag], rand_dict[key][lag:], s=1)
        plt.title("Scatter plot of {%s} vs lag %s" % (key, lag))
        plt.show()

# 3
# Box-Muller method
def box_muller(U1, U2):
    assert len(U1) == len(U2)
    X = []
    Y = []
    for i in range(len(U1)):
        theta = 2 * np.pi * U1[i]
        R = np.sqrt(-2 * np.log(U2[i]))
        X.append(R * np.cos(theta))
        Y.append(R * np.sin(theta))
    return X, Y

U1 = [np.random.uniform()]
U2 = [np.random.uniform()]
print("Two independent uniform variables (%s, %s) are used to generate two independent normal variables (%s)"
     % (U1, U2, box_muller(U1, U2)))

# sample statistics
normal_dict = {}
for key in rand_dict:
    u_ls = rand_dict[key]
    U1 = u_ls[:int(len(u_ls)/2)]
    U2 = u_ls[int(len(u_ls)/2):]
    X, Y = box_muller(U1, U2)
    normal_dict[key] = np.array(X + Y)
    print("Sequence {%s} of generated normal variables: %s" % (key, X + Y))

table2 = sample_stats(normal_dict, max_min=True)

# histogram
plt.rcParams['figure.figsize'] = [6, 4]
for key in normal_dict:
    seq_min = min(normal_dict[key])
    seq_max = max(normal_dict[key])
    plt.hist(normal_dict[key], bins=int((seq_max-seq_min)/0.1), normed=True)
    plt.title("Histogram of normal sequence {%s}" % key)
    plt.show()

# K-S test
for key in normal_dict:
    ks_rlt = kstest(normal_dict[key], 'norm')
    print("Sequence {%s} has a K-S statistics value: %s"
          % (key, ks_rlt[0]))

# test autocorrelation
for key in normal_dict:
    plot_pacf(normal_dict[key], lags=20, title="Partial Autocorrelation of normal {%s}" % key)
    plot_pacf(np.power(normal_dict[key], 2), lags=20, title="Partial Autocorrelation of {%s}^2" % key)

# two-dimentional scatter plot
plt.rcParams['figure.figsize'] = [10, 4]
lags = [1, 2, 3]
for lag in lags:
    for key in normal_dict:
        plt.subplot(121)
        plt.scatter(normal_dict[key][:-lag], normal_dict[key][lag:], s=1)
        plt.title("Scatter plot of normal {%s} vs lag %s" % (key, lag))
        plt.subplot(122)
        plt.scatter(norm.cdf(normal_dict[key][:-lag]), norm.cdf(normal_dict[key][lag:]), s=1)
        plt.title("Scatter plot of normal cdf {%s} vs lag %s" % (key, lag))
        plt.show()

# 4
# variation properties of SBM
def SBM_variation(m, W, length, n):
    s = 0
    for i in range(1, n+1):
        #print("iteration: %s, value: %s" % (i, np.power(abs(W[int(i * length / n)] - W[int((i - 1) * length / n)]), m)))
        s += np.power(abs(W[int(i * length / n)] - W[int((i - 1) * length / n)]), m)
    return s

plt.rcParams['figure.figsize'] = [10, 4]
inc = 20
j_ls = np.arange(1, inc+1)
m_ls = np.arange(4)
n = 2**(inc)
for m in m_ls:
    true_ls = []
    sbm_ls = []
    for j in j_ls:
        np.random.seed(123)
        dt = 1 / 2**j
        rand_ls = np.sqrt(dt) * np.random.normal(0, 1, size=2**j)
        W = [0] + np.cumsum(rand_ls).tolist()
        true_ls.append((1/2**j)**(m/2-1) * 2**(m / 2) / np.sqrt(np.pi) * gamma((m + 1) / 2))
        sbm_ls.append(SBM_variation(m, W, 2**j, 2**j))
    plt.subplot(121)
    plt.plot(j_ls, np.divide(sbm_ls, true_ls), '-')
    plt.xlabel("Variation Power")
    plt.ylabel("V%s / expected value" % m)
    plt.subplot(122)
    plt.plot(j_ls, np.subtract(sbm_ls, true_ls), '-')
    plt.xlabel("Variation Power")
    plt.ylabel("V%s - expected value" % m)
    plt.show()