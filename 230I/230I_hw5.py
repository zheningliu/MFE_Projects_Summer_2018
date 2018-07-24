import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.sparse.linalg import splu
from hw4.Fixed_Income_Toolbox import *


# data from PS1
df = pd.read_excel('HW1_data.xls')
bond = ZeroCouponBond(100, df['Maturity'])
df['Spot'] = bond.get_spot(df['Price'], k=2)
# estimate term structure
new_df = pd.DataFrame(df['Maturity'].apply(lambda x: x**i) for i in range(1,6)).T
new_df.columns = ['M1','M2','M3','M4','M5']
df['Discount Function'] = bond.get_discount_function(df['Spot'], df['Maturity'], k=2)
new_df['logZ'] = np.log(df['Discount Function'])
rls = sm.ols(formula="logZ ~ %s + 0" % "+".join(new_df.loc[:,'M1':'M5'].columns.tolist()),data=new_df).fit()
(a, b, c, d, e) = rls.params
r0 = - a
r0


# Q1

def theta(t, params=rls.params, kappa=0.15, sigma=0.015):
    a, b, c, d, e = params
    f = - a - 2*b*t - 3*c*t**2 - 4*d*t**3 - 5*e*t**4
    df = - 2*b - 6*c*t - 12*d*t**2 - 20*e*t**3
    return df + kappa * f + sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * t))

def simulate_rates(rand_df, theta, k, r0=r0, kappa=0.15, sigma=0.015):
    delta_t = 1 / k
    r_df = pd.DataFrame([r0] * len(rand_df.index), index=rand_df.index)
    for i in range(len(rand_df.columns)):
        mu = theta(i*delta_t) - kappa * r_df.loc[:, i]
        r_df.loc[:, i+1] = r_df.loc[:, i] + mu * delta_t + sigma * np.sqrt(delta_t) * rand_df.loc[:, i]
    return r_df

k_t = 1000
t_ls = np.linspace(0, 30, k_t+1)
theta_ls = [theta(t, rls.params) for t in t_ls]
plt.plot(t_ls, theta_ls)


# Q2

# monte carlo simulation
np.random.seed(123)
N = 5000
normal_df = pd.DataFrame(np.random.normal(size=(N, 12*10)))
rate_df = simulate_rates(normal_df, theta, k=12)
rate_2yr = rate_df.rolling(24).sum().dropna()
normal_max = rate_2yr.max(axis=1)
normal_min = rate_2yr.min(axis=1)
payout_normal = 10000000 * (normal_max - normal_min)
# anti mc
antinormal_df = - normal_df
anti_df = simulate_rates(antinormal_df, theta, k=12)
anti_2yr = anti_df.rolling(24).sum().dropna()
anti_max = anti_2yr.max(axis=1)
anti_min = anti_2yr.min(axis=1)
payout_anti = 10000000 * (anti_max - anti_min)
payout = (payout_normal + payout_anti) / 2
payout.mean()

# standard error
std_err = np.std(payout) / np.sqrt(N)
std_err


# Q3

# implicit finite difference
def integrate(f, x1, x2, k = 10000):
    ls = np.linspace(x1, x2, k+1)
    f_ls = []
    for i in range(k-1):
        f_ls.append((f(ls[i]) + f(ls[i+1])) / (k * 2))
    return sum(f_ls)
    
def get_value(r_ls, t1, t2, theta, kappa=0.15, sigma=0.015):
    B = lambda tau: (1 - np.exp(-kappa * (t2 - tau))) / kappa
    integrand = lambda tau: B(tau) * theta(tau)
    A = - quad(integrand, t1, t2)[0] + sigma**2 / (2 * kappa**2)         * (t2 - t1 + (1 - np.exp(-2 * kappa * (t2 - t1))) / (2 * kappa) - 2 * B(t1))
    return np.exp(A - B(t1) * r_ls)

def bond_tree(p_ls, cp_ls, r_ls, t_ls, theta, coupon_k=2, kappa=0.15, sigma=0.015):
    value_mat = np.zeros((len(r_ls), len(t_ls)))
    value_mat[...,-1] = p_ls + cp_ls / coupon_k
    for i in reversed(range(len(t_ls)-1)):
        frac_t = (t_ls[i] - np.floor(t_ls[i]*coupon_k)/coupon_k) * coupon_k
        value_mat[...,i] = np.multiply(get_value(pd.Series(r_ls), t_ls[i], t_ls[i+1], theta), value_mat[...,i+1]) + cp_ls / coupon_k * frac_t
    return value_mat

def implicit_fd(p_ls, r_ls, k_r, k_t, theta, coupon_k=2, kappa=0.15, sigma=0.015):
    delta_r = 1 / k_r
    delta_t = 1 / k_t
    V_mat = np.zeros((k_r, k_r))
    for i in range(len(r_ls)):
        down = delta_t * (-sigma**2 / (2 * delta_r**2) + (theta - kappa * r_ls[i]) / (2 * delta_r))
        middle = 1 + delta_t * (sigma**2 / delta_r**2 + r_ls[i])
        up = delta_t * (-sigma**2 / (2 * delta_r**2) - (theta - kappa * r_ls[i]) / (2 * delta_r))
        if i==0:
            V_mat[i][i] = middle
            V_mat[i][i+1] = up
        elif i==len(r_ls)-1:
            V_mat[i][i-1] = down
            V_mat[i][i] = middle
        else:
            V_mat[i][i-1] = down
            V_mat[i][i] = middle
            V_mat[i][i+1] = up
    lu = splu(V_mat)
    x = lu.solve(p_ls.values)
    return x.tolist()

# construct bond price tree
k_r = 100
r_ls = np.linspace(0, 0.2, k_r+1)
bond = ZeroCouponBond(100, pd.Series([5]*len(r_ls)))
par_yield = bond.get_par_yield(r_ls, k=2, coupon_k=2)
bond_k = 100
t_ls = np.linspace(0, 5, bond_k+1)
p_ls = pd.Series([100] * len(r_ls))
cp_ls = pd.Series(par_yield)
bond_df = pd.DataFrame(bond_tree(p_ls, cp_ls, r_ls, t_ls, theta), index=r_ls, columns=t_ls)
bond_df.head()

# construct put option tree using implicit finite differencing
put_df = bond_df.loc[:, :2].copy()
k_r, k_t = put_df.shape
put_df = put_df.applymap(lambda x: max(100-x, 0))
put_t = put_df.columns.tolist()
put_r = put_df.index.tolist()
for i in reversed(range(k_t-1)):
    euro_value = implicit_fd(put_df.iloc[:, i+1], put_r, k_r, k_t, theta(put_t[i]))
    put_df.iloc[:, i] = [max(euro_value[j], put_df.iloc[j, i]) for j in range(k_r)]
put_df.plot(y=0.0, title="Put Price vs. Initial Rate")
put_df.head()

# put option price at r0
put_df.loc[np.floor(r0/0.002)*0.002, 0]


# Q4

def coupon_bond_price(p, cp, r_ls, t_ls, theta, coupon_k=2):
    z_ls = [get_value(r_ls[i], t_ls[i-1], t_ls[i], theta) for i in range(1, len(r_ls))]
    price = p
    for z in reversed(z_ls):
        price = (price + cp / coupon_k) * z
    return price

# monte carlo simulation to find future prices
np.random.seed(123)
N = 10
k = 2
future = []
normal_df = pd.DataFrame(np.random.normal(size=(N, 9*k)))
rate_normal = simulate_rates(normal_df, theta, k=2)
index = 10
r_df = rate_normal
t_ls = np.linspace(0, 9, 9*k+1)
bond_5 = [coupon_bond_price(100, 4, r_df.iloc[i, index:].tolist(), t_ls[index:], theta) for i in range(N)]
print(np.mean(bond_5))
bond_0 = [coupon_bond_price(bond_5[i], 0, r_df.iloc[i, :index].tolist(), t_ls[:index], theta) for i in range(N)]
np.mean(bond_0)



