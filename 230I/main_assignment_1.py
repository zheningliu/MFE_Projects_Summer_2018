import pandas as pd
import numpy as np
from Fixed_Income_Toolbox import ZeroCouponBond as ZCB
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

def main():
    # Question 1
    bond = ZCB(100, 10)
    Q1_a = 20000 * bond.disc_ret(0.06,1,8)
    Q1_b = 20000*bond.disc_ret(0.06,12,8)
    print('Q1')
    print(['Annual Disctete Compounding',Q1_a])
    print(['Monthly Discrete Compounding',Q1_b])

    # Question 2
    r = 0.055
    k = 12 
    wealth = 80000*bond.disc_ret(r,k,20)
    for i in range(0,240):
        wealth = wealth + 3000*bond.disc_ret(r,k,20-(i+1)/12)
    print('Q2')
    print(['Total Wealth',wealth])

    # Question 4
    print('Q4')
    f_0_1 = ((1+0.05)**1)/(1)-1
    f_1_2 = ((1+0.06)**2)/(1+0.05)-1
    f_2_3 = ((1+0.07)**3)/((1+0.06)**2)-1
    print(['f_0_1',f_0_1])
    print(['f_1_2',f_1_2])
    print(['f_2_3',f_2_3])

    borrow = 1000/(1+0.05)
    print(['Borrow this amount for 3 years and invest in 1 year',borrow])

    # Question 5
    r_1 = 0.11
    r_2 = (1.13*(r_1+1))**(1/2)-1
    r_3 = (1.17*(r_2+1)*(r_2+1))**(1/3)-1
    print('Q4')
    print(['r_1',r_1])
    print(['r_2',r_2])
    print(['r_3',r_3])

    # Question 6
    P = 10000*(1-32*0.052/360)
    print('Q6')
    print(['The price of the T Bill is:',P])

    # Question 7
    df = pd.read_excel('HW1_data.xls')
    bond = ZCB(100, df['Maturity'])
    df['Spot'] = bond.get_spot(df['Price'], k=2)
    df['Forward'] = bond.get_forward(df['Spot'], df['Spot'].shift(-1), df['Maturity'], df['Maturity'].shift(-1), k=2)
    df['Par Yield'] = bond.get_par_yield(df['Spot'], k=2, coupon_k=2)
    df.loc[:,'Spot':].plot(title="STRIPS Question 7")
    plt.show()

    # Question 8
    new_df = pd.DataFrame(df['Maturity'].apply(lambda x: x ** i) for i in range(1, 6)).T
    new_df.columns = ['M1', 'M2', 'M3', 'M4', 'M5']
    df['Discount Function'] = bond.get_discount_function(df['Spot'], df['Maturity'], k=2)
    new_df['logZ'] = np.log(df['Discount Function'])
    rls = sm.ols(formula="logZ ~ %s + 0" % "+".join(new_df.columns.tolist()), data=new_df).fit()
    new_df['y_hat'] = rls.predict()
    print(rls.params)
    # Plot Z(T) and predicted z(T)
    plt.plot(df['Maturity'], np.exp(new_df['y_hat']), 'y-', df['Maturity'], df['Discount Function'], 'g-')
    plt.show()

    # Question 9
    est_price_ns = bond.nelson_siegel_estimation(df['Price'], k=2, variation=False)
    estimate_df2 = estimate_term_structure(bond, est_price_ns, ['Spot','Par','Forward6M'])
    estimate_df2.plot(title="Nelson Siegel")
    plt.show()

    # Question 10
    est_price_s = bond.svensson_estimation(df['Price'], k=2)
    estimate_df3 = estimate_term_structure(bond, est_price_s, ['Spot','Par','Forward6M'])
    estimate_df3.plot(title="Svensson")
    plt.show()


def estimate_term_structure(bond, est_price, columns):
    ts_df = pd.DataFrame(columns = columns)
    for col in columns:
        if col == 'Spot':
            ts_df.loc[:,col] = bond.get_spot(est_price, k=2)
        elif col == 'Par':
            ts_df.loc[:,col] = bond.get_par_yield(ts_df['Spot'], k=2, coupon_k=2)
        elif col == 'Forward6M':
            ts_df.loc[:,col] = bond.get_forward(ts_df['Spot'], ts_df['Spot'].shift(-2), bond.maturity, bond.maturity.shift(-2), k=2)
    return ts_df


if __name__ == '__main__':
    main()