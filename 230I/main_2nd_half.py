import pandas as pd
import numpy as np
from Fixed_Income_Toolbox import ZeroCouponBond as ZCB
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

def main():
    df = pd.read_excel('HW1_data.xls')
    bond = ZCB(100, df['Maturity'])
    df['Spot'] = bond.get_spot(df['Price'], k=2)
    df['Forward'] = bond.get_forward(df['Spot'], df['Spot'].shift(-1), df['Maturity'], df['Maturity'].shift(-1), k=2)
    df['Par Yield'] = bond.get_par_yield(df['Spot'], k=2, coupon_k=2)
    df.loc[:,'Spot':].plot(title="STRIPS Question 7")
    plt.show()

    # Regress logZ on M1-M5
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

    # Nelson Siegel Estimation
    est_price_ns = bond.nelson_siegel_estimation(df['Price'], k=2, variation=False)
    estimate_df2 = estimate_term_structure(bond, est_price_ns, ['Spot','Par','Forward6M'])
    estimate_df2.plot(title="Nelson Siegel")
    plt.show()

    # Svensson Estimation
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


if __name__ == "__main__":
    main()