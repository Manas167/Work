"""
This file contains the trinomial tree architecture
"""
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import re
import sys, os
sys.path.append(os.getcwd())
try:
    from utils import ConvertibleBondSpec, StockSpec
except:
    from cbv_models.mtlb_model.utils import ConvertibleBondSpec, StockSpec


class STTStock:
    def __init__(self, stock, conv_bond, forward_curve_func, N_steps=500):
        assert N_steps > 0
        self.T = conv_bond.T
        self.C = conv_bond.C
        self.spr = conv_bond.spr
        self.S = stock.S*conv_bond.conversion_ratio
        self.sigma = stock.sigma
        self.q = stock.q
        self.N_steps = N_steps
        self.conversion_ratio = conv_bond.conversion_ratio
        self.call_yrs = conv_bond.call_yrs
        self.put_yrs = conv_bond.put_yrs
        self.call_strikes=conv_bond.call_strikes
        self.put_strikes = conv_bond.put_strikes
        if self.call_yrs is None:
            self.call_yrs = [conv_bond.T-1]
            self.call_strikes = [np.inf]
        if self.put_yrs is None:
            self.put_yrs = [conv_bond.T-1]
            self.put_strikes = [-np.inf]
        self.forward_curve = forward_curve_func


    def price_conv_bond(self):

        def find_nearest(array, value):
            return (np.abs(np.asarray(array) - value)).argmin()

        # Divide the Coupon by 2 for convention
        c = self.C / 2

        # Keep track of the years when cash flows occur
        cf_yrs = np.sort(np.arange(self.T, 0.5, -0.5))

        # Keep track of the coupons that occur in the above years
        cf = np.array([self.C * 100 / 2 for idx in cf_yrs])

        # Add the final notional at the end
        cf[-1] += 100

        # Break the Maturity into N_steps+1
        T_yrs = np.arange(0, self.T + self.T / self.N_steps, self.T / self.N_steps)

        # Helper function
        find_nearest_custom = lambda x: find_nearest(T_yrs, x)

        # This helps us replace the nearest years with the cash flow years
        T_yrs[list(map(find_nearest_custom, cf_yrs))] = cf_yrs

        # Difference between time grid values
        dt = np.diff(T_yrs)
		
        #### Callable and Putable functionalities
        def get_call_strike(yr,call_dict):
            try:
                return call_dict[yr]
            except:
                return np.inf
        T_yrs[list(map(find_nearest_custom,self.call_yrs))] = self.call_yrs
        call_steps = list(map(find_nearest_custom,self.call_yrs))
        call_dict = dict(zip(self.call_yrs,self.call_strikes))
        call_T = list(map(lambda x:get_call_strike(x,call_dict),T_yrs))
        def get_put_strike(yr,put_dict):
            try:
                return put_dict[yr]
            except:
                return -np.inf
        T_yrs[list(map(find_nearest_custom,self.put_yrs))] = self.put_yrs
        put_steps = list(map(find_nearest_custom,self.put_yrs))
        put_dict = dict(zip(self.put_yrs,self.put_strikes))
        put_T = list(map(lambda x:get_put_strike(x,put_dict),T_yrs))
        ####################################

        # Initialize the Stock Price Tree
        S_T = np.zeros([2 * self.N_steps + 1, self.N_steps + 1])

        # Generate forward rates
        f = self.forward_curve(T_yrs)

        # Calculate the up, down, and middle moves in Trinomial Tree
        u = np.exp(self.sigma * np.sqrt(2 * dt) - dt * self.q)
        d = np.exp(-self.sigma * np.sqrt(2 * dt) - dt * self.q)
        m = np.exp(-dt * self.q)

        # Temporary variables to calculate the Q-probabilities
        r_help = np.exp(f * dt / 2)  # Helping Variables
        u_help = np.exp(self.sigma * np.sqrt(dt / 2))  # Helping Variables
        d_help = np.exp(-self.sigma * np.sqrt(dt / 2))  # Helping Variables

        # Calculate the Q-probabilities for trinomial tree
        q_u = ((r_help - d_help) / (u_help - d_help)) ** 2
        q_d = ((u_help - r_help) / (u_help - d_help)) ** 2
        q_m = 1 - q_u - q_d

        # The first value of the stock price tree is the current stock price
        S_T[0, 0] = self.S
        
        
        # These values will help calculate gamma in the end
        S_T_gamma_up = self.S*np.sqrt(u[0])
        S_T_gamma_down = self.S*np.sqrt(d[0])
        

        # Temporary variables useful to generate the stock price tree
        u_help2 = np.exp(self.sigma * np.sqrt(2 * dt))
        d_help2 = np.exp(-self.sigma * np.sqrt(2 * dt))
        m_help2 = np.exp(-dt * self.q)

        # Temporary variable useful to generate the stock price tree
        Stock_help = np.array([self.S])

        # Using this loop, generate the prices going forward in the tree
        for idx in range(1, self.N_steps + 1):
            Stock_up_end = Stock_help[0]
            Stock_down_end = Stock_help[-1]
            Stock_help = np.insert(Stock_help * m_help2[idx - 1], [0, Stock_help.size], values
            =[Stock_up_end * u_help2[idx - 1] * m_help2[idx - 1],
              Stock_down_end * d_help2[idx - 1] * m_help2[idx - 1]])
            S_T[0:(2 * idx + 1), idx] = Stock_help

        # A similar operation as before, except that we generate a sparse array
        # of N_steps, with cashflows occuring at the cf_yrs
        cf_T = np.zeros(self.N_steps + 1)

        for jdx in range(len(cf_yrs)):
            cf_T[find_nearest(T_yrs, cf_yrs[jdx])] = cf[jdx]

        C_T = np.zeros([2 * self.N_steps + 1, self.N_steps + 1])
        A_T = np.zeros([2 * self.N_steps + 1, self.N_steps + 1])

        # Pricing Algorithm
        # A_T is the action tree
        # C_T is the Convertible Bond Tree
        # H are the holding values
        
        def vanilla_convert(idx, S_T, A_T, C_T, H, q_u, q_m, q_d):
            A_T[0:(2 * idx + 1), idx] = np.maximum(S_T[0:(2 * idx + 1), idx] > H,
                                                   A_T[0:(2 * idx + 1), idx + 1] * q_u[idx]
                                                   + A_T[1:(2 * idx + 2), idx + 1] * q_m[idx] + A_T[2:(2 * idx + 3),
                                                                                                idx + 1] * q_d[idx])
            # max(Stock Value, Holding Value)
            C_T[0:(2 * idx + 1), idx] = np.maximum(S_T[0:(2 * idx + 1), idx], H)
            
            
            
            
        def call_put_convert(idx,S_T,A_T,C_T,H,q_u,q_m,q_d,call_T,put_T,cf_T):
            
            C_T[0:(2*idx+1),idx] = np.maximum(np.maximum(S_T[0:(2*idx+1),idx],
                                                         np.minimum(H,call_T[idx]+cf_T[idx])),put_T[idx]+cf_T[idx])
            A_T[0:(2*idx+1),idx] = np.maximum(S_T[0:(2*idx+1),idx]>H,
                                              A_T[0:(2*idx+1),idx+1]*q_u[idx] + A_T[1:(2*idx+2),idx+1]*q_m[idx] + A_T[2:(2*idx+3),idx+1]*q_d[idx])

                
            A_T[(C_T[:,idx] == (put_T[idx]+cf_T[idx])) | (C_T[:,idx] == (call_T[idx]+cf_T[idx])),idx] = 0

        for idx in range(self.N_steps, -1, -1):
            if idx == self.N_steps:
                C_T[0:(2 * idx + 1), idx] = np.maximum(cf_T[idx], S_T[0:(2 * idx + 1), idx])
                A_T[0:(2 * idx + 1), idx] = S_T[0:(2 * idx + 1), idx] > cf_T[idx]

            else:
                # This is same for every type of convertible, Sum of present values + any coupon at that time
                H = C_T[0:(2 * idx + 1), idx + 1] * q_u[idx] * np.exp(
                    -dt[idx] * (f[idx] + self.spr * (1 - A_T[0:(2 * idx + 1), idx + 1]))) + C_T[1:(2 * idx + 2), idx + 1] * \
                    q_m[idx] * np.exp(
                    -dt[idx] * (f[idx] + self.spr * (1 - A_T[1:(2 * idx + 2), idx + 1]))) + C_T[2:(2 * idx + 3), idx + 1] * \
                    q_d[idx] * np.exp(
                    -dt[idx] * (f[idx] + self.spr * (1 - A_T[2:(2 * idx + 3), idx + 1]))) + cf_T[idx]

                #vanilla_convert(idx, S_T, A_T, C_T, H, q_u, q_m, q_d)
                call_put_convert(idx,S_T,A_T,C_T,H,q_u,q_m,q_d,call_T,put_T,cf_T)

#         delta = (C_T[0, 1] - C_T[2, 1]) / (S_T[0, 1] - S_T[2, 1])

#         gamma = ((C_T[0, 1] - C_T[1, 1]) / (S_T[0, 1] - S_T[1, 1]) + (C_T[1, 1] - C_T[2, 1]) / (
#                     S_T[1, 1] - S_T[2, 1])) / ((S_T[0, 1] - S_T[2, 1]) / 2)


        delta  = (C_T[0,1]-C_T[1,1])/(S_T[0,1]-S_T[1,1])
        
        gamma = ( (C_T[0,1]-C_T[1,1])/(S_T[0,1]-S_T[1,1])
                 - (C_T[1,1]-C_T[2,1])/(S_T[1,1]-S_T[2,1]) ) /(S_T_gamma_up/S_T_gamma_down-1)

        res = {'price': C_T[0, 0], 'delta': delta, 'gamma': gamma}

        return res


class Sensitivities:

    def __init__(self, sttstock):
        self.sttstock = sttstock

    def delta(self, l, plot=True):
        res = []
        initial_value = self.sttstock.S
        for idx in l:
            self.sttstock.S = idx
            res.append(self.sttstock.price_conv_bond()['delta'])
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(l, res, label='Delta')
            plt.xlabel('S')
            plt.ylabel('delta')
            plt.grid()
            plt.legend()
            plt.title('Delta')
        self.sttstock.S = initial_value
        return res

    def prices(self, l, plot=True):
        res = []
        initial_value = self.sttstock.S
        for idx in l:
            self.sttstock.S = idx
            res.append(self.sttstock.price_conv_bond()['price'])
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(l, res, label='Convertible bond price')
            plt.xlabel('S')
            plt.ylabel('Convertible bond price')
            plt.grid()
            plt.legend()
            plt.title('Price')
        self.sttstock.S = initial_value
        return res

    def gamma(self, l, plot=True):
        res = []
        initial_value = self.sttstock.S
        for idx in l:
            self.sttstock.S = idx
            res.append(self.sttstock.price_conv_bond()['gamma'])
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(l, res, label='Gamma')
            plt.xlabel('S')
            plt.ylabel('gamma')
            plt.grid()
            plt.legend()
            plt.title('Gamma')
        self.sttstock.S = initial_value
        return res

    @staticmethod
    def first_order_derivative(f_plus, f_minus, h):
        return (f_plus - f_minus) / (2 * h)

    @staticmethod
    def second_order_derivative(f, f_plus, f_minus, h):
        return (f_plus - 2 * f + f_minus) / h ** 2

    def delta_FD(self, l, bump=.01, plot=True):
        res = []
        initial_value = self.sttstock.S
        for idx in l:
            self.sttstock.S = idx * (1 + bump)
            f_plus = self.sttstock.price_conv_bond()['price']
            self.sttstock.S = idx * (1 - bump)
            f_minus = self.sttstock.price_conv_bond()['price']
            res.append(self.first_order_derivative(f_plus, f_minus, idx * bump))
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(l, res, label='Delta (Finite differencing')
            plt.xlabel('S')
            plt.ylabel('delta')
            plt.grid()
            plt.legend()
            plt.title('Delta')
        self.sttstock.S = initial_value
        return res

    def gamma_FD(self, l, bump=.01, plot=True):
        res = []
        initial_value = self.sttstock.S
        for idx in l:
            self.sttstock.S = idx
            f = self.sttstock.price_conv_bond()['price']
            self.sttstock.S = idx * (1 + bump)
            f_plus = self.sttstock.price_conv_bond()['price']
            self.sttstock.S = idx * (1 - bump)
            f_minus = self.sttstock.price_conv_bond()['price']
            res.append(self.second_order_derivative(f, f_plus, f_minus, idx * bump))
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(l, res, label='Gamma (Finite differencing')
            plt.xlabel('S')
            plt.ylabel('gamma')
            plt.grid()
            plt.legend()
            plt.title('Gamma')
        self.sttstock.S = initial_value
        return res

    def vega_FD(self, l, bump=.1, plot=True):
        res = []
        initial_value = self.sttstock.sigma
        for idx in l:
            self.sttstock.sigma = idx * (1 + bump)
            f_plus = self.sttstock.price_conv_bond()['price']
            self.sttstock.sigma = idx * (1 - bump)
            f_minus = self.sttstock.price_conv_bond()['price']
            res.append(self.first_order_derivative(f_plus, f_minus, idx * bump))
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(l, res, label='Vega (Finite differencing)')
            plt.xlabel('S')
            plt.ylabel('vega')
            plt.grid()
            plt.legend()
            plt.title('Vega')
        self.sttstock.sigma = initial_value
        return res

    def vega_delta(self, l, bump_vega=.1, plot=True):
        """
        Vega as a function of delta
        :param l:
        :param bump_vega:
        :return:
        """
        res_delta, res_vega = [], []
        initial_value = self.sttstock.S
        for idx in l:
            self.sttstock.S = idx
            res_delta.append(self.sttstock.price_conv_bond()['delta'])
            res_vega += self.vega_FD([self.sttstock.sigma], bump=bump_vega, plot=False)
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(res_delta, res_vega, label='Vega vs. delta')
            plt.xlabel('delta')
            plt.ylabel('vega')
            plt.grid()
            plt.legend()
            plt.title('Vega vs. delta')
        self.sttstock.S = initial_value
        return res_delta, res_vega

    def sensitivities_plot(self, S, sigma, bump_vega=.1):
        """
        Plot all the sensitivies
        :param bump_vega:
        :return:
        """
        plt.figure(figsize=(15, 10))
        prices, deltas, gammas, vegas = [], [], [], []
        initial_value = self.sttstock.S
        for idx in S:
            self.sttstock.S = idx
            res = self.sttstock.price_conv_bond()
            prices.append(res['price'])
            deltas.append(res['delta'])
            gammas.append(res['gamma'])
        self.sttstock.S = initial_value
        initial_value = self.sttstock.sigma
        for idx in sigma:
            self.sttstock.sigma = idx
            vegas += self.vega_FD([idx], bump=bump_vega, plot=False)
        self.sttstock.sigma = initial_value

        plt.subplot(2, 2, 1)
        plt.plot(S, prices, label='Price')
        plt.xlabel('S')
        plt.ylabel('price')
        plt.grid()
        plt.legend()
        plt.title('Price')

        plt.subplot(2, 2, 2)
        plt.plot(S, deltas, label='Delta')
        plt.xlabel('S')
        plt.ylabel('delta')
        plt.grid()
        plt.legend()
        plt.title('Delta')

        plt.subplot(2, 2, 3)
        plt.plot(S, gammas, label='Gamma')
        plt.xlabel('S')
        plt.ylabel('gamma')
        plt.grid()
        plt.legend()
        plt.title('Gamma')

        plt.subplot(2, 2, 4)
        plt.plot(sigma, vegas, label='Vega')
        plt.xlabel('sigma')
        plt.ylabel('vega')
        plt.grid()
        plt.legend()
        plt.title('Vega')

        return prices, deltas, gammas, vegas


class Valuation:
    def __init__(self, data_utils):
        """
        Class used to perform time series valuation
        :param data_utils: DataUtils object, with attribute data containing dataframes
        """
        self.data_utils = data_utils
        self.res = None
        self.compare_res = None

    def price(self, path_output=None, progress_bar=True, N_steps=500):
        res = {'date': [], 'isin': [], 'issuer': [], 'price': [], 'delta': [], 'gamma': [],
               'equity_price': []}
        errors = {}
        n = len(self.data_utils.data['pricing_data_by_isin'])
        progress = ProgressBar(n, fmt=ProgressBar.FULL) if progress_bar else None
        # using the pricing_data_by_isin.csv file
        for index, row in self.data_utils.data['pricing_data_by_isin'].iterrows():
            try:
                T = (pd.to_datetime(row.maturity_date) - pd.to_datetime(row.date)).days / 365
                C = row.coupon_annualized / 100
                spr = row.credit_spread / 100 / 100
                if 'isin_cusip_mapping' in self.data_utils.data.keys():
                    conversion_ratio = self.data_utils.data['isin_cusip_mapping'][
                                           self.data_utils.data['isin_cusip_mapping']['isin'] == row['isin']].iloc[0]['conversion_ratio'] / 10
                    cusip = self.data_utils.isin_cusip_mapping(isin=row['isin'])
                    notional = self.data_utils.data['isin_cusip_mapping'][
                        self.data_utils.data['isin_cusip_mapping']['isin'] == row['isin']].iloc[0]['par_amt']
                    call_yrs, call_strikes, put_yrs, put_strikes = self.data_utils.put_call_schedule(cusip, row.date)
                else:
                    conversion_ratio = self.data_utils.data['conversion_features_by_isin'][
                                           self.data_utils.data['conversion_features_by_isin']['isin'] == row[
                                               'isin']].iloc[0]['conversion_ratio'] / 10
                    notional = self.data_utils.data['conversion_features_by_isin'][
                        self.data_utils.data['conversion_features_by_isin']['isin'] == row['isin']].iloc[0]['par_amt']
                    call_yrs, call_strikes, put_yrs, put_strikes = None, None, None, None
                isin = row['isin']
                issuer = row.issuer
                S = row.ul_eq_price
                sigma = row.ul_eq_vol / 100
                q = row.ul_eq_divyld / 100
                conv_bond = ConvertibleBondSpec(T, C, spr, conversion_ratio, call_yrs, put_yrs, call_strikes,
                                                put_strikes, notional, isin, issuer)
                stock = StockSpec(sigma, S, q)
                rates_function = self.data_utils.forward_rates(row.date)
                pricing = STTStock(stock, conv_bond, rates_function, N_steps=N_steps)
                result = pricing.price_conv_bond()
                res['date'].append(row.date)
                res['isin'].append(isin)
                res['issuer'].append(issuer)
                res['price'].append(result['price'])
                res['delta'].append(result['delta'])
                res['gamma'].append(result['gamma'])
                res['equity_price'].append(S)
            except Exception as e:
                errors[row['isin']] = e
            if progress and index % (n // 100) == 0:
                progress.current = index
                progress()
        if progress:
            progress.done()
        res = pd.DataFrame(res)
        if path_output:
            res.to_csv(path_output)
        print('# Errors #')
        print(errors.keys())

        self.res = res
        return res

    @staticmethod
    def resample_data(res, freq):
        res.index = pd.to_datetime(res['date'])
        res.sort_index(inplace=True)
        return res.resample(freq).last().reset_index(drop=True)

    def compare_results(self, plot=True, resample=None, plot_xlabel=False):
        """

        :param plot:
        :param resample:
        :param plot_xlabel: in PyCharm or Jupyter lab, indexing of xlabels takes a lot of time, use plot_xlabel=False
        to have the code run, it should work with plot_xlabel=True in Jupyter notebook
        :return:
        """
        data = self.data_utils.data['pricing_data_by_isin']
        data = data[['date', 'isin', 'convertible_price',
                     'delta', 'gamma']].rename(columns={'convertible_price': 'convertible_price_given',
                                                        'delta': 'delta_given', 'gamma': 'gamma_given'})
        res = self.res.merge(data, left_on=['date', 'isin'], right_on=['date', 'isin'], how='inner')
        res['delta_given'] = res['delta_given'] / 100
        if resample:
            res = self.resample_data(res, resample)

        if plot:
            for e in res['isin'].unique():
                try:
                    temp_df = res[res['isin'] == e].set_index('date').sort_index()
                    issuer = temp_df['issuer'].iloc[0]
                    print('# Issuer: {} #'.format(issuer))
                    if plot_xlabel:
                        fig = plt.figure(figsize=(20, 5))
                        ax = fig.add_subplot(131)
                        ax.plot(temp_df.index, temp_df['price'], label='Model price')
                        ax.plot(temp_df.index, temp_df['convertible_price_given'], label='Market price')
                        plt.xticks(temp_df.index[::int(len(temp_df) * .95) // 5],
                                   [str(e)[:7] for e in temp_df.index[::int(len(temp_df) * .95) // 5]])

                        plt.xlabel('Date')
                        plt.ylabel('Price')
                        plt.title('Prices')
                        plt.legend()
                        plt.grid()

                        ax = fig.add_subplot(132)
                        ax.plot(temp_df.index, temp_df['delta'], label='Model delta')
                        ax.plot(temp_df.index, temp_df['delta_given'], label='Market delta')
                        plt.xticks(temp_df.index[::int(len(temp_df) * .95) // 5],
                                   [str(e)[:7] for e in temp_df.index[::int(len(temp_df) * .95) // 5]])
                        plt.ylim((0., 1.05))
                        plt.xlabel('Date')
                        plt.ylabel('Delta')
                        plt.title('Delta')
                        plt.legend()
                        plt.grid()

                        ax = fig.add_subplot(133)
                        ax.plot(temp_df.index, temp_df['gamma'], label='Model gamma')
                        ax.plot(temp_df.index, temp_df['gamma_given'], label='Market gamma')
                        plt.xticks(temp_df.index[::int(len(temp_df) * .95) // 5],
                                   [str(e)[:7] for e in temp_df.index[::int(len(temp_df) * .95) // 5]])
                        plt.xlabel('Date')
                        plt.ylabel('Gamma')
                        plt.title('Gamma')
                        plt.legend()
                        plt.grid()
                        plt.show()
                    else:
                        fig = plt.figure(figsize=(20, 5))
                        ax = fig.add_subplot(131)
                        ax.plot(temp_df['price'], label='Model price')
                        ax.plot(temp_df['convertible_price_given'], label='Market price')
                        plt.xticks([])

                        plt.xlabel('Date')
                        plt.ylabel('Price')
                        plt.title('Prices')
                        plt.legend()
                        plt.grid()

                        ax = fig.add_subplot(132)
                        ax.plot(temp_df['delta'], label='Model delta')
                        ax.plot(temp_df['delta_given'], label='Market delta')
                        plt.xticks([])

                        plt.ylim((0., 1.05))
                        plt.xlabel('Date')
                        plt.ylabel('Delta')
                        plt.title('Delta')
                        plt.legend()
                        plt.grid()

                        ax = fig.add_subplot(133)
                        ax.plot(temp_df['gamma'], label='Model gamma')
                        ax.plot(temp_df['gamma_given'], label='Market gamma')
                        plt.xticks([])

                        plt.xlabel('Date')
                        plt.ylabel('Gamma')
                        plt.title('Gamma')
                        plt.legend()
                        plt.grid()
                        plt.show()
                except:
                    print('Error: ', issuer)
                    pass

        self.compare_res = res
        return self.compare_res

    def strategy_implied_delta(self, resample=None, plot_xlabel=False):
        # construct the table of equity and convert returns and implied returns for each isin
        for e in self.compare_res['isin'].unique():
            try:
                res = self.compare_res[self.compare_res['isin'] == e]
                if resample:
                    res = self.resample_data(res, resample)
                res = res.set_index('date').sort_index()
                issuer = res['issuer'].iloc[0]
                print('# Issuer: {} #'.format(issuer))
                res[['model_return', 'market_return', 'equity_return']] = (res[['price', 'convertible_price_given', 'equity_price']] / res[['price', 'convertible_price_given', 'equity_price']].shift() - 1)
                res['equity_implied_return'] = res['delta'] * res['equity_return']

                # compute the difference between the market returns and the equity implied return
                res['spread_return'] = res['market_return'] - res['equity_implied_return']

                res.dropna(inplace=True)

                # study autocorrelation of spread return
                plot_pacf(res['spread_return'], lags=20)
                plt.title('PACF')
                plt.show()
                adf_test = adfuller(res['spread_return'], 1)
                print('ADF test: ', adf_test[0], 'p-value ', adf_test[1])

                # plot cumulative returns and equity implied returns (mean reversion)
                res['cumulative_market_return'] = 1 + res['market_return']
                res['cumulative_market_return'].iloc[0] = 1
                res['cumulative_market_return'] = res['cumulative_market_return'].cumprod()
                res['cumulative_equity_implied_return'] = 1 + res['equity_implied_return']
                res['cumulative_equity_implied_return'].iloc[0] = 1
                res['cumulative_equity_implied_return'] = res['cumulative_equity_implied_return'].cumprod()
                res[['cumulative_market_return', 'cumulative_equity_implied_return']].plot()
                plt.show()

                # study spread returns
                if plot_xlabel:
                    plt.figure(figsize=(8, 6))
                    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
                    ax0 = plt.subplot(gs[0])
                    ax0.plot(res.index, res['convertible_price_given'], label='Market price')
                    plt.legend()
                    plt.grid()
                    plt.xticks([])
                    plt.title('Spread vs market price')
                    ax1 = plt.subplot(gs[1])
                    ax1.plot(res.index, res['spread_return'], label='Spread', color='k')
                    plt.xticks(res.index[::int(len(res) * .95) // 5],
                               [str(e)[:7] for e in res.index[::int(len(res) * .95) // 5]])
                    plt.legend()
                    plt.grid()
                    plt.tight_layout()
                    plt.show()
                else:
                    plt.figure(figsize=(8, 6))
                    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
                    ax0 = plt.subplot(gs[0])
                    ax0.plot(res['convertible_price_given'], label='Market price')
                    plt.legend()
                    plt.grid()
                    plt.xticks([])
                    plt.title('Spread vs market price')
                    ax1 = plt.subplot(gs[1])
                    ax1.plot(res['spread_return'], label='Spread', color='k')
                    plt.xticks([])
                    plt.legend()
                    plt.grid()
                    plt.tight_layout()
                    plt.show()

                # trading convergence of underpriced convertible bond
                if plot_xlabel:
                    signals = res[res['spread_return'] < res['spread_return'].quantile(.1)]
                    plt.plot(res.index, res['convertible_price_given'], label='Market price')
                    plt.scatter(signals.index, signals['convertible_price_given'])
                    plt.xticks(res.index[::int(len(res) * .95) // 5],
                               [str(e)[:7] for e in res.index[::int(len(res) * .95) // 5]])
                    plt.show()
                else:
                    signals = res[res['spread_return'] < res['spread_return'].quantile(.1)]
                    plt.plot(res['convertible_price_given'], label='Market price')
                    plt.scatter(signals.index, signals['convertible_price_given'])
                    plt.xticks([])
                    plt.show()

                # trading analysis
                if plot_xlabel:
                    plt.figure(figsize=(8, 6))
                    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
                    ax0 = plt.subplot(gs[0])
                    ax0.plot(res.index, res['cumulative_market_return'], label='Cumulative market return')
                    ax0.plot(res.index, res['cumulative_equity_implied_return'], label='Cumulative equity implied return')
                    plt.legend()
                    plt.grid()
                    plt.xticks([])
                    plt.title('Spread vs market price')
                    ax1 = plt.subplot(gs[1])
                    ax1.plot(res.index, res['spread_return'], label='Spread', color='k')
                    plt.xticks(res.index[::int(len(res) * .95) // 5],
                               [str(e)[:7] for e in res.index[::int(len(res) * .95) // 5]])
                    plt.legend()
                    plt.grid()
                    plt.tight_layout()
                    plt.show()
                else:
                    plt.figure(figsize=(8, 6))
                    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
                    ax0 = plt.subplot(gs[0])
                    ax0.plot(res['cumulative_market_return'], label='Cumulative market return')
                    ax0.plot(res['cumulative_equity_implied_return'], label='Cumulative equity implied return')
                    plt.legend()
                    plt.grid()
                    plt.xticks([])
                    plt.title('Spread vs market price')
                    ax1 = plt.subplot(gs[1])
                    ax1.plot(res['spread_return'], label='Spread', color='k')
                    plt.xticks([])
                    plt.legend()
                    plt.grid()
                    plt.tight_layout()
                    plt.show()

            except:
                print('Error: ', issuer)
                pass

    def trading(self, up_trade, down_trade, up_out, down_out, resample=None, plot_xlabel=False):
        # construct the table of equity and convert returns and implied returns for each isin
        for e in self.compare_res['isin'].unique():
            try:
                res = self.compare_res[self.compare_res['isin'] == e]
                if resample:
                    res = self.resample_data(res, resample)
                res = res.set_index('date').sort_index()
                issuer = res['issuer'].iloc[0]
                print('# Issuer: {} #'.format(issuer))
                res[['model_return', 'market_return', 'equity_return']] = (
                            res[['price', 'convertible_price_given', 'equity_price']] / res[
                        ['price', 'convertible_price_given', 'equity_price']].shift() - 1)
                res['equity_implied_return'] = res['delta'] * res['equity_return']

                # compute the difference between the market returns and the equity implied return
                res['spread_return'] = res['market_return'] - res['equity_implied_return']

                res.dropna(inplace=True)

                # plot cumulative returns and equity implied returns (mean reversion)
                res['cumulative_market_return'] = 1 + res['market_return']
                res['cumulative_market_return'].iloc[0] = 1
                res['cumulative_market_return'] = res['cumulative_market_return'].cumprod()
                res['cumulative_equity_implied_return'] = 1 + res['equity_implied_return']
                res['cumulative_equity_implied_return'].iloc[0] = 1
                res['cumulative_equity_implied_return'] = res['cumulative_equity_implied_return'].cumprod()

                # trading analysis
                if not all([up_trade, down_trade, up_out, down_out]):
                    up_trade = res['spread_return'].quantile(.95)
                    down_trade = res['spread_return'].quantile(.05)
                    up_out = res['spread_return'].quantile(.75)
                    down_out = res['spread_return'].quantile(.25)
                ptf_values = [1]
                current_position = 0
                indexes_short, indexes_long = [], []
                for i in range(len(res['spread_return'])-1):
                    if res['spread_return'].iloc[i] > up_trade:
                        current_position = -1
                        indexes_short.append(res.index[i])
                    if res['spread_return'].iloc[i] < down_trade:
                        current_position = 1
                        indexes_long.append(res.index[i])
                    if res['spread_return'].iloc[i] < down_out and current_position == -1:
                        current_position = 0
                    if res['spread_return'].iloc[i] > up_out and current_position == 1:
                        current_position = 0
                    ptf_values.append(ptf_values[-1] + ptf_values[-1] * current_position * res['market_return'].iloc[i+1])
                if plot_xlabel:
                    plt.figure(figsize=(8, 6))
                    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
                    ax0 = plt.subplot(gs[0])
                    ax0.plot(res.index, res['cumulative_market_return'], label='Cumulative market return')
                    ax0.plot(res.index, res['cumulative_equity_implied_return'], label='Cumulative equity implied return')
                    ax0.plot(res.index, ptf_values, label='Cumulative portfolio return')
                    plt.legend()
                    plt.grid()
                    plt.xticks([])
                    plt.title('Spread vs market price')
                    ax1 = plt.subplot(gs[1])
                    ax1.plot(res.index, res['spread_return'], label='Spread', color='k')
                    plt.axhline(up_trade, c='orange', label='trade upper bound')
                    plt.axhline(down_trade, c='orange', label='trade lower bound')
                    plt.axhline(up_out, c='b', label='out upper bound')
                    plt.axhline(down_out, c='b', label='out lower bound')
                    plt.xticks(res.index[::int(len(res) * .95) // 5],
                               [str(e)[:7] for e in res.index[::int(len(res) * .95) // 5]])
                    plt.legend()
                    plt.grid()
                    plt.tight_layout()
                    plt.show()
                else:
                    plt.figure(figsize=(8, 6))
                    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
                    ax0 = plt.subplot(gs[0])
                    ax0.plot(res['cumulative_market_return'], label='Cumulative market return')
                    ax0.plot(res['cumulative_equity_implied_return'], label='Cumulative equity implied return')
                    ax0.plot(res.index, ptf_values, label='Cumulative portfolio return')
                    plt.legend()
                    plt.grid()
                    plt.xticks([])
                    plt.title('Spread vs market price')
                    ax1 = plt.subplot(gs[1])
                    ax1.plot(res['spread_return'], label='Spread', color='k')
                    plt.axhline(up_trade, c='orange', label='trade upper bound')
                    plt.axhline(down_trade, c='orange', label='trade lower bound')
                    plt.axhline(up_out, c='b', label='out upper bound')
                    plt.axhline(down_out, c='b', label='out lower bound')
                    plt.xticks([])
                    plt.legend()
                    plt.grid()
                    plt.tight_layout()
                    plt.show()

                # sharpe ratio
                print('Sharpe ratio')
                temp_ptf_values = [ptf_values[i] - ptf_values[i-1] for i in range(1, len(ptf_values))]
                print(np.mean(temp_ptf_values) / np.std(temp_ptf_values) * 252 ** .5)

            except:
                print('Error: ', issuer)
                pass


class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)
