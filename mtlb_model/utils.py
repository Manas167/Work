"""
This file contains classes to specify objects such as stocks and interest rates
"""

import datetime
import pandas as pd
import numpy as np
import os


class StockSpec:
    def __init__(self, sigma, S, q):
        self.sigma = sigma
        self.S = S
        self.q = q
        

class ConvertibleBondSpec:
    def __init__(self, T, C, spr, conversion_ratio, call_yrs=None, put_yrs=None,call_strikes=None,
                 put_strikes=None, notional=100, isin=None, issuer=None):
        """

        Notes: this is $100 of notional (careful Bloomberg values may be for $1000)
        """
        self.T = T
        self.C = C
        self.spr = spr
        self.conversion_ratio = conversion_ratio
        self.call_yrs = call_yrs
        self.call_strikes = call_strikes
        self.put_strikes = put_strikes
        self.put_yrs = put_yrs
        self.notional = notional
        self.isin = isin
        self.issuer = issuer


class ForwardRates:
    def __init__(self, maturities, rates):
        """

        :param maturities: zero-coupon bond maturities
        :param rates: zero-coupon bond yields (semi-annually compounded)
        """
        assert len(maturities) == len(rates)
        self.maturities = maturities
        self.rates = rates
        self.beta_hat = None

    def forward_rates(self):
        """

        :return: function
        """
        # do the calibration
        rates_arr = np.array(self.rates)
        mat_arr = np.array(self.maturities)
        zero_prices = 1 / ((1 + rates_arr / 200) ** (mat_arr * 2))
        log_prices = np.log(zero_prices)
        X = np.column_stack([mat_arr,mat_arr**2,mat_arr**3,mat_arr**4,mat_arr**5])
        self.beta_hat = np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ log_prices)

        def forward_curve(x):
            """instantaneous forward curve
            Takes the argument as times in years
            """
            x = np.array(x)
            X_poly = np.column_stack([x, x ** 2, x ** 3, x ** 4, x ** 5])
            prices = np.exp(X_poly @ self.beta_hat)
            return (prices[1:] / prices[0:-1]) ** (-1 / (x[1:] - x[0:-1])) - 1

        return forward_curve


class DataUtils:
    def __init__(self, path):
        self.path = path
        self.data = {}

    @staticmethod
    def open_csv(path):
        df = pd.read_csv(path, sep=',')
        if len(df.columns) == 1:
            df = pd.read_csv(path, sep=';')
        return df

    def load(self):
        files = ['conversion_features_by_isin', 'conversion', 'convert_sched',
                 'interest_rate_term_structure', 'isin_cusip_mapping', 'pricing_data_by_isin',
                 'put_call_info', 'put_call_schedule']

        # loading csv files
        for file in files:
            print('#--- Loading {}.csv ---#'.format(file))
            try:
                df = self.open_csv('{}{}.csv'.format(self.path, file))
                self.data[file] = df
            except:
                print('### Error loading {}.csv ###'.format(file))
                print('Verify that the format is correct and that the file is located in the folder')

    def isin_cusip_mapping(self, isin=None, cusip=None):
        if 'isin_cusip_mapping' not in self.data.keys():
            self.load()
        try:
            if isin:
                cusip = self.data['isin_cusip_mapping'][self.data['isin_cusip_mapping']['isin'] == isin].iloc[0]['cusip']
            elif cusip:
                isin = self.data['isin_cusip_mapping'][self.data['isin_cusip_mapping']['cusip'] == cusip].iloc[0]['isin']
        except:
            print('Can not find mapping for cusip isin', cusip, isin)
            return None
        return cusip

    def put_call_schedule(self, cusip, date):
        if 'put_call_schedule' not in self.data.keys():
            self.load()
        call_yrs = []
        put_yrs = []
        call_strikes = []
        put_strikes = []
        try:
            put_call_schedule = self.data['put_call_schedule'][self.data['put_call_schedule'].cusip == cusip]
            for index, row in put_call_schedule.iterrows():
                yrs = (pd.to_datetime(row.pc_date) - pd.to_datetime(date)).days / 365
                if row.pc_type == 'C':
                    call_yrs.append(yrs)
                    call_strikes.append(row.pc_price)
                elif row.pc_type == 'P':
                    put_yrs.append(yrs)
                    put_strikes.append(row.pc_price)
        except:
            print('Error in loading put_call_schedule for cusip {}'.format(cusip))
        return call_yrs, call_strikes, put_yrs, put_strikes

    def forward_rates(self, date):
        if 'interest_rate_term_structure' not in self.data.keys():
            self.load()

        # create the forward rate function from the term structure for the given date
        rates = self.data['interest_rate_term_structure']
        rates['date'] = pd.to_datetime(rates['date'])
        rates = rates[rates['date'] == pd.to_datetime(date)]
        f = lambda x: int(x[:-1]) if x[-1] == 'Y' else int(x[:-1]) / 12
        maturities = rates['term'].apply(f).values
        rates = rates['rate_zero'].values
        forward_rates = ForwardRates(maturities, rates)
        rates_function = forward_rates.forward_rates()
        return rates_function


class Mapping:
    """
    This class is used to perform a mapping between dates and period defined by steps in order to easily
    work with indexes in the tree and dates (for instance, call and put dates, coupon dates)
    """

    def __init__(self, startDate: datetime.datetime, endDate: datetime.datetime,
                 nb_steps: int):
        """

        :param startDate:
        :param endDate:
        :param nb_steps:
        """
        assert startDate <= endDate
        self.startDate = startDate
        self.endDate = endDate
        self.timeperiod = (endDate - startDate).days
        self.nb_steps = nb_steps
        self.timestep = self.timeperiod / self.nb_steps
        self.timeperiods = [k * self.timestep for k in range(self.nb_steps + 1)]
        self.mapped_from_dates = None
        self.mapped_from_indexes = None

    def get_mapping_from_dates(self):
        """
        Create a mapping from dates to indexes
        :return: dictionary mapping dates in keys to indexes in values, starting indexes at value 0
        """
        assert self.timestep is not None and self.nb_steps is not None
        self.mapped_from_dates = {self.startDate + datetime.timedelta(days=k * self.timestep): k for k in range(self.nb_steps+1)}
        return self.mapped_from_dates

    def get_mapping_from_indexes(self):
        """
        Create a mapping from indexes to dates
        :return: dictionary mapping indexes in keys to dates in values
        """
        assert self.mapped_from_dates is not None
        self.mapped_from_indexes = {v: k for k, v in self.mapped_from_dates.items()}
        return self.mapped_from_indexes

    def find_nearest_date(self, date):
        """
        Find the nearest date from date in mapped_from_dates dictionary
        :return: nearest date and value (index) associated
        """
        i = int(np.argmin([abs((e - date).days) for e in self.mapped_from_dates.keys()]))
        key = list(self.mapped_from_dates.keys())[i]
        return key, self.mapped_from_dates[key]