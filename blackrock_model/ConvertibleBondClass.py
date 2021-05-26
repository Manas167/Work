# -----------------------------------------------------
# class(es) to encapsulate security specific information
# (e.g., contractual details, underlying market data)
# -----------------------------------------------------

import datetime as dt
import numpy as np
from risk_free_curve import risk_free_curve
from cds_function import credit_curve

class ConvertibleBond():

    def __init__(self, contract_info: dict, model_params: dict, rf_curve: risk_free_curve, p=0.0):

        self.p = p  # control elasticity between eq spot and default intensity

        self.m_date = contract_info['maturity_date']
        self.c_rate = contract_info['coupon_rate'] / 100.
        self.c_freq = contract_info['coupon_freq']
        self.notional = contract_info['notional']

        if (('conv_ratio' in contract_info.keys()) and (contract_info['conv_ratio'] != None)):
            self.cv_ratio = contract_info['conv_ratio']
        else:
            self.cv_ratio = self.notional / contract_info['conv_price']

        self.callable = contract_info['callable']
        self.puttable = contract_info['puttable']
        if (self.callable):
            self.call_schedule = contract_info['call_schedule']
        if (self.puttable):
            self.put_schedule = contract_info['put_schedule']

        self.softcall = contract_info['softcall']
        if (self.softcall):
            self.softcallinfo = {
                'start': contract_info['softcall_start'],
                'end': contract_info['softcall_end'],
                'barrier': contract_info['softcall_barrier'],
                'redemption': contract_info['softcall_redempt'] * (self.notional / 100.),  # data was per 100 notional
            }

        # risk free rate term structure
        self.rf_curve = rf_curve

        # hazard rate term structure
        self.rr = model_params['recovery_rate']
        self.ctenors = model_params['credit tenors']
        self.cspread = model_params['credit spread']
        self.cd_curve = credit_curve(self.ctenors, self.cspread, self.rr, self.rf_curve)

        self.eq_spot = model_params['equity_spot']
        self.eq_divd = model_params['equity_dividend_yield']
        self.eq_vol = model_params['equity_flat_vol'] / 100.
        self.eta = model_params['eta']

        # pseudo private variables
        self._y0 = np.log(self.eq_spot)

    def reset_p(self, newp):
        self.p = newp

    def reset_eq_spot(self, new_eq_spot):
        self.eq_spot = new_eq_spot
        self._y0 = np.log(self.eq_spot)

    def r(self, t):
        return self.rf_curve.r(t)

    def Z(self, t):
        return self.rf_curve.Z(t)

    def q(self, t):
        return self.eq_divd

    def lambd(self, t, S):
        y = np.log(S)
        e = self.p * (self._y0 - y)
        h = self.cd_curve.hazard_rate(t)
        return h * np.exp(e)

    def lambd_yspace(self, t, y):
        e = self.p * (self._y0 - y)
        h = self.cd_curve.hazard_rate(t)
        return h * np.exp(e)

    def sigma(self, t):
        return self.eq_vol

    def simulate_eq_spot_paths(self, M, paths, dt):
        np.random.seed(1234567)
        y = np.array([[self._y0] * paths])

        def simulate_next_y(curr_y, t, dt, paths=paths):
            z = np.random.normal(size=paths)
            u = np.random.rand(paths)
            lambd = self.lambd_yspace(t, curr_y)
            drift = (self.r(t) - self.q(t) - (0.5 * self.sigma(t) ** 2) + self.eta * lambd) * dt
            randw = self.sigma(t) * np.sqrt(dt) * z
            jumpN = u <= lambd * dt
            return curr_y + drift + randw - self.eta * jumpN

        for m in range(0, M):
            nexty = simulate_next_y(y[-1], m * dt, dt)
            y = np.vstack((y, nexty))

        return np.exp(y).T  # return paths x T-step matrix

    def soft_call_prob(self, pricing_date, M=20, N=30, paths=1000):
        T2 = (self.softcallinfo['end'] - pricing_date).days / 365.25
        T1 = (self.softcallinfo['start'] - pricing_date).days / 365.25

        dt = 1. / 260  # apply 20-30 soft call logic in business days
        D1 = int(T1 / dt)
        D2 = int(T2 / dt)

        paths = self.simulate_eq_spot_paths(D2, paths, dt)
        self.soft_call_paths = paths[:, D1 - 1:D2 - 1]

        def soft_call_triggered(path, M, N, barrier):
            for i in range(N, len(path)):
                roll_window = path[i - N:i]
                hits = len(roll_window[roll_window >= barrier])
                if (hits >= M):
                    return True
            return False

        b = self.softcallinfo['barrier']
        triggered = np.array([soft_call_triggered(path, M, N, b) for path in self.soft_call_paths])
        soft_call_prob = len(triggered[triggered]) / len(self.soft_call_paths)
        return soft_call_prob

    def soft_call_approx_1of1_barrier(self, pricing_date):
        soft_call_prob = self.soft_call_prob(pricing_date)
        soft_call_paths = self.soft_call_paths
        max_per_path = soft_call_paths.max(axis=1)
        barrier_rank = int(soft_call_prob * len(soft_call_paths))
        barrier_argu = max_per_path.argsort()[-barrier_rank]
        barrier_1of1 = max_per_path[barrier_argu]
        return barrier_1of1

    def fixed_coupon_bond_price(self, pricing_date: dt.datetime):
        pdate_ttm = (self.m_date - pricing_date).days / 365.25
        coupon_dt = 1.0 / float(self.c_freq)
        coupon_ttm = np.arange(0, pdate_ttm + 1e-4, coupon_dt)
        coupon = self.c_rate * coupon_dt * self.notional
        pv_notional = self.notional * self.rf_curve.Z(pdate_ttm)
        pv_coupons = [self.rf_curve.Z(pdate_ttm - tau) * coupon for tau in coupon_ttm]
        return np.sum(pv_coupons) + pv_notional
