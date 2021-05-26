"""
This file contains functions for risk free curve calibration
"""

import numpy as np
from scipy.interpolate import splev, splrep, splint

# -----------------------------------------------------
# Construct Mark-to-Market Risk Free Discount Curve
# from discrete term structure data
# -----------------------------------------------------
class Libor_Swap_Curve():

    def __init__(self, tenors, zero_rates):
        n_knots_out = 5
        self.knots = np.hstack(
            (np.linspace(0, tenors[0], n_knots_out), tenors[1:-1], np.linspace(tenors[-1], 50, n_knots_out)))
        self.zero_rates_knots = np.hstack(
            (np.ones(n_knots_out) * zero_rates[0], zero_rates[1:-1], np.ones(n_knots_out) * zero_rates[-1]))
        self.tenors = tenors
        self.zero_rates = zero_rates
        self.tck = splrep(self.knots, self.zero_rates_knots)  # B-spline cubic (d=3)

    def continuous_zero_curve(self):
        return splev(self.tenors, self.tck)

    def continuous_zero_rate(self, t):
        return splev(t, self.tck)

    def continuous_forward_rate(self, t):
        self.derv = splev(t, self.tck, der=1)
        return splev(t, self.tck) + self.derv * t

    def discount_factor(self, start_t, end_t):
        return np.exp(-splint(start_t, end_t, self.tck))


# -----------------------------------------------------
# class to encapsulate risk free rate term structure
# -----------------------------------------------------
class risk_free_curve():

    def __init__(self, tenors, zero_rates):
        self.LSC = Libor_Swap_Curve(tenors, zero_rates)

    def r(self, t):
        return self.LSC.continuous_forward_rate(t)

    def Z(self, t):
        dt = 0.01
        if (t<dt):
            return np.exp(-t*self.r(t))
        else:
            rs = [self.r(s) for s in np.arange(dt,t+1e-4,dt)]
            return np.exp(-dt*np.sum(rs))
        