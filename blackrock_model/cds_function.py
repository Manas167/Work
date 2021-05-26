# -----------------------------------------------------
# Function library for Hazard Rate Calibration
# -----------------------------------------------------
import numpy as np
from risk_free_curve import risk_free_curve

# -----------------------------------------------------
# This function requires: Tenors (years), CDS par spread(bps),
# Discount curve (units), Recovery rate (units)
# -----------------------------------------------------
def CDS_Calibration(Tenors, CDS_par_spread, Discount_curve, RR):
    # Vectors to store values:
    dt = np.zeros_like(Tenors)
    P_t = np.zeros(len(Tenors) + 1)
    Lambda_t = np.zeros(len(Tenors))
    P_t[0] = 1

    # Discrete dt:
    dt[0] = Tenors[0]
    dt[1:] = np.diff(Tenors)

    # Define L:
    L = 1 - RR

    # JPMorgan method's Boostrapping of Cumulative survival Prob:
    Summ = 0
    for t, s in enumerate(CDS_par_spread):
        if t == 0:
            P_t[t + 1] = L / (s / 10000.0 * dt[t] + L)
            Lambda_t[t] = -np.log(P_t[t + 1]) / dt[t]
        else:
            Numerator = 0.0
            for i in range(t):
                Numerator += Discount_curve[i] * (L * P_t[i] - (L + dt[i] * s / 10000.0) * P_t[i + 1])

            Denominator = Discount_curve[t] * (L + dt[t] * s / 10000.0)
            P_t[t + 1] = Numerator / Denominator + P_t[t] * L / (L + s / 10000.0 * dt[t])
            Lambda_t[t] = (-np.log(P_t[t + 1]) - np.dot(Lambda_t[:t], dt[:t])) / dt[t]

    return Lambda_t


# -----------------------------------------------------
# class to encapsulate hazard rate term structure
# -----------------------------------------------------
class credit_curve():

    def __init__(self, tenors, spread, recovery_rate, rf_curve: risk_free_curve):
        self.rf_curve = rf_curve
        self.tenors = tenors
        self.spread = spread
        self.rr = recovery_rate
        self._calibrate()

    def _calibrate(self):
        discount_curve = [self.rf_curve.Z(t) for t in self.tenors]
        self.piecewise_hr = CDS_Calibration(self.tenors, self.spread, discount_curve, self.rr)

    def hazard_rate(self, t):
        for i in range(len(self.tenors)):
            if (t <= self.tenors[i]):
                return self.piecewise_hr[i]
        return self.piecewise_hr[-1]