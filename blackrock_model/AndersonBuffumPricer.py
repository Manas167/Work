# ------------------------------------------------------
# Crank-Nicolson Finite Difference Solver for
# Andersen-Buffum Convertible Bond model
# ------------------------------------------------------
import numpy as np
import datetime as dt
from ConvertibleBondClass import ConvertibleBond


# -----------------------------------------------------
# an efficient tridiagonal-matrix solver
# -----------------------------------------------------
def TDMA(a, b, c, d):
    # tridiagonal-matrix solver: a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector
    n = len(d)
    w = np.zeros(n - 1, float)
    g = np.zeros(n, float)
    p = np.zeros(n, float)

    w[0] = c[0] / b[0]
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    p[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]

    return p


# ------------------------------------------------------
# Crank-Nicolson Finite Difference Solver for
# Andersen-Buffum Convertible Bond model
# ------------------------------------------------------
class AndersenBuffumPricer():
    theta = 0.5  # keep it as 0.5 to use Crank-Nicolson Method
    y_grid_coverage = 4  # no of stddev covered by the y-grid

    def __init__(self, sec: ConvertibleBond, pdate: dt.datetime, dt=1 / 48, dy=0.05):
        self.dt = dt
        self.dy = dy
        self.sec = sec
        self.pdate = pdate

        # discretize time to maturity
        self.ttm = (sec.m_date - pdate).days / 365.25
        self.M = int(self.ttm / dt)  # round down to nearest integer
        self.T = self.M * dt
        self.rf_T = self.ttm - self.T

        # discretize spatial mesh in log-price space
        self.y0 = np.log(sec.eq_spot)
        y_half_range = self.y_grid_coverage * sec.sigma(self.T) * np.sqrt(self.T)
        N_half = int(y_half_range / dy)
        self.N = int(2 * N_half)

        self.t_knots = np.arange(0, self.M + 1, 1) * self.dt
        self.y_knots = self.y0 + np.arange(-N_half, N_half + 0.5, 1) * self.dy
        self.S_knots = np.exp(self.y_knots)
        self.v_knots = np.array([])
        self.S0_index = N_half

        self._init_lambd()
        self._init_sigma_sq()
        self._init_K()
        self._init_L()
        self._init_R()  # recovery value of the bond upon default

        self._init_u()
        self._init_d()
        self._init_l()

        if (self.sec.callable):
            self.callable_grid = self._setup_callable_grid()

        if (self.sec.puttable):
            self.puttable_grid = self._setup_puttable_grid()

        if (self.sec.softcall):
            self.softcall_barrier = self.sec.soft_call_approx_1of1_barrier(pdate)
            self.softcall_barrier_index = np.argmax(self.S_knots >= self.softcall_barrier)
            self.softcall_end_tau = (self.sec.m_date - self.sec.softcallinfo['end']).days / 365.25
            self.softcall_start_tau = (self.sec.m_date - self.sec.softcallinfo['start']).days / 365.25

    def _setup_strike_grid(self, schedule):
        if schedule is None:
            return np.ones(self.M + 1) * self.sec.notional

        dates = schedule['date'].values
        prices = schedule['price'].values * (self.sec.notional / 100.)  # data was per 100 notional
        call_taus = [(self.sec.m_date - d).days / 365.25 for d in dates]
        call_grid = np.zeros(self.M + 1)
        for m in range(0, self.M + 1):
            for call_t, call_price in zip(call_taus, prices):
                if (self.t_knots[m] < call_t and call_t <= self.t_knots[m + 1]):
                    call_grid[m + 1] = call_price

        return call_grid

    def _setup_callable_grid(self):
        return self._setup_strike_grid(self.sec.call_schedule)

    def _setup_puttable_grid(self):
        return self._setup_strike_grid(self.sec.put_schedule)

    def reset_p(self, newp):
        self.sec.reset_p(newp)
        self.__init__(self.sec, self.pdate, self.dt, self.dy)

    def refresh_security(self, sec: ConvertibleBond):
        self.sec = sec
        self.__init__(self.sec, self.pdate, self.dt, self.dy)

    def coupon_stream(self):
        cp_dt = 1.0 / float(self.sec.c_freq)
        cp = self.sec.c_rate * cp_dt * self.sec.notional
        cp_stream = np.zeros(self.M + 1)
        # enumerate in backward time (to be compatible with solver)
        step = int(round(cp_dt / self.dt, 0))
        for m in np.arange(0, self.M + 1, step):
            cp_stream[m] = cp
        return cp_stream

    def accrued_interests(self):
        coupons = self.coupon_stream()
        accrued = np.zeros(self.M + 1)
        c_index = np.argwhere(coupons > 0).flatten()
        c_start = c_index
        c_end = np.append(c_index[1:] - 1, len(coupons) - 2)
        for start, end in zip(c_start, c_end):
            c = coupons[start]
            slots = end - start + 1.
            accrued[start:end + 1] = (c / slots) * np.arange(slots, 0, -1)
        accrued = accrued - coupons
        return accrued

    def initial_condition(self):
        eq = self.sec.cv_ratio * self.S_knots
        bond = self.sec.notional
        return np.maximum(eq, bond) + self.coupon_stream()[0]

    def bc_lower(self):
        recovery = np.array([self.sec.rr * self.sec.notional] * (self.M + 1))
        return recovery + self.accrued_interests()

    def bc_upper(self):
        eq = np.array([self.sec.cv_ratio * self.S_knots[-1]] * (self.M + 1))
        return eq + self.accrued_interests()

    def _init_lambd(self):
        lambd = np.zeros((self.M + 1, self.N + 1))
        for m, tau in np.ndenumerate(self.t_knots):
            t = self.ttm - tau  # tau is backward time here
            lambd[m] = self.sec.lambd(t, self.S_knots)
        self.lambd = lambd

    def _init_sigma_sq(self):
        self.sigma_sq = np.array([self.sec.sigma(self.ttm - tau) ** 2 for tau in self.t_knots])

    def _init_K(self):
        rt = np.array([self.sec.r(self.ttm - tau) for tau in self.t_knots])
        qt = np.array([self.sec.q(self.ttm - tau) for tau in self.t_knots])
        self.K = self.sec.eta * self.lambd + rt[:, np.newaxis] - qt[:, np.newaxis] - 0.5 * self.sigma_sq[:,
                                                                                           np.newaxis]  # -----------Warning----------------------

    def _init_L(self):
        rt = np.array([self.sec.r(self.ttm - tau) for tau in self.t_knots])
        self.L = -self.lambd - rt[:, np.newaxis]

    def _init_R(self):
        R = np.ones((self.M + 1, self.N + 1))
        Bond_RR = self.sec.rr * self.sec.notional * R
        EQ_conv = (1 - self.sec.eta) * self.sec.cv_ratio * self.S_knots
        EQ_RR = np.repeat(np.array([EQ_conv]), self.M + 1, axis=0)
        self.R = np.maximum(Bond_RR, EQ_RR)

    def _init_u(self):
        u = self.K + self.sigma_sq[:, np.newaxis] / self.dy
        self.u = (0.5 * self.dt / self.dy) * u

    def _init_d(self):
        d = -self.L + (self.sigma_sq[:, np.newaxis] / (self.dy ** 2))
        self.d = self.dt * d

    def _init_l(self):
        l = self.K - self.sigma_sq[:, np.newaxis] / self.dy
        self.l = (0.5 * self.dt / self.dy) * l

    def solve(self):
        M = self.M
        N = self.N
        theta = self.theta

        coupons = self.coupon_stream()
        accrued = self.accrued_interests()
        eq_conv = self.sec.cv_ratio * self.S_knots

        bc_lower = self.bc_lower()
        bc_upper = self.bc_upper()
        v = self.initial_condition()  # (N+1)-array
        v_dt = self.initial_condition()  # (N+1)-array

        for m in range(0, M):
            z = (1 - theta) * (self.u[m, 1:-1] * v[2:])
            z += (1 - (1 - theta) * self.d[m, 1:-1]) * v[1:-1]
            z -= (1 - theta) * self.l[m, 1:-1] * v[0:-2]
            z += self.dt * theta * (self.lambd[m + 1, 1:-1] * self.R[m + 1, 1:-1])
            z += self.dt * (1 - theta) * (self.lambd[m, 1:-1] * self.R[m, 1:-1])

            v[0] = bc_lower[m + 1]
            # v[N] = bc_upper[m+1]
            v[N] = 2 * v[N - 1] - v[N - 2]

            b = np.zeros(N - 1)
            b[0] = -theta * self.l[m + 1, 1] * v[0]
            b[-1] = theta * self.u[m + 1, N - 1] * v[N]

            C_l_diag = theta * self.l[m + 1, 2:-1]
            C_m_diag = 1 + theta * self.d[m + 1, 1:-1]
            C_u_diag = -theta * self.u[m + 1, 1:-2]

            # solve tridiagonal-matrix problem
            v[1:-1] = TDMA(C_l_diag, C_m_diag, C_u_diag, z + b)

            # 20-of-30 soft-call by issuer
            if (self.sec.softcall):
                tau = (m + 1) * self.dt
                if (tau >= self.softcall_end_tau and tau <= self.softcall_start_tau):
                    barx = self.softcall_barrier_index
                    v[barx:-1] = np.minimum(v[barx:-1], self.sec.softcallinfo['redemption'] + accrued[m + 1])

            # callable option by issuer has the least priority (iterating backward here)
            if (self.sec.callable and self.callable_grid[m + 1] > 0):
                call_price = self.callable_grid[m + 1]
                v[1:-1] = np.minimum(v[1:-1], call_price + accrued[m + 1])

            # puttable option by borrower (priority over issuer's call)
            if (self.sec.puttable and self.puttable_grid[m + 1] > 0):
                put_price = self.puttable_grid[m + 1]
                v[1:-1] = np.maximum(v[1:-1], put_price + accrued[m + 1])

                # borrower has option to convert any time (priority over issuer's call)
            v[1:-1] = np.maximum(v[1:-1], eq_conv[1:-1] + accrued[m + 1])

            # add discrete coupons - coupons are always paid before any conversion/call/put
            v[1:-1] = v[1:-1] + coupons[m + 1]

        # discount by the risk-free rate for the residual period smaller than dt
        self.v_knots = v * self.sec.Z(self.rf_T)

    def price(self):
        if (len(self.v_knots) == 0):
            self.solve()
        return self.v_knots[self.S0_index]

    def dirty_price(self):
        p = self.price()
        return p * (100. / self.sec.notional)

    def clean_price(self):
        dt = 1. / float(self.sec.c_freq)
        cp = self.sec.c_rate * dt * 100  # coupon per 100 notional
        remain = int(self.ttm / dt)
        accrued_t = dt - (self.ttm - remain * dt)
        accrued_c = cp * (accrued_t / dt)
        return self.dirty_price() - accrued_c

    def eq_spot_delta(self):
        self.solve()
        dvdy = (self.v_knots[self.S0_index + 1] - self.v_knots[self.S0_index - 1]) / (2 * self.dy)
        dvds = dvdy / self.sec.eq_spot
        delta = dvds / self.sec.cv_ratio  # same convention as  Bloomberg and BlackRock
        return delta

    def eq_spot_gamma(self):
        base_spot = self.sec.eq_spot
        abs_shock = min(10.0 / self.sec.cv_ratio, 0.01 * base_spot)

        self.sec.reset_eq_spot(base_spot + abs_shock)
        self.__init__(self.sec, self.pdate, dt=self.dt, dy=self.dy)
        up_delta = self.eq_spot_delta()

        self.sec.reset_eq_spot(base_spot - abs_shock)
        self.__init__(self.sec, self.pdate, dt=self.dt, dy=self.dy)
        dn_delta = self.eq_spot_delta()

        # revert to the base state
        self.sec.reset_eq_spot(base_spot)
        self.__init__(self.sec, self.pdate, dt=self.dt, dy=self.dy)
        return (up_delta - dn_delta) / (2 * abs_shock / base_spot)

    def eq_vega(self, shock=0.01):
        base_vol = self.sec.eq_vol

        self.sec.eq_vol = base_vol + shock
        self.__init__(self.sec, self.pdate, dt=self.dt, dy=self.dy)
        shocked_price = self.dirty_price()

        self.sec.eq_vol = base_vol
        self.__init__(self.sec, self.pdate, dt=self.dt, dy=self.dy)
        base_price = self.dirty_price()

        vega = (shocked_price - base_price)
        return vega
