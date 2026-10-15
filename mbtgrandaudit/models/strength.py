"""Attack/defence strength models fitted by time-weighted maximum likelihood.

All share the log-linear structure (Maher 1982):
    log E[home goals] = c_div + home_adv + att[home] + def[away]
    log E[away goals] = c_div +            att[away] + def[home]
with exponential time decay (Dixon & Coles 1997) and a ridge prior on team
parameters, which keeps teams with little data near a sensible default.
Second-tier results are included so promoted teams carry ratings with them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from .base import (MatchContext, Model, bivariate_poisson_matrix, dc_tau_matrix, nb_pmf,
                   poisson_matrix, time_weights)

NEWCOMER_PRIOR = (-0.2, 0.2)  # (att, def) for teams first seen this season with no lower-tier data


class StrengthModel(Model):
    name = "poisson"
    description = "Independent Poisson attack/defence model (Maher), time-weighted MLE"
    dist = "poisson"
    use_xg = False

    def __init__(self, xi: float = 0.0019, l2: float = 2.0):
        self.xi, self.l2 = xi, l2
        self.fitted = False

    # ----------------------------------------------------------------- data
    def _targets(self, h: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return h["hg"].to_numpy(float), h["ag"].to_numpy(float)

    # ------------------------------------------------------------------ fit
    def fit(self, hist: pd.DataFrame, as_of: pd.Timestamp, target_div: str) -> "StrengthModel":
        h = hist[(hist["date"] < as_of)].dropna(subset=["hg", "ag"]).reset_index(drop=True)
        if len(h) < 50:
            self.fitted = False
            return self
        self.teams = sorted(set(h["home"]) | set(h["away"]))
        self.tidx = {t: i for i, t in enumerate(self.teams)}
        self.divs = sorted(h["div"].unique())
        self.didx = {d: i for i, d in enumerate(self.divs)}
        self.target_div = target_div if target_div in self.didx else self.divs[0]

        T, D = len(self.teams), len(self.divs)
        hi = h["home"].map(self.tidx).to_numpy()
        ai = h["away"].map(self.tidx).to_numpy()
        di = h["div"].map(self.didx).to_numpy()
        w = time_weights(h["date"], as_of, self.xi)
        yh, ya = self._targets(h)

        # priors: newcomers without lower-tier history start below average
        cur_season = h["season"].max()
        first_season = h.groupby("home")["season"].min().combine(
            h.groupby("away")["season"].min(), min, fill_value=cur_season)
        prior_att, prior_def = np.zeros(T), np.zeros(T)
        if h["season"].nunique() > 1:
            for t, s in first_season.items():
                if s == cur_season:
                    prior_att[self.tidx[t]], prior_def[self.tidx[t]] = NEWCOMER_PRIOR
        self.prior_att, self.prior_def = prior_att, prior_def

        avg = np.log(max(np.average((yh + ya) / 2, weights=w), 0.1))
        x0 = np.concatenate([np.full(D, avg), [0.25], prior_att, prior_def, self._extra_x0()])
        self._data = (hi, ai, di, w, yh, ya, T, D)
        self.r = 12.0
        bounds = [(None, None)] * (D + 1 + 2 * T) + self._extra_bounds()
        for _ in range(3 if self.dist == "nb" else 1):
            res = minimize(self._objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 500})
            x0 = res.x
            if self.dist == "nb":
                self._fit_r(res.x)
        self._unpack(res.x)
        self.fitted = True
        self._post_fit(h)
        return self

    def _extra_x0(self):
        return []

    def _extra_bounds(self):
        return []

    def _post_fit(self, h: pd.DataFrame) -> None:
        pass

    def _linear(self, x):
        hi, ai, di, w, yh, ya, T, D = self._data
        c, home = x[:D], x[D]
        att, dfn = x[D + 1:D + 1 + T], x[D + 1 + T:D + 1 + 2 * T]
        eh = c[di] + home + att[hi] + dfn[ai]
        ea = c[di] + att[ai] + dfn[hi]
        return eh, ea, att, dfn

    def _loglik(self, y, eta, mu):
        """Return (log-likelihood per obs, d ll / d eta)."""
        if self.dist == "nb":
            r = self.r
            ll = gammaln(y + r) - gammaln(r) + r * np.log(r) - (r + y) * np.log(r + mu) + y * eta
            return ll, r * (y - mu) / (r + mu)
        return y * eta - mu, y - mu

    def _objective(self, x):
        hi, ai, di, w, yh, ya, T, D = self._data
        eh, ea, att, dfn = self._linear(x)
        mh, ma = np.exp(eh), np.exp(ea)
        llh, gh = self._loglik(yh, eh, mh)
        lla, ga = self._loglik(ya, ea, ma)
        ll = llh + lla
        extra_grad = []
        if self.dist == "dc":
            ll, gh, ga, extra_grad = self._dc_terms(x[-1], yh, ya, mh, ma, ll, gh, ga, w)
        wgh, wga = w * gh, w * ga
        grad = np.zeros_like(x)
        grad[:D] = np.bincount(di, wgh + wga, D)
        grad[D] = wgh.sum()
        grad[D + 1:D + 1 + T] = np.bincount(hi, wgh, T) + np.bincount(ai, wga, T)
        grad[D + 1 + T:D + 1 + 2 * T] = np.bincount(ai, wgh, T) + np.bincount(hi, wga, T)
        if extra_grad:
            grad[-1] = extra_grad[0]
        f = -(w * ll).sum()
        grad = -grad
        da, dd = att - self.prior_att, dfn - self.prior_def
        f += 0.5 * self.l2 * (da @ da + dd @ dd)
        grad[D + 1:D + 1 + T] += self.l2 * da
        grad[D + 1 + T:D + 1 + 2 * T] += self.l2 * dd
        return f, grad

    def _fit_r(self, x):
        hi, ai, di, w, yh, ya, T, D = self._data
        eh, ea, _, _ = self._linear(x)
        mh, ma = np.exp(eh), np.exp(ea)

        def nll(logr):
            self.r = np.exp(logr)
            return -(w * (self._loglik(yh, eh, mh)[0] + self._loglik(ya, ea, ma)[0])).sum()
        self.r = float(np.exp(minimize_scalar(nll, bounds=(0.5, 7.0), method="bounded").x))

    def _unpack(self, x):
        T, D = len(self.teams), len(self.divs)
        self.c = dict(zip(self.divs, x[:D]))
        self.home_adv = x[D]
        self.att = dict(zip(self.teams, x[D + 1:D + 1 + T]))
        self.dfn = dict(zip(self.teams, x[D + 1 + T:D + 1 + 2 * T]))
        self.extra = x[D + 1 + 2 * T:]

    # -------------------------------------------------------------- predict
    def _team(self, t):
        if t in self.att:
            return self.att[t], self.dfn[t]
        return NEWCOMER_PRIOR

    def expected_goals(self, ctx: MatchContext):
        if not self.fitted:
            return None
        ah, dh = self._team(ctx.home)
        aa, da = self._team(ctx.away)
        c = self.c[self.target_div]
        return float(np.exp(c + self.home_adv + ah + da)), float(np.exp(c + aa + dh))

    def predict(self, ctx: MatchContext, max_goals: int = 10):
        xg = self.expected_goals(ctx)
        if xg is None:
            return None
        return poisson_matrix(*xg, max_goals)

    def ratings(self) -> pd.DataFrame:
        return pd.DataFrame({"attack": self.att, "defence": self.dfn}).sort_values("attack", ascending=False)


class DixonColes(StrengthModel):
    name = "dixon_coles"
    description = "Dixon-Coles: Poisson with low-score dependence correction (rho), fitted jointly"
    dist = "dc"

    def _extra_x0(self):
        return [-0.05]

    def _extra_bounds(self):
        return [(-0.25, 0.25)]

    def _dc_terms(self, rho, yh, ya, lh, la, ll, gh, ga, w):
        tau = np.ones_like(lh)
        dtau_dlh = np.zeros_like(lh)   # d log tau / d log lambda_h
        dtau_dla = np.zeros_like(lh)
        dtau_drho = np.zeros_like(lh)
        m00 = (yh == 0) & (ya == 0)
        m01 = (yh == 0) & (ya == 1)
        m10 = (yh == 1) & (ya == 0)
        m11 = (yh == 1) & (ya == 1)
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m11] = 1 - rho
        tau = np.clip(tau, 1e-6, None)
        dtau_dlh[m00] = -lh[m00] * la[m00] * rho / tau[m00]
        dtau_dla[m00] = -lh[m00] * la[m00] * rho / tau[m00]
        dtau_dlh[m01] = lh[m01] * rho / tau[m01]
        dtau_dla[m10] = la[m10] * rho / tau[m10]
        dtau_drho[m00] = -lh[m00] * la[m00] / tau[m00]
        dtau_drho[m01] = lh[m01] / tau[m01]
        dtau_drho[m10] = la[m10] / tau[m10]
        dtau_drho[m11] = -1 / tau[m11]
        return ll + np.log(tau), gh + dtau_dlh, ga + dtau_dla, [(w * dtau_drho).sum()]

    @property
    def rho(self):
        return float(self.extra[0])

    def predict(self, ctx, max_goals=10):
        xg = self.expected_goals(ctx)
        if xg is None:
            return None
        return dc_tau_matrix(poisson_matrix(*xg, max_goals), *xg, self.rho)


class NegativeBinomial(StrengthModel):
    name = "negative_binomial"
    description = "Negative-binomial attack/defence model allowing over-dispersed scoring"
    dist = "nb"

    def predict(self, ctx, max_goals=10):
        xg = self.expected_goals(ctx)
        if xg is None:
            return None
        g = np.arange(max_goals + 1)
        return np.outer(nb_pmf(g, xg[0], self.r), nb_pmf(g, xg[1], self.r))


class XGPoisson(DixonColes):
    """Rates teams on chance quality rather than goals alone.

    Target per match = 0.65 * expected goals + 0.35 * actual goals, where expected
    goals is the published xG when available and otherwise a shots-based proxy
    (goals regressed on shots on / off target across the dataset).
    """
    name = "xg_poisson"
    description = "Dixon-Coles fitted on expected goals (xG, or shots-based xG proxy) blended with goals"

    def _targets(self, h):
        hg, ag = h["hg"].to_numpy(float), h["ag"].to_numpy(float)
        sot = np.concatenate([h["hst"], h["ast"]]).astype(float)
        off = np.concatenate([h["hs"], h["as"]]).astype(float) - sot
        goals = np.concatenate([hg, ag])
        ok = np.isfinite(sot) & np.isfinite(off) & (off >= 0)
        if ok.sum() > 100:
            X = np.column_stack([sot[ok], off[ok]])
            self.shot_coef = np.linalg.lstsq(X, goals[ok], rcond=None)[0]
        else:
            self.shot_coef = None
        proxy = np.full(len(goals), np.nan)
        if self.shot_coef is not None:
            proxy[ok] = np.column_stack([sot[ok], off[ok]]) @ self.shot_coef
        xg = np.concatenate([h["hxg"], h["axg"]]).astype(float)
        exp_goals = np.where(np.isfinite(xg), xg, proxy)
        y = np.where(np.isfinite(exp_goals), 0.65 * exp_goals + 0.35 * goals, goals)
        self.xg_coverage = float(np.isfinite(exp_goals).mean())
        n = len(hg)
        return y[:n], y[n:]

    def _dc_terms(self, rho, yh, ya, lh, la, ll, gh, ga, w):
        # rho is not identifiable on continuous targets; borrow a typical value.
        return ll, gh, ga, [0.0]

    @property
    def rho(self):
        return -0.06


class BivariatePoisson(StrengthModel):
    """Karlis & Ntzoufras (2003) bivariate Poisson.

    Marginal means come from the Poisson strength fit; the shared covariance
    component lambda3 = k * min(mu_home, mu_away) is then fitted by MLE so the
    model captures positive goal correlation (open vs. cagey games).
    """
    name = "bivariate_poisson"
    description = "Bivariate Poisson with fitted covariance term (correlated home/away goals)"

    def _post_fit(self, h):
        hi, ai, di, w, yh, ya, T, D = self._data
        x = np.concatenate([[self.c[d] for d in self.divs], [self.home_adv],
                            [self.att[t] for t in self.teams], [self.dfn[t] for t in self.teams]])
        eh, ea, _, _ = self._linear(x)
        mh, ma = np.exp(eh), np.exp(ea)
        yh_i, ya_i = yh.astype(int), ya.astype(int)
        mink = np.minimum(yh_i, ya_i)
        from scipy.special import gammaln as gl

        def nll(k):
            l3 = k * np.minimum(mh, ma)
            l1, l2 = mh - l3, ma - l3
            base = (-(l1 + l2 + l3) + yh_i * np.log(l1) + ya_i * np.log(l2)
                    - gl(yh_i + 1) - gl(ya_i + 1))
            s = np.zeros_like(mh)
            ratio = l3 / (l1 * l2)
            for j in range(int(mink.max()) + 1):
                term = np.exp(gl(yh_i + 1) - gl(yh_i - j + 1) - gl(j + 1)
                              + gl(ya_i + 1) - gl(ya_i - j + 1) + j * np.log(ratio))
                s += np.where(j <= mink, term, 0.0)
            return -(w * (base + np.log(s))).sum()

        self.k = float(minimize_scalar(nll, bounds=(1e-4, 0.6), method="bounded").x)

    def predict(self, ctx, max_goals=10):
        xg = self.expected_goals(ctx)
        if xg is None:
            return None
        l3 = self.k * min(xg)
        return bivariate_poisson_matrix(xg[0] - l3, xg[1] - l3, l3, max_goals)
