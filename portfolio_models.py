
import pandas as pd
import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB, quicksum

from asset_data_module import read_close_prices, read_close_prices_all_merged
from features import make_weekly_windows


def solve_markowitz_zymler7(assets, window, p=0.1):
    """Solves Zymler et al. (Eq. 11) robust mean-variance portfolio.   
    
    p : float in [0,1)
        Robustness/confidence parameter controlling the size of the covariance
        uncertainty set. Larger p => more conservative. Mapped to, delta = sqrt(p/(1-p)).

    delta scales the penalty term for covariance (risk) model uncertainty:
        -delta * || Sigma^{1/2} w ||_2.
    """
    ## delta
    delta = math.sqrt(p/(1-p))   ## covariance-uncertainty penalty term
    
    m = gp.Model("zymler_eq07")

    # window = rolling_windows[0]
    # assets = returns.columns.to_list()

    mu = window['past_returns'].mean()
    Sigma = window['past_returns'].cov()

    # align to assets order
    mu_vec = mu.loc[assets].values if hasattr(mu, "loc") else np.asarray(mu)
    Sigma_mat = Sigma.loc[assets, assets].values if hasattr(Sigma, "loc") else np.asarray(Sigma)

    n = len(assets)

    w = m.addVars(assets, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="w")

    # Decision variables: weights
    w = m.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w")

    # Auxiliary variable for the norm part
    t = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t")

    # Budget / simplex constraint (typical)
    m.addConstr(quicksum(w[i] for i in range(n)) == 1.0, name="budget")

    # Quadratic constraint: w^T Sigma w <= t^2
    quad = gp.QuadExpr()
    for i in range(n):
        for j in range(n):
            if Sigma_mat[i, j] != 0:
                quad += Sigma_mat[i, j] * w[i] * w[j]

    m.addQConstr(quad <= t * t, name="soc_like")

    # Objective: maximize mu^T w - delta * t
    m.setObjective(
        quicksum(mu_vec[i] * w[i] for i in range(n)) - delta * t,
        GRB.MAXIMIZE
    )

    m.optimize()

    w_opt = np.array([w[i].X for i in range(n)])
    print("obj =", m.ObjVal)
    print("sum w =", w_opt.sum())

    return w_opt, round(m.ObjVal, 4)



def solve_markowitz_zymler11(assets, window, p=0.1, q=0.1):
    """Solves Zymler et al. (Eq. 11) robust mean-variance portfolio.   
    
    ----------Parameters----------

    p : float in [0,1)
        Robustness/confidence parameter controlling the size of the covariance
        uncertainty set. Larger p => more conservative. Mapped to, delta = sqrt(p/(1-p)).

    q : float in [0,1)
        Robustness/confidence parameter controlling the size of the mean-return
        uncertainty set. Larger q => more conservative. Mapped to, kappa = sqrt(q/(1-q)).

    -----Notes-----

    kappa scales the penalty term for mean estimation risk:
        -kappa * || Lambda^{1/2} w ||_2,  with Lambda = Sigma / E.

    delta scales the penalty term for covariance (risk) model uncertainty:
        -delta * || Sigma^{1/2} w ||_2.
    """
    ## hyper-parameters
    kappa = math.sqrt(q/(1-q))   ## sqrt(q/(1-q)) if you use that mapping
    delta = math.sqrt(p/(1-p))   ## covariance-uncertainty penalty term

    m = gp.Model("zymler_eq11")

    # window = rolling_windows[0]
    # assets = returns.columns.to_list()
    n = len(assets)

    # ----- data (align safely) -----
    mu = window['past_returns'].mean()   # ideally pd.Series indexed by assets
    Sigma = window['past_returns'].cov() # ideally pd.DataFrame indexed/cols by assets
    E = window["past_returns"].shape[0]  # number of samples used to compute mu_hat
    # mu = window["mu_hat"]          # ideally pd.Series indexed by assets
    # Sigma = window["Sigma_hat"]    # ideally pd.DataFrame indexed/cols by assets
    # E = window["sample_size"]      # number of samples used to compute mu_hat
    

    ## align to assets order
    mu_vec = mu.loc[assets].values if hasattr(mu, "loc") else np.asarray(mu)
    Sigma_mat = Sigma.loc[assets, assets].values if hasattr(Sigma, "loc") else np.asarray(Sigma)

    ## Lambda = (1/E) Sigma   (from the paper)
    Lambda_mat = Sigma_mat / float(E)

    # (optional) tiny ridge for numerical stability / PSD issues
    # eps = 1e-10
    # Sigma_mat = Sigma_mat + eps * np.eye(n)
    # Lambda_mat = Lambda_mat + eps * np.eye(n)

    ## Decision variables: weights
    w = m.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w")

    ## Budget / simplex constraint (typical)
    m.addConstr(quicksum(w[i] for i in range(n)) == 1.0, name="budget")

    ## Auxiliary variable for the norm part
    t_mu = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t_mu")       # for ||Lambda^{1/2} w||
    t_sig = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t_sig")     # for ||Sigma^{1/2} w||

    ## Quadratic constraint: w^T Sigma w <= t_mu^2  & w^T Lambda w <= t_Sig^2
    quad_L = gp.QuadExpr()
    quad_S = gp.QuadExpr()

    for i in range(n):
        for j in range(n):
            aL = Lambda_mat[i, j]
            aS = Sigma_mat[i, j]
            if aL != 0:
                quad_L += aL * w[i] * w[j]
            if aS != 0:
                quad_S += aS * w[i] * w[j]

    ## enforce w'Lambda w <= t_mu^2 and w'Sigma w <= t_sig^2
    m.addQConstr(quad_L <= t_mu * t_mu, name="Lambda_norm")
    m.addQConstr(quad_S <= t_sig * t_sig, name="Sigma_norm")

    ## objective
    m.setObjective(
        quicksum(mu_vec[i] * w[i] for i in range(n)) - kappa * t_mu - delta * t_sig,
        GRB.MAXIMIZE
    )

    m.optimize()

    w_opt = np.array([w[i].X for i in range(n)])
    print("Obj:", m.ObjVal)
    print("sum(w):", w_opt.sum())

    return w_opt, round(m.ObjVal, 4)


def solve_markowitz_robust(assets, expected_returns, Sigma, Lambda, kappa, delta):
    """ 
    Parameters:
        -expected returns: array with size=len(assets)
        -Sigma: Covariances of returns - from past returns (NxN)
        -Lambda: Covariance of prediction errors (residuals) of returns (NxN)
        -kappa: Penalty parameter for return uncertainty
        -delta: Penalty parameter for risk - sqrt(p/(1-p))
    """
    rhat = np.asarray(expected_returns, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)

    n = len(assets)
    assert rhat.shape == (n,)
    assert Sigma.shape == (n, n)
    assert Lambda.shape == (n, n)

    m = gp.Model("robust_markowitz_photo")

    # -------------------------
    # Variables
    # -------------------------
    w = m.addVars(n, lb=0.0, ub=1.0, name="w")  # 0 <= w_i <= 1
    t_r = m.addVar(lb=0.0, name="t_r")          # >= 0
    t_s = m.addVar(lb=0.0, name="t_s")          # >= 0  (this is your t_Σ)

    # -------------------------
    # Constraints
    # -------------------------
    # sum_i w_i = 1
    m.addConstr(gp.quicksum(w[i] for i in range(n)) == 1.0, name="budget")

    # w^T Lambda w <= t_r^2
    m.addQConstr(
        gp.quicksum(Lambda[i, j] * w[i] * w[j] for i in range(n) for j in range(n))
        <= t_r * t_r,
        name="mean_uncertainty"
    )

    # w^T Sigma w <= t_s^2
    m.addQConstr(
        gp.quicksum(Sigma[i, j] * w[i] * w[j] for i in range(n) for j in range(n))
        <= t_s * t_s,
        name="risk"
    )

    # -------------------------
    # Objective: max rhat^T w - kappa t_r - delta t_s
    # -------------------------
    m.setObjective(
        gp.quicksum(rhat[i] * w[i] for i in range(n)) - kappa * t_r - delta * t_s,
        GRB.MAXIMIZE
    )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Optimization status: {m.Status}")

    w_sol = np.array([w[i].X for i in range(n)], dtype=float)
    # zero-out tiny weights
    w_sol[np.abs(w_sol) < 1e-5] = 0.0

    # (optional but recommended) renormalize if sum should be 1
    s = w_sol.sum()
    if s > 0:
        w_sol = w_sol / s
    obj_val = float(m.ObjVal)
    
    return w_sol, obj_val


# ## Test on data 
# tickers, close_df = read_close_prices_all_merged(['bist100', 'dow30', 'commodities', 'bonds', 'funds_mini'])
# print(close_df.shape)

# rolling_windows = make_weekly_windows(close_prices=close_df, lookback=5) ## lookback weeks!

# print(rolling_windows[0]['past_prices'].shape[0])

# test_window = rolling_windows[0]
# assets = list(close_df.columns)
# sol7, val7 = solve_markowitz_zymler7(assets, test_window)
# sol11, val11 = solve_markowitz_zymler11(assets, test_window, q=0.2)

# # print(val7 == val11)
# print(val7)
# print(val11)


