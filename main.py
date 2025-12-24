
import pandas as pd
import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB, quicksum

from bist100_data_module import read_close_prices_bist100

## Read close price df
close_df = read_close_prices_bist100()
print(close_df.head()) 
print(close_df.shape)

##calculate log-returns
returns = np.log(close_df / close_df.shift(1)).dropna() ##drop first row
print(returns.shape)

DAYS_PER_WEEK = 5
train_weeks, test_weeks = 4, 1

train_len, test_len = train_weeks * DAYS_PER_WEEK, test_weeks  * DAYS_PER_WEEK ## 50 days, # 5 days

rolling_windows = []
T = len(returns)

for start in range(0, T - train_len - test_len + 1, test_len):
    train_slice = returns.iloc[start : start + train_len]
    test_slice  = returns.iloc[start + train_len : start + train_len + test_len]

    ## Sample mean and cov (TRAIN ONLY) -- mu and sigma
    mu_hat = train_slice.mean()          # (79,)
    Sigma_hat = train_slice.cov()        # (79,79)

    ## (Optional) realized test returns
    realized_test_mean = test_slice.mean()

    rolling_windows.append({
        "train_start": train_slice.index[0],
        "train_end":   train_slice.index[-1],
        "test_start":  test_slice.index[0],
        "test_end":    test_slice.index[-1],
        "sample_size": train_len,
        "mu_hat": mu_hat,
        "Sigma_hat": Sigma_hat,
        "realized_test_mean": realized_test_mean
    })

# print("rolling window 0")
# print(rolling_windows[0]["train_start"], rolling_windows[0]["train_end"])
# print(rolling_windows[0]["train_end"], rolling_windows[0]["test_end"])
# print(rolling_windows[0]["mu_hat"].shape, rolling_windows[0]["Sigma_hat"].shape)
# print()

def solve_markowitz_7(window, p=0.1):
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

    window = rolling_windows[0]
    assets = returns.columns.to_list()

    mu = window['mu_hat']
    Sigma = window['Sigma_hat']
    sample_size = window['sample_size']

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



def solve_markowitz_11(window, p=0.1, q=0.1):
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

    window = rolling_windows[0]
    assets = returns.columns.to_list()
    n = len(assets)

    # ----- data (align safely) -----
    mu = window["mu_hat"]          # ideally pd.Series indexed by assets
    Sigma = window["Sigma_hat"]    # ideally pd.DataFrame indexed/cols by assets
    E = window["sample_size"]      # number of samples used to compute mu_hat

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


sol7, val7 = solve_markowitz_7(rolling_windows[0])
sol11, val11 = solve_markowitz_11(rolling_windows[0], q=0.2)

# print(val7 == val11)
print(val7)
print(val11)
# ## Train-Test Split
# T = len(returns)
# train_size = int(0.8 * T)

# returns_train = returns.iloc[:train_size]
# returns_test  = returns.iloc[train_size:]

# print(returns_train.shape, returns_test.shape)


