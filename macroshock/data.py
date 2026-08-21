#data.py
# Builds the regressand Y, the regressor matrix X (deterministic terms +
# lags + exogenous + seasonal dummies), and the restriction pair
# (r_matrix, r_vector) that var.py's GLS estimator needs.

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.linalg import null_space

@dataclass
class YXR:
    Y: np.ndarray
    X: np.ndarray
    r_matrix: np.ndarray
    r_vector: np.array

def YX(model) -> YXR:
    
    model.n_obs, _ = model.data.shape

    # Algorithm: Y drops the first `lags` observations (no history to
    # regress on); X stacks the `lags` lagged copies of the data side by
    # side, most recent lag first (y_{t-1}, y_{t-2}, ..., y_{t-lags}).
    Y = model.data[model.lags:,:]
    X = []

    for i in range(model.lags):
        X.append(model.data[(model.lags-i-1) : model.n_obs - model.lags + (model.lags-i-1),:])
    X = np.hstack(X)
    
    # Prepend deterministic terms: constant, linear trend, squared trend,
    # depending on `const` (0 = none, 1 = const, 2 = const+trend, 3 =
    # const+trend+trend^2). Column order from here on is therefore
    # [deterministic terms, lags, exogenous (below), dummies (below)].
    if model.const == 1:
        X = np.hstack([np.ones((model.n_obs - model.lags, 1)), X])
    elif model.const == 2:
        trend = np.arange(1, model.n_obs - model.lags + 1).reshape(-1, 1)
        X = np.hstack([np.ones((model.n_obs - model.lags, 1)), trend, X])
    elif model.const == 3:
        trend = np.arange(1, model.n_obs - model.lags + 1).reshape(-1, 1)
        X = np.hstack([np.ones((model.n_obs - model.lags, 1)), trend, trend**2, X])
    
    model.n_obs, _ = X.shape
    
    # Append exogenous regressors, if any, right after the lag block.
    # Realigns to the last n_obs rows if the exogenous series has a
    # different (longer) history than the endogenous data.
    if model.data_exogenous is not None:
        if model.n_obsex==model.n_obs:
            X_exog=model.data_exogenous
        else:
            X_exog=model.data_exogenous[-model.n_obs:]
            if len(X_exog)!=model.n_obs:
                print('Exogenous variables in different domain')
        X=np.hstack([X,X_exog])
        
    # Append seasonal dummies, if requested. Builds the FULL dummy matrix
    # (covering the estimation sample AND the future forecast horizon, so
    # fc.py has dummy values ready for out-of-sample periods), then keeps
    # only n_dum = 3 (quarterly) or 11 (monthly) columns -- one category is
    # dropped in each case to avoid perfect collinearity with the constant.
    if model.DUM is not None:
        if model.DUM=='Q':
            model.dummies=['Q'+f"{i+1}" for i in range(4)]
            model.n_dum=3
            model.D=np.zeros((model.n_obs+model.horizon,4))
            for j in range(model.n_obs+model.horizon):
                for x in range(4):
                    if j in range(x,model.n_obs+model.horizon,4):
                        model.D[j,x]=1
        if model.DUM=='M':
            model.dummies=['M'+f"{i+1}" for i in range(12)]
            model.n_dum=11
            model.D=np.zeros((model.n_obs+model.horizon,12))
            for j in range(model.n_obs+model.horizon):
                for x in range(12):
                    if j in range(x,model.n_obs+model.horizon,12):
                        model.D[j,x]=1
        model.D=model.D[:,:model.n_dum]
        model.Dhorizon=model.D[model.n_obs:model.n_obs+model.horizon,:]
        X=np.hstack([X,model.D[0:model.n_obs,:]])
    else:
        model.n_dum=0

    model.Y=Y
    model.X=X
    
    if isinstance(model.Y, pd.Series):
        model.Y=model.Y.values.reshape(-1, 1)
    if isinstance(model.X, pd.Series):
        model.X=model.X.values.reshape(-1, 1)
        
    model.Y = model.Y.reshape(-1, 1) if model.Y.ndim == 1 else model.Y
    model.X = model.X.reshape(-1, 1) if model.X.ndim == 1 else model.X
    
    _ , model.n_vars=model.Y.shape
    model.n_obs, model.n_cols= model.X.shape
    
    model.X = np.array(model.X, dtype=float)
    model.Y = np.array(model.Y, dtype=float)
    
    n_beta = model.n_cols * model.n_vars

    # Build (r_matrix, r_vector) such that vec(beta) = r_vector + r_matrix @ gamma,
    # the affine reparametrization var.py's GLS estimator solves for gamma.
    # r_vector = 0 in every branch except 'standard'.
    if model.restrictions is None:
        # No restrictions: r_matrix = identity (gamma IS beta, nothing
        # reparametrized away), r_vector = 0 (no offset).
        model.r_matrix = np.eye(n_beta)
        model.r_vector = np.zeros(n_beta)

    elif model.restrictions == 'standard':
        # General affine restriction C @ vec(beta) = c, supplied by the
        # user as model.c_matrix (shape (n_constraints, n_cols*n_vars))
        # and model.c_vector (shape (n_constraints,)) BEFORE calling YX().
        #
        # Algorithm: convert C @ vec(beta) = c into vec(beta) = r_vector +
        # r_matrix @ gamma by taking
        #   - r_vector = minimum-norm particular solution of C @ r = c
        #     (via the Moore-Penrose pseudoinverse of C);
        #   - r_matrix = a basis for the null space of C (so C @ r_matrix = 0),
        #     spanning every direction gamma is free to move in without
        #     violating the constraint.
        # This is strictly more general than 'zeros'/'custom zeros' below
        # (which only ever restrict individual coefficients to exactly
        # zero): C can encode arbitrary linear combinations of coefficients,
        # and c need not be zero.
        
        C = np.asarray(model.c_matrix, dtype=float)
        c = np.asarray(model.c_vector, dtype=float).flatten()

        if C.shape[1] != n_beta:
            raise ValueError(
                f"restrictions='standard': c_matrix must have {n_beta} columns "
                f"(n_cols * n_vars = {model.n_cols} * {model.n_vars}), "
                f"got {C.shape[1]}."
            )
        if C.shape[0] != c.shape[0]:
            raise ValueError(
                f"restrictions='standard': c_matrix has {C.shape[0]} rows but "
                f"c_vector has {c.shape[0]} entries; they must match."
            )

        # C must have full row rank for the restrictions to be independent
        # (i.e. non-redundant); otherwise C @ C^+ wouldn't recover c exactly.
        rank_C = np.linalg.matrix_rank(C)
        if rank_C < C.shape[0]:
            raise ValueError(
                "restrictions='standard': c_matrix is row-rank deficient "
                f"(rank {rank_C} < {C.shape[0]} rows) -- the restrictions "
                "are redundant or contradictory."
            )

        r_particular = np.linalg.pinv(C) @ c
        if not np.allclose(C @ r_particular, c, atol=1e-8):
            print(
                "Warning: restrictions='standard' -- c_matrix @ [solution] does not "
                "exactly reproduce c_vector; the system may be inconsistent. "
                "Using the least-squares particular solution."
            )

        model.r_matrix = null_space(C)   # (n_beta, n_free); n_free = n_beta - rank(C)
        model.r_vector = r_particular    # (n_beta,)

    elif model.restrictions == 'zeros':
        # Zero out ALL lags of variable j in equation i, for every (i, j)
        # pair the user marked 0 in the n_vars x n_vars mask model.r_matrix.
        # Algorithm: start from the identity (one column per coefficient)
        # and DELETE the columns corresponding to restricted coefficients --
        # the remaining columns are exactly a null-space basis for "these
        # coefficients are 0", so r_vector stays 0 (pure reparametrization,
        # no offset needed).
        RR = np.eye(n_beta)
        drop = []
        
        for i in range(model.n_vars):
            for j in range(model.n_vars):
                if model.r_matrix[i][j]==0:
                    for k in range(model.lags):
                        drop.append(model.const*model.n_vars+i+(model.n_vars*j)+(k*model.n_vars*model.n_vars))
        
        model.r_matrix = np.delete(RR, drop, axis=1)
        model.r_vector = np.zeros(n_beta)
                    
    elif model.restrictions == 'zeros custom':
        # Same idea as 'zeros', but coefficient-by-coefficient instead of
        # lag-block-by-lag-block: model.r_matrix is expected to already be a
        # full (n_vars, n_cols) 0/1 mask (not just an n_vars x n_vars one),
        # letting the user zero out individual entries of beta directly.
        RR = model.r_matrix.ravel(order='F')
        RRR = np.eye(len(RR))
        drop = []
        
        for i in range(len(RR)):
            if RR[i]==0:
                drop.append(i)
                
        model.r_matrix = np.delete(RRR, drop, axis=1)
        model.r_vector = np.zeros(n_beta)
        
    return YXR(Y=model.Y, X=model.X, r_matrix=model.r_matrix, r_vector=model.r_vector)