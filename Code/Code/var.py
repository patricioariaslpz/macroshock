# var.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# relative
# from .stats import t_pvalue

# use this instead:
from .stats import t_pvalue


@dataclass
class VARResults:
    beta: np.ndarray
    sigma_u: np.ndarray
    resid: np.ndarray
    Y_hat: np.ndarray
    ll: float
    aic: float
    bic: float
    hqic: float
    det: float
    fpe: float
    beta_std: np.ndarray
    tstat: np.ndarray
    pvalue: np.ndarray
    n_obs: int
    n_vars: int
    n_cols: int
    n_params: int

def estimate_var(Y: np.ndarray, X: np.ndarray, R: np.ndarray) -> VARResults:
    # 1. Basic Setup
    Y = np.asarray(Y, dtype=float)  # (T, n_vars)
    X = np.asarray(X, dtype=float)  # (T, n_cols)
    R = np.asarray(R, dtype=float)
    
    n_obs, n_vars = Y.shape
    _, n_cols = X.shape

    # 2. Initial OLS to get Sigma_u
    # We need a consistent Sigma to weight the GLS
    XX = X.T @ X
    XX_inv = np.linalg.inv(XX)
    beta_ols = (XX_inv @ (X.T @ Y)).T  # (n_vars, n_cols)
    resid_ols = Y - X @ beta_ols.T
    sigma_u = (resid_ols.T @ resid_ols) / (n_obs - n_cols)
    sigma_inv = np.linalg.inv(sigma_u)

    # 3. Restricted GLS (The Lütkepohl Core)
    # LHS: [R' (X'X ⊗ Σu^-1) R]
    # Note: Lütkepohl uses (Z*Z') which is (X.T @ X) in your notation
    lhs_inner = np.kron(XX, sigma_inv)
    lhs = R.T @ lhs_inner @ R

    # RHS: R' * vec(Σu^-1 * Y' * X) 
    # This matches the MATLAB 'R' * vec(inv(SIGMAu) * Y * Z')'
    # Note: In MATLAB Y is (q, T) and Z is (k, T). In Python Y is (T, q) and X is (T, k)
    target_mat = sigma_inv @ Y.T @ X 
    target_vec = target_mat.flatten(order='F').reshape(-1, 1)
    rhs = R.T @ target_vec

    # 4. Solving for Gamma and Beta
    # Use solve() for numerical stability to prevent "blowing up"
    gamma = np.linalg.solve(lhs, rhs)
    beta_vec = R @ gamma
    
    # Reshape back to (n_vars, n_cols) using 'F' order
    beta = beta_vec.reshape((n_vars, n_cols), order='F')

    # 5. Update Results based on restricted Beta
    Y_hat = X @ beta.T
    resid = Y - Y_hat
    sigma_u = (resid.T @ resid) / (n_obs - n_cols)

    # 6. Standard Errors for Restricted Parameters
    # Cov(gamma) = [R' (XX ⊗ Σu^-1) R]^-1
    cov_gamma = np.linalg.inv(lhs)
    cov_beta_vec = np.diagonal(R @ cov_gamma @ R.T)
    beta_std = np.sqrt(cov_beta_vec).reshape((n_vars, n_cols), order='F')

    # 7. T-stats and P-values
    tstat = beta / beta_std
    pvalue = np.zeros((n_vars, n_cols))
    for i in range(n_vars):
        for j in range(n_cols):
            pvalue[i, j] = t_pvalue(tstat[i, j], n_obs - n_cols)

    # 8. Information Criteria
    ll = (-(n_obs * n_vars * np.log(2.0 * np.pi)) / 2.0 
          - (n_obs / 2.0) * np.log(np.linalg.det(sigma_u)) 
          - (n_obs * n_vars / 2.0))
    
    # Number of free parameters is the number of columns in R
    n_params = R.shape[1]
    aic = 2.0 * n_params - 2.0 * ll
    bic = n_params * np.log(n_obs) - 2.0 * ll
    hqic = 2.0 * n_params * np.log(np.log(n_obs)) - 2.0 * ll
    fpe = np.linalg.det(sigma_u) * ((n_obs + n_cols) / (n_obs - n_cols)) ** n_vars
    
    return VARResults(
        beta=beta, sigma_u=sigma_u, resid=resid, Y_hat=Y_hat,
        ll=ll, aic=aic, bic=bic, hqic=hqic, det=np.linalg.det(sigma_u),
        fpe=fpe, beta_std=beta_std, tstat=tstat, pvalue=pvalue,
        n_obs=n_obs, n_vars=n_vars, n_cols=n_cols, n_params=n_params
    )