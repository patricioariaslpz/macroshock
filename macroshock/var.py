# var.py
# Restricted-GLS estimation of the reduced-form VAR, following the
# Lütkepohl (2005, Ch. 5) affine-reparametrization approach: vec(beta) =
# r_vector + r_matrix @ gamma, solved for the free parameter vector gamma.
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

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

def estimate_var(Y: np.ndarray, X: np.ndarray, r_matrix: np.ndarray, r_vector: np.array) -> VARResults:
    # 1. Basic Setup
    Y = np.asarray(Y, dtype=float)  # (T, n_vars)
    X = np.asarray(X, dtype=float)  # (T, n_cols)
    r_matrix = np.asarray(r_matrix, dtype=float)
    r_vector = np.asarray(r_vector, dtype=float).flatten()
    
    n_obs, n_vars = Y.shape
    _, n_cols = X.shape

    # 2. Initial OLS to get Sigma_u
    # We need a consistent Sigma to weight the GLS
    # Algorithm: run one equation-by-equation OLS pass (unrestricted) purely
    # to get a first-pass estimate of the residual covariance Sigma_u, which
    # the GLS step below needs as a weighting matrix.
    XX = X.T @ X
    XX_inv = np.linalg.inv(XX)
    beta_ols = (XX_inv @ (X.T @ Y)).T  # (n_vars, n_cols)
    resid_ols = Y - X @ beta_ols.T
    sigma_u = (resid_ols.T @ resid_ols) / (n_obs - n_cols)
    sigma_inv = np.linalg.inv(sigma_u)

    # 3. Restricted GLS (The Lütkepohl Core), generalized to an AFFINE
    # reparametrization vec(beta) = r_vector + r_matrix @ gamma.
    #
    # Derivation: the unrestricted GLS normal equations are
    #   Lambda @ vec(beta) = rho,   Lambda = (X'X) kron Sigma_u^-1,
    #                                 rho   = vec(Sigma_u^-1 Y' X).
    # Substituting vec(beta) = r + R @ gamma and minimizing the equivalent
    # GLS quadratic form (vec(beta) - beta_unrestricted)' Lambda (...) over
    # gamma gives the FOC
    #   R' Lambda R @ gamma = R' (rho - Lambda @ r),
    # i.e. exactly the original lhs/rhs below, except the RHS now also
    # subtracts R' Lambda r. When r_vector = 0 this is byte-for-byte the
    # same computation as before.
    lhs_inner = np.kron(XX, sigma_inv)          # Lambda, (n_cols*n_vars, n_cols*n_vars)
    lhs = r_matrix.T @ lhs_inner @ r_matrix       # (n_free, n_free)

    # rho = vec(Sigma_u^-1 * Y' * X), same as before
    target_mat = sigma_inv @ Y.T @ X 
    target_vec = target_mat.flatten(order='F').reshape(-1, 1)  # rho

    # Lambda @ r_vector, via the same vec-kron identity used for target_vec
    # (vec(A X B) = (B' kron A) vec(X)): Lambda @ vec(r_mat) =
    # vec(Sigma_u^-1 @ r_mat @ XX). Skipped (stays exactly zero) when
    # r_vector is all zeros, which is the 'zeros'/'custom zeros'/None case.
    if np.any(r_vector != 0):
        r_mat = r_vector.reshape((n_vars, n_cols), order='F')
        offset_vec = (sigma_inv @ r_mat @ XX).flatten(order='F').reshape(-1, 1)
    else:
        offset_vec = np.zeros_like(target_vec)

    rhs = r_matrix.T @ (target_vec - offset_vec)

    # 4. Solving for Gamma and Beta
    # Use solve() for numerical stability to prevent "blowing up"
    # Algorithm: solve the (n_free x n_free) linear system for gamma, then
    # map back to the full-length vec(beta) via the affine reparametrization,
    # reshaping to (n_vars, n_cols) in column-major ('F') order.
    gamma = np.linalg.solve(lhs, rhs)
    beta_vec = r_vector.reshape(-1, 1) + r_matrix @ gamma
    
    # Reshape back to (n_vars, n_cols) using 'F' order
    beta = beta_vec.reshape((n_vars, n_cols), order='F')

    # 5. Update Results based on restricted Beta
    Y_hat = X @ beta.T
    resid = Y - Y_hat
    sigma_u = (resid.T @ resid) / (n_obs - n_cols)

    # 6. Standard Errors for Restricted Parameters
    # Cov(gamma) = [R' (XX ⊗ Σu^-1) R]^-1 -- unaffected by the constant
    # offset r_vector, since Cov(r + R@gamma) = R Cov(gamma) R' regardless
    # of r.
    cov_gamma = np.linalg.inv(lhs)
    cov_beta_vec = np.diagonal(r_matrix @ cov_gamma @ r_matrix.T)
    beta_std = np.sqrt(cov_beta_vec).reshape((n_vars, n_cols), order='F')

    # 7. T-stats and P-values
    # Algorithm: elementwise t = coefficient / standard error, then a
    # two-sided p-value from the Student-t CDF with n_obs - n_cols degrees
    # of freedom (see stats.t_pvalue), looped over every (equation, column)
    # pair.
    tstat = beta / beta_std
    pvalue = np.zeros((n_vars, n_cols))
    for i in range(n_vars):
        for j in range(n_cols):
            pvalue[i, j] = t_pvalue(tstat[i, j], n_obs - n_cols)

    # 8. Information Criteria
    # Algorithm: Gaussian log-likelihood evaluated at the restricted
    # estimates, then the usual AIC/BIC/HQIC penalties using n_params =
    # number of FREE parameters (columns of r_matrix) -- since r_vector is
    # a fixed, known offset rather than something estimated, it contributes
    # 0 to the parameter count. FPE follows Lütkepohl (2005, eq. 4.3.9).
    ll = (-(n_obs * n_vars * np.log(2.0 * np.pi)) / 2.0 
          - (n_obs / 2.0) * np.log(np.linalg.det(sigma_u)) 
          - (n_obs * n_vars / 2.0))
    
    # Number of free parameters is the number of columns in r_matrix
    # (r_vector is fixed/known, so it contributes 0 estimated parameters)
    n_params = r_matrix.shape[1]
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