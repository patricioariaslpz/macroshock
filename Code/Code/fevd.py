# fevd.py
from __future__ import annotations

import numpy as np


def forecast_error_variance_decomposition(
    B: np.ndarray,
    F: np.ndarray,
    steps: int = 10,
) -> dict[int, np.ndarray]:
    """Compute Forecast Error Variance Decomposition (FEVD) for a given structural matrix B

    and companion matrix F.

    Parameters
    ----------
    B : np.ndarray
        Impact matrix of shape (n_vars, n_vars).
    F : np.ndarray
        Companion matrix of shape (n_vars * lags, n_vars * lags).
    steps : int, optional
        Forecast horizon steps (default is 10).

    Returns
    -------
    dict[int, np.ndarray]
        Dictionary keyed by 1-based variable index (1, 2, ..., n_vars).
        Each value is an array of shape (steps, n_vars) containing the share
        of variance contributed by each shock j to variable i over horizon h.
    """
    B = np.asarray(B, dtype=float)
    F = np.asarray(F, dtype=float)

    n_vars = B.shape[0]

    # 1. Pre-calculate Orthogonalized Impulse Responses for all horizons.
    # Accumulate F^h incrementally (F^h = F^(h-1) @ F) instead of calling
    # np.linalg.matrix_power(F, h) from scratch at every horizon.
    theta = np.zeros((steps, n_vars, n_vars))
    F_power = np.eye(F.shape[0])  # F^0
    for h in range(steps):
        Psi_h = F_power[:n_vars, :n_vars]
        theta[h] = Psi_h @ B
        F_power = F_power @ F  # advance to F^(h+1) for the next iteration

    # 2. Calculate Variance Decomposition per variable
    VD = {}
    for i in range(n_vars):
        vardec = np.zeros((steps, n_vars))
        for k in range(steps):
            # Squared responses of variable i to all shocks j up to step k
            squared_responses = theta[: k + 1, i, :] ** 2
            mse_ij = np.sum(squared_responses, axis=0)

            total_mse_i = np.sum(mse_ij)
            if total_mse_i > 0:
                vardec[k, :] = mse_ij / total_mse_i

        # Store using 1-based variable index
        VD[i + 1] = vardec

    return VD