# irf.py
from __future__ import annotations

import numpy as np


def impulse_response(
    B: np.ndarray,
    F: np.ndarray,
    shock: int,
    impact: float = 1.0,
    steps: int = 10,
) -> np.ndarray:
    """
    Compute impulse responses for a given structural matrix B and companion matrix F.

    Returns an array of shape (steps, n_vars).
    """
    B = np.asarray(B, dtype=float)
    F = np.asarray(F, dtype=float)

    n_vars = B.shape[0]
    total_dim = F.shape[0]
    lags = total_dim // n_vars

    ir = np.zeros((steps, n_vars))
    impulse = np.zeros((n_vars, 1))
    impulse[shock - 1, 0] = impact

    # Accumulate F^h incrementally (F^h = F^(h-1) @ F) instead of calling
    # np.linalg.matrix_power(F, h) from scratch at every horizon.
    F_power = np.eye(n_vars * lags)  # F^0
    for h in range(steps):
        Psi_h = F_power[:n_vars, :n_vars]
        response = Psi_h @ B @ impulse
        ir[h, :] = response[:, 0]
        F_power = F_power @ F  # advance to F^(h+1) for the next iteration

    return ir


def compute_bands_from_draws(
    draws: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, dict[str, np.ndarray]]:
    """
    A partir de draws[var] (reps x T), calcula medias y cuantiles.

    Parameters
    ----------
    draws : dict
        {var_name: array (reps, T)}.
    alpha : float
        Nivel para el intervalo central (p.ej. 32 → central 68%).

    Returns
    -------
    stats : dict
        Diccionario con sub-dicts por tipo:
        {
          'point': {var: (T,)},
          'mean':  {var: (T,)},
          'low':   {var: (T,)},
          'high':  {var: (T,)},
          'q2_5':  {var: (T,)},
          'q97_5': {var: (T,)},
          'q5':    {var: (T,)},
          'q95':   {var: (T,)},
          'q0_5':    {var: (T,)},
          'q99_5':   {var: (T,)},
        }
    """
    lower_q = alpha / 2.0
    upper_q = 100.0 - lower_q

    stats = {
        "point": {},
        "mean": {},
        "low": {},
        "high": {},
        "q2_5": {},
        "q97_5": {},
        "q5": {},
        "q95": {},
        "q0_5": {},
        "q99_5": {},
    }

    for var, arr in draws.items():
        arr = np.asarray(arr, dtype=float)  # (reps, T)

        stats["point"][var] = np.mean(arr, axis=0)
        stats["mean"][var] = np.mean(arr, axis=0)
        stats["low"][var] = np.percentile(arr, lower_q, axis=0)
        stats["high"][var] = np.percentile(arr, upper_q, axis=0)
        stats["q2_5"][var] = np.percentile(arr, 2.5, axis=0)
        stats["q97_5"][var] = np.percentile(arr, 97.5, axis=0)
        stats["q5"][var] = np.percentile(arr, 5.0, axis=0)
        stats["q95"][var] = np.percentile(arr, 95.0, axis=0)
        stats["q0_5"][var] = np.percentile(arr, 0.5, axis=0)
        stats["q99_5"][var] = np.percentile(arr, 99.5, axis=0)
        

    return stats