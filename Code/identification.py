# identification.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.linalg import solve_triangular  # you were already using this
from scipy.optimize import minimize

from .irf import impulse_response
from .fevd import forecast_error_variance_decomposition


@dataclass
class StructuralResults:
    B: np.ndarray
    method: str
    n_rejected: Optional[int] = None
    m_accepted: Optional[int] = None
    corr: Optional[Any] = None
    extra: Dict[str, Any] = None


# ---------- Short-run zero ----------

def identify_short_run(sigma_u: np.ndarray, matrix: np.ndarray = None) -> StructuralResults:
    n = sigma_u.shape[0]
    
    # Default to Cholesky (lower triangular) if no mask is provided
    if matrix is None:
        B = np.linalg.cholesky(sigma_u)
        method = "short_cholesky"
    else:
        # 1 means free parameter, 0 means restricted to zero
        matrix = np.asarray(matrix)
        
        # Check for order condition (maximum of n*(n+1)/2 free parameters)
        if np.sum(matrix) > n * (n + 1) // 2:
            raise ValueError("The system is underidentified. Too few zero restrictions.")

        # Objective function: Minimize the distance between B*B.T and Sigma_u
        def objective(free_params):
            # Reconstruct B using the mask
            B_curr = np.zeros((n, n))
            B_curr[matrix == 1] = free_params
            
            # Loss: Frobenius norm of the difference
            diff = sigma_u - B_curr @ B_curr.T
            return np.sum(diff**2)

        # Initial guess: start from Cholesky but masked to fit the restriction structure
        B_start = np.linalg.cholesky(sigma_u)
        # Force initial guess to respect the mask
        B_start[matrix == 0] = 0
        x0 = B_start[matrix == 1]

        # Optimize
        res = minimize(objective, x0, method='BFGS', options={'gtol': 1e-8})
        
        if not res.success and res.fun > 1e-5:
            raise ValueError(f"Optimization failed to find a valid B matrix. Loss: {res.fun}")

        # Reconstruct final B
        B = np.zeros((n, n))
        B[matrix == 1] = res.x
        method = "short_custom_restrictions"

    return StructuralResults(
        B=B,
        method=method,
        n_rejected=0,
        m_accepted=None,
        corr=0,
        extra={},
    )


# ---------- Long-run zero ----------

def identify_long_run(
    sigma_u: np.ndarray,
    F: np.ndarray,
    lags: int,
    matrix: np.ndarray = None,
) -> StructuralResults:
    """
    Long-run identification.

    By default (matrix=None) imposes a lower-triangular long-run
    impact matrix C = M1 @ B via the Cholesky decomposition of
    M1 @ sigma_u @ M1.T, exactly as in the original Blanchard-Quah scheme.

    If matrix is provided (n x n array, 1 = free parameter,
    0 = restricted to zero), C is instead solved numerically so that
    C @ C.T matches M1 @ sigma_u @ M1.T subject to the given zero-restriction
    pattern (any pattern with at least n*(n+1)/2 free parameters works, not
    just the lower-triangular one).
    """
    n_vars = sigma_u.shape[0]

    # Long-run cumulative impact matrix M1 = (I - sum_s Phi_s)^{-1}
    M1 = np.eye(n_vars)
    for k in range(lags):
        M1 = M1 - F[0:n_vars, k * n_vars : (k + 1) * n_vars]
    M1 = np.linalg.inv(M1)

    # Sigma_C is the variance-covariance matrix implied for C = M1 @ B,
    # since Sigma_u = B B.T  =>  M1 @ Sigma_u @ M1.T = (M1 @ B)(M1 @ B).T
    sigma_C = M1 @ sigma_u @ M1.T

    if matrix is None:
        # Default: C is lower triangular (standard long-run zero restrictions)
        C = np.linalg.cholesky(sigma_C)
        method = "long_cholesky"
    else:
        # 1 means free parameter, 0 means restricted to zero (on C, not B)
        matrix = np.asarray(matrix)

        # Order condition (maximum of n*(n+1)/2 free parameters)
        if np.sum(matrix) > n_vars * (n_vars + 1) // 2:
            raise ValueError("The system is underidentified. Too few zero restrictions.")

        # Objective function: minimize the distance between C*C.T and Sigma_C
        def objective(free_params):
            C_curr = np.zeros((n_vars, n_vars))
            C_curr[matrix == 1] = free_params

            # Loss: Frobenius norm of the difference
            diff = sigma_C - C_curr @ C_curr.T
            return np.sum(diff**2)

        # Initial guess: start from Cholesky but masked to fit the restriction structure
        C_start = np.linalg.cholesky(sigma_C)
        C_start[matrix == 0] = 0
        x0 = C_start[matrix == 1]

        # Optimize
        res = minimize(objective, x0, method='BFGS', options={'gtol': 1e-8})

        if not res.success and res.fun > 1e-5:
            raise ValueError(f"Optimization failed to find a valid C matrix. Loss: {res.fun}")

        # Reconstruct final C
        C = np.zeros((n_vars, n_vars))
        C[matrix == 1] = res.x
        method = "long_custom_restrictions"

    # Recover the impact matrix B = M1^{-1} C
    B = np.linalg.solve(M1, C)

    return StructuralResults(
        B=B,
        method=method,
        n_rejected=0,
        m_accepted=None,
        corr=0,
        extra={},
    )

# ---------- Sign restrictions ----------

def identify_signs(
    sigma_u: np.ndarray,
    F: np.ndarray,
    matrix_signs: np.ndarray,
    steps_signs: int,
    steps: int,
    reps: int,
    alpha: float,
) -> StructuralResults:
    """
    Sign-restriction identification, mirroring your original 'S()' logic.

    Returns B as the median of the accepted set, plus IR and FEVD bands in 'extra'.
    """
    n_vars = sigma_u.shape[0]
    A = np.linalg.cholesky(sigma_u)

    n_rejected = 0
    m_accepted = 0
    MB: List[np.ndarray] = []
    IRL: Dict[int, List[np.ndarray]] = {x: [] for x in range(1, n_vars + 1)}
    FEVDL: Dict[int, List[np.ndarray]] = {x: [] for x in range(1, n_vars + 1)}

    print("Reps:\n")

    while True:
        continuacion = 0

        # Draw orthonormal Q
        D = np.random.randn(n_vars, n_vars)
        Q, R = np.linalg.qr(D)
        Q *= np.sign(np.diag(R))

        Bp = A @ Q

        # Check contemporaneous sign restrictions on Bp
        for i in range(n_vars):
            for j in range(n_vars):
                s = 1 if Bp[i, j] > 0 else -1
                if matrix_signs[i][j] == 0 or s == matrix_signs[i][j]:
                    continue
                else:
                    # flip the column j and re-check
                    Bp[:, j] = -Bp[:, j]
                    for ii in range(n_vars):
                        for jj in range(n_vars):
                            s_inner = 1 if Bp[ii, jj] > 0 else -1
                            if matrix_signs[ii][jj] == 0:
                                continue
                            elif s_inner == matrix_signs[ii][jj]:
                                continue
                            else:
                                continuacion += 1

        if continuacion == 0:
            B = Bp
            MB.append(B)
            count = 0
            # Horizontal sign restrictions on IRFs
            for k1 in range(n_vars):
                ir_check = impulse_response(
                    B=B,
                    F=F,
                    shock=k1 + 1,
                    impact=1.0,
                    steps=steps
                )
                for k2 in range(n_vars):
                    for k3 in range(steps_signs):
                        ir_check_v = ir_check[k3, k2]
                        s = 1 if ir_check_v > 0 else -1
                        if s == matrix_signs[k2][k1] or matrix_signs[k2][k1] == 0:
                            continue
                        else:
                            count += 1
                if count == 0:
                    IRL[k1 + 1].append(ir_check)

                    # FEVD attributable to shock (k1 + 1) for every variable,
                    # over the full horizon. We slice out shock (k1+1)'s
                    # column right away so FEVDL[k1+1] holds (steps, n_vars)
                    # matrices exactly like IRL[k1+1] does for IRFs, which
                    # lets the stats below reuse the exact same stacking logic.
                    vd_full = forecast_error_variance_decomposition(B, F, steps=steps)
                    vd_check = np.zeros((steps, n_vars))
                    for vi in range(n_vars):
                        vd_check[:, vi] = vd_full[vi + 1][:, k1]
                    FEVDL[k1 + 1].append(vd_check)

                    m_accepted += 1
                    if m_accepted in list(range(100, reps + 1, 100)):
                        print(f"{m_accepted}/{reps}")
        else:
            n_rejected += 1

        if m_accepted >= reps:
            break

        if n_rejected >= 100000:
            print(
                "==============================\n"
                "Number of total draws exceeded\n"
                "=============================="
            )
            B_zero = np.zeros((n_vars, n_vars))
            return StructuralResults(
                B=B_zero,
                method="signs",
                n_rejected=n_rejected,
                m_accepted=m_accepted,
                corr=0,
                extra={"IRL": IRL, "FEVDL": FEVDL, "MB": MB},
            )

    # Build IR bands across accepted draws
    ir_median = {}
    ir_mean = {}
    ir_high = {}
    ir_low = {}
    ir_00 = {}
    ir_99 = {}
    ir_02 = {}
    ir_97 = {}
    ir_05 = {}
    ir_95 = {}
    ir_10 = {}
    ir_90 = {}
    ir_16 = {}
    ir_84 = {}
    ir_25 = {}
    ir_75 = {}

    alpha_low = alpha / 2.0
    alpha_high = 100.0 - alpha_low

    for key, lst in IRL.items():
        stack = np.stack(lst, axis=0)  # (n_accept, steps, n_vars)
        ir_median[key] = np.median(stack, axis=0)
        ir_mean[key] = np.mean(stack, axis=0)
        ir_high[key] = np.percentile(stack, alpha_low, axis=0)
        ir_low[key] = np.percentile(stack, alpha_high, axis=0)
        ir_00[key] = np.percentile(stack, 0.5, axis=0)
        ir_99[key] = np.percentile(stack, 99.5, axis=0)
        ir_02[key] = np.percentile(stack, 2.5, axis=0)
        ir_97[key] = np.percentile(stack, 97.5, axis=0)
        ir_05[key] = np.percentile(stack, 5.0, axis=0)
        ir_95[key] = np.percentile(stack, 95.0, axis=0)
        ir_10[key] = np.percentile(stack, 10.0, axis=0)
        ir_90[key] = np.percentile(stack, 90.0, axis=0)
        ir_16[key] = np.percentile(stack, 16.0, axis=0)
        ir_84[key] = np.percentile(stack, 84.0, axis=0)
        ir_25[key] = np.percentile(stack, 25.0, axis=0)
        ir_75[key] = np.percentile(stack, 75.0, axis=0)

    # Build FEVD bands across accepted draws (same convention as IR bands
    # above: 'high' uses the lower percentile and 'low' the upper one,
    # since that is the swap SVAR.ImpulseResponse() already compensates
    # for on read -- SVAR.VarianceDecomp() will do the same for FEVD).
    vd_median = {}
    vd_mean = {}
    vd_high = {}
    vd_low = {}
    vd_00 = {}
    vd_99 = {}
    vd_02 = {}
    vd_97 = {}
    vd_05 = {}
    vd_95 = {}

    for key, lst in FEVDL.items():
        if len(lst) == 0:
            continue
        stack = np.stack(lst, axis=0)  # (n_accept, steps, n_vars)
        vd_median[key] = np.median(stack, axis=0)
        vd_mean[key] = np.mean(stack, axis=0)
        vd_high[key] = np.percentile(stack, alpha_low, axis=0)
        vd_low[key] = np.percentile(stack, alpha_high, axis=0)
        vd_00[key] = np.percentile(stack, 0.5, axis=0)
        vd_99[key] = np.percentile(stack, 99.5, axis=0)
        vd_02[key] = np.percentile(stack, 2.5, axis=0)
        vd_97[key] = np.percentile(stack, 97.5, axis=0)
        vd_05[key] = np.percentile(stack, 5.0, axis=0)
        vd_95[key] = np.percentile(stack, 95.0, axis=0)

    B_median = np.median(np.stack(MB, axis=0), axis=0)
    B_mean = np.mean(np.stack(MB, axis=0), axis=0)
    
    extra = {
        "IRL": IRL,
        "MB": MB,
        "ir_median": ir_median,
        "ir_mean": ir_mean,
        "ir_high": ir_high,
        "ir_low": ir_low,
        "ir_00": ir_00,
        "ir_99": ir_99,
        "ir_02": ir_02,
        "ir_97": ir_97,
        "ir_05": ir_05,
        "ir_95": ir_95,
        "ir_10": ir_10,
        "ir_90": ir_90,
        "ir_16": ir_16,
        "ir_84": ir_84,
        "ir_25": ir_25,
        "ir_75": ir_75,
        "FEVDL": FEVDL,
        "vd_median": vd_median,
        "vd_mean": vd_mean,
        "vd_high": vd_high,
        "vd_low": vd_low,
        "vd_00": vd_00,
        "vd_99": vd_99,
        "vd_02": vd_02,
        "vd_97": vd_97,
        "vd_05": vd_05,
        "vd_95": vd_95,
        "alpha": alpha,
        "steps_signs": steps_signs,
    }

    return StructuralResults(
        B=B_mean,
        method="signs",
        n_rejected=n_rejected,
        m_accepted=m_accepted,
        corr=0,
        extra=extra,
    )

# ---------- IV (your IV identification) ----------

def identify_iv(
    sigma_u: np.ndarray,
    resid: np.ndarray,
    iv: np.ndarray,
    data_array: np.ndarray,
) -> StructuralResults:
    """
    IV identification as in your 'S()' method for method == 'IV'.

    Parameters
    ----------
    sigma_u : (n_vars, n_vars)
    resid : (T, n_vars)
        VAR residuals.
    iv : (T_iv, n_iv)
        Instrument(s); can contain 123456789 as missing sentinel.
    data_array : (T, n_vars)
        Original data used to align instruments and residuals.

    Returns
    -------
    StructuralResults
        B matrix and instrument diagnostics in 'extra'.
    """
    sigma_u = np.asarray(sigma_u, dtype=float)
    resid = np.asarray(resid, dtype=float)
    iv = np.asarray(iv, dtype=float)
    data_array = np.asarray(data_array, dtype=float)

    n_obs_u, n_vars = resid.shape
    n_obsi, n_iv = iv.shape

    corr_list: List[float] = []
    z_list: List[np.ndarray] = []
    varz_list: List[float] = []

    B = np.zeros((n_vars, n_vars))

    for j in range(n_iv):
        # replace sentinel 123456789 with NaN
        iv_col = iv[:, j].copy()
        iv_col[iv_col == 123456789] = np.nan

        # keep non-NaN instruments
        mask = ~np.isnan(iv_col)
        instrument = iv_col[mask]
        z_list.append(instrument)

        # correlate with the corresponding variable j in data_array (last len(instrument) obs)
        y_for_corr = data_array[-len(instrument) :, j]
        corr_j = abs(np.corrcoef(instrument, y_for_corr)[0, 1])
        corr_list.append(float(corr_j))
        varz_list.append(float(np.var(instrument)))

        # First stage
        XX_invfs = (np.var(instrument)) ** (-1)
        u_j = resid[-len(instrument) :, j]
        betafs = XX_invfs * (instrument.T @ u_j)
        ujhat = instrument * betafs

        # Initialize column j of B
        B[:, :] = 0.0
        B[j, j] = 1.0

        # Second stage: other entries in column j
        XX_invss = (np.var(ujhat)) ** (-1)
        for i in range(n_vars):
            if i != j:
                u_i = resid[-len(instrument) :, i]
                betass = XX_invss * (ujhat.T @ u_i)
                B[i, j] = betass

        C = np.linalg.cholesky(sigma_u)
        q = solve_triangular(C, B[:, j], lower=True)
        v = np.linalg.norm(q)
        B[:, j] = B[:, j] * v

    extra = {
        "z": z_list,
        "varz": varz_list,
        "corr": corr_list,
    }

    return StructuralResults(
        B=B,
        method="IV",
        n_rejected=0,
        m_accepted=None,
        corr=corr_list,
        extra=extra,
    )