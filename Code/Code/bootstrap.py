# bootstrap.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from .var import estimate_var
from .irf import impulse_response
from .fc import compute_forecast
from .fevd import forecast_error_variance_decomposition


def bootstrap_svar(model, typ: str, shock: int = 1, reps: int = 100, impact: float = 1.0, steps: int = 30, horizon: int = 5, past: int = 24):
    """
    Replica la lógica original de SVAR.Bootstrap, pero fuera de la clase.

    Parameters
    ----------
    model : objeto tipo SVAR
        Debe tener los mismos atributos que usaba el método Bootstrap original.
    typ : {'IR', 'FC'}
        Tipo de bootstrap: IRFs o Forecasts.
    shock : int
        Índice (1-based) del shock para IRFs.

    Returns
    -------
    draws : dict[str, np.ndarray]
        Diccionario {variable: draws} con shape:
        - IR: (reps, steps)
        - FC: (reps, past + horizon)
    """
    if model.method == 'signs':
        raise ValueError("No bootstrap in sign restriction SVAR")
    
    
    # BEGIN BOOTSTRAP
    b = 0

    if typ == "IR":
        draws = {var: np.zeros((reps, steps)) for var in model.variables}
    elif typ == "FC":
        draws = {var: np.zeros((reps, past + horizon)) for var in model.variables}
    elif typ == "VD":
        draws = {var: np.zeros((reps, steps)) for var in model.variables}
    else:
        raise ValueError("typ must be 'IR', 'FC' or 'VD'.")

    while True:
        # progreso
        if b in list(
            range(int(reps / 10) - 1, reps, int(reps / 10))
        ):
            print(str(b + 1) + "/" + str(reps))

        # ---------- tipo de remuestreo ----------
        if model.resampling == 1:  # normal
            resampled_resid = model.resid.copy()
            for i in range(model.n_obs):
                rnd = np.random.randint(0, model.n_obs)
                resampled_resid[i] = model.resid[rnd]
        elif model.resampling == 2:  # wild
            rsigns = np.random.choice([-1, 1], size=model.resid.shape[0])
            # Nota: igual que el código original, esto modifica model.resid en sitio
            resampled_resid = model.resid
            for i in range(model.resid.shape[0]):
                resampled_resid[i, :] = model.resid[i, :] * rsigns[i]
        else:
            raise ValueError("resampling must be 1 (normal) or 2 (wild).")

        # ---------- condición inicial para recursión hacia adelante ----------
        Y_sim = model.Y.copy()
        Y_0 = np.zeros((model.n_vars * model.lags,))

        for i in range(model.lags):
            Y_0[i * model.n_vars : (i + 1) * model.n_vars] = model.data[
                model.lags - i - 1, :
            ]

        # ---------- generar datos simulados ----------
        for i in range(model.n_obs):
            u = np.zeros((model.n_vars * model.lags,))
            u[0 : model.n_vars] = resampled_resid[i]

            if model.const == 0:
                FY_0 = model.F @ Y_0
                Y_0 = FY_0 + u
            elif model.const == 1:
                beta_const = np.zeros((model.n_vars * model.lags,))
                beta_const[0 : model.n_vars] = model.beta[:, 0]
                FY_0 = model.F @ Y_0
                Y_0 = beta_const + FY_0 + u
            elif model.const == 2:
                beta_const = np.zeros((model.n_vars * model.lags,))
                beta_const[0 : model.n_vars] = model.beta[:, 0]
                beta_trend = np.zeros((model.n_vars * model.lags,))
                beta_trend[0 : model.n_vars] = model.beta[:, 1]
                FY_0 = model.F @ Y_0
                Y_0 = beta_const + ((i + model.lags) * beta_trend) + FY_0 + u
            elif model.const == 3:
                beta_const = np.zeros((model.n_vars * model.lags,))
                beta_const[0 : model.n_vars] = model.beta[:, 0]
                beta_trend = np.zeros((model.n_vars * model.lags,))
                beta_trend[0 : model.n_vars] = model.beta[:, 1]
                beta_sqrd = np.zeros((model.n_vars * model.lags,))
                beta_sqrd[0 : model.n_vars] = model.beta[:, 2]
                FY_0 = model.F @ Y_0
                Y_0 = (
                    beta_const
                    + (i * beta_trend)
                    + ((i**2) * beta_sqrd)
                    + FY_0
                    + u
                )

            if model.DUM is not None:
                Y_0[0 : model.n_vars] = Y_0[0 : model.n_vars] + model.beta[
                    :,
                    model.const
                    + (model.n_vars * model.lags)
                    + model.n_ex : model.const
                    + (model.n_vars * model.lags)
                    + model.n_ex
                    + model.n_dum,
                ] @ model.D[i, :]

            Y_sim[i] = Y_0[0 : model.n_vars]

        Y_sim = np.vstack([model.data[0 : model.lags, :], Y_sim])

        # ---------- remuestreo del instrumento (método IV) ----------
        if model.method == "IV":
            iv = model.z
            if model.resampling == 2 and model.iv is not None:
                for i in range(model.n_iv):
                    for j in range(len(iv[i])):
                        iv[i][j] = iv[i][j] * rsigns[-len(iv[i]) :][j]

        # ---------- construir Ysim y Xsim ----------
        Ysim = Y_sim[model.lags :]
        Xsim_list = []

        for i in range(model.lags):
            Xsim_list.append(
                Y_sim[(model.lags - i - 1) : model.n_obs + model.lags - i - 1]
            )
        Xsim = np.hstack(Xsim_list)

        if model.const == 1:
            Xsim = np.hstack([np.ones((model.n_obs, 1)), Xsim])
        elif model.const == 2:
            trend = np.arange(1, model.n_obs + 1).reshape(-1, 1)
            Xsim = np.hstack([np.ones((model.n_obs - model.lags, 1)), trend, Xsim])
        elif model.const == 3:
            trend = np.arange(1, model.n_obs + 1).reshape(-1, 1)
            Xsim = np.hstack(
                [np.ones((model.n_obs, 1)), trend, trend**2, Xsim]
            )

        if model.data_exogenous is not None:
            if len(model.variables_exogenous) == model.n_obs:
                Xsim_exog = model.data_exogenous
            else:
                Xsim_exog = model.data_exogenous[-model.n_obs :]
                if len(Xsim_exog) != model.n_obs:
                    print("Exogenous variables in different domain")
            Xsim = np.hstack([Xsim, Xsim_exog])

        if model.DUM is not None:
            Xsim = np.hstack([Xsim, model.D[: model.n_obs, :]])

        # ---------- asegurar shapes ----------
        if isinstance(Ysim, pd.Series):
            Ysim = Ysim.values.reshape(-1, 1)
        if isinstance(Xsim, pd.Series):
            Xsim = Xsim.values.reshape(-1, 1)

        Ysim = Ysim.reshape(-1, 1) if Ysim.ndim == 1 else Ysim
        Xsim = Xsim.reshape(-1, 1) if Xsim.ndim == 1 else Xsim

        Xsim = np.array(Xsim, dtype=float)
        Ysim = np.array(Ysim, dtype=float)

        # ---------- estimar VAR simulado ----------
        simres=estimate_var(Ysim, Xsim, model.R)
        
        XXsim = Xsim.T @ Xsim
        XXsim_inv = np.linalg.inv(XXsim)
        betasim = simres.beta
        sigma_usim = simres.sigma_u
        residsim = simres.resid
    
        beta_stdsim = np.zeros((model.n_vars, model.n_cols))
        for i in range(model.n_vars):
            beta_stdsim[i, :] = np.sqrt(
                np.diagonal(sigma_usim[i, i] * XXsim_inv)
            )

        # ---------- construir Fsim ----------
        topsim = betasim[:, -(model.n_vars * model.lags) :]
        bottomsim = np.zeros(
            ((model.lags - 1) * model.n_vars, model.lags * model.n_vars)
        )
        for i in range((model.lags - 1) * model.n_vars):
            bottomsim[i, i] = 1
        Fsim = np.vstack([topsim, bottomsim])

        # ---------- identificar Bsim ----------
        if model.method == "short":
            Bsim = np.linalg.cholesky(sigma_usim)

        elif model.method == "long":
            M1 = np.eye(model.n_vars)
            for k in range(model.lags):
                M1 = M1 - Fsim[
                    0 : model.n_vars, k * model.n_vars : (k + 1) * model.n_vars
                ]
            M1 = np.linalg.inv(M1)
            D = np.linalg.cholesky(M1 @ sigma_usim @ M1.T)
            Bsim = np.linalg.solve(M1, D)

        elif model.method == "IV":
            usim = residsim
            for j in range(model.n_iv):
                XX_invfssim = (np.var(iv[j])) ** (-1)
                betafssim = XX_invfssim * (iv[j].T @ usim[-len(iv[j]) :, j])
                ujhatsim = iv[j] * betafssim
                Bsim = np.zeros((model.n_vars, model.n_vars))
                Bsim[j, j] = 1
                for i in range(model.n_vars):
                    XX_invsssim = (np.var(ujhatsim)) ** (-1)
                    if i != j:
                        betasssim = XX_invsssim * (
                            ujhatsim.T.T @ usim[-len(iv[j]) :, i]
                        )
                        Bsim[i, j] = betasssim
                Csim = np.linalg.cholesky(sigma_usim)
                q = solve_triangular(Csim, Bsim[:, j], lower=True)
                vsim = np.linalg.norm(q)
                # vsim = np.linalg.norm(Csim[:,j])
                Bsim[:, j] = Bsim[:, j] / vsim

        # ---------- almacenar draws ----------
        if typ == "IR":
            ir_b = impulse_response(Bsim, Fsim, shock, steps=steps, impact=impact)
        
            if np.any(Bsim != 0):
                if model.method in ("short", "long", "IV"):
                    for k in range(model.n_vars):
                        draws[model.variables[k]][b, :] = ir_b[:, k]
                    b += 1

        if typ == "FC":
            fc_b = compute_forecast(model, Fsim)
            if np.any(Bsim != 0):
                if model.method in ("short", "long", "IV"):
                    for k in range(model.n_vars):
                        draws[model.variables[k]][b, :] = fc_b[:, k]
                    b += 1
        
        if typ == "VD":
            vd_dict = forecast_error_variance_decomposition(
                Bsim, Fsim, steps=steps
            )
            if np.any(Bsim != 0):
                if model.method in ("short", "long", "IV"):
                    for k in range(model.n_vars):
                        # vd_dict[k + 1] gets the (steps, n_vars) FEVD matrix for variable k+1.
                        # [:, shock - 1] isolates the proportion of variance explained by 'shock'.
                        draws[model.variables[k]][b, :] = vd_dict[k + 1][:, shock - 1]
                    b += 1

        if b >= reps:
            break

    return draws
