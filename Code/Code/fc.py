#fc.py
import numpy as np

def compute_forecast(modelo, F):
    
    # 1. Initialize State Vector (start)
    start = np.zeros((modelo.n_vars * modelo.lags, 1))
    for i in range(modelo.lags):
        # Stacking lags: y_t, y_{t-1}, ...
        start[i*modelo.n_vars : (i+1)*modelo.n_vars, :] = \
            modelo.Y[modelo.n_obs - i - 1, :].reshape(-1, 1)
    
    # 2. Initialize Forecast Array
    Yfc = np.zeros((modelo.horizon + modelo.past, modelo.n_vars))
    Yfc[0:modelo.past, :] = modelo.Y[-modelo.past:, :]
    
    # 3. Pre-compute Deterministic Vectors (Optimization)
    # Moving this outside the loop prevents creating new arrays 120 times
    beta_const = np.zeros((modelo.n_vars * modelo.lags, 1))
    beta_trend = np.zeros((modelo.n_vars * modelo.lags, 1))
    beta_sqrd  = np.zeros((modelo.n_vars * modelo.lags, 1))
    
    if modelo.const >= 1:
        beta_const[0:modelo.n_vars, 0] = modelo.beta[:, 0]
    if modelo.const >= 2:
        beta_trend[0:modelo.n_vars, 0] = modelo.beta[:, 1]
    if modelo.const == 3:
        beta_sqrd[0:modelo.n_vars, 0]  = modelo.beta[:, 2]

    # 4. Forecast Loop
    for i in range(modelo.horizon):
        
        # A. Transition Step
        FYs = F @ start
        
        # B. Add Deterministic Terms (Const, Trend, etc)
        # Note: Ensure your trend logic (i + lags) matches your estimation definition
        if modelo.const == 1:
            FYs += beta_const
        elif modelo.const == 2:
            FYs += beta_const + (i * beta_trend)
        elif modelo.const == 3:
            FYs += beta_const + (i * beta_trend) + ((i**2) * beta_sqrd)
            
        # C. Add Dummies (if present)
        if modelo.DUM is not None:
            # Calculate dummy index offset for readability
            idx_start = modelo.const + (modelo.lags * modelo.n_vars) + modelo.n_ex
            
            if modelo.DUM == 'Q':
                dummy_coefs = modelo.beta[:, idx_start : idx_start+3]
                dummy_vals  = modelo.Dhorizon[i, :].reshape(-1, 1)
                FYs[0:modelo.n_vars, :] += dummy_coefs @ dummy_vals
                
            elif modelo.DUM == 'M':
                dummy_coefs = modelo.beta[:, idx_start : idx_start+11]
                dummy_vals  = modelo.Dhorizon[i, :].reshape(-1, 1)
                FYs[0:modelo.n_vars, :] += dummy_coefs @ dummy_vals

        # D. Store Forecast (THE FIX)
        # We take the first 'n_vars' rows from the 'FYs' column vector
        # and flatten them to fit into the Yfc row.
        Yfc[modelo.past + i, :] = FYs[0:modelo.n_vars, 0]
        
        # E. Update state for next iteration
        start = FYs

    return Yfc