# stats.py
from __future__ import annotations

from typing import Union, Iterable

import numpy as np

# SciPy es obligatorio en esta versión
try:
    from scipy.stats import t as t_dist  # type: ignore[attr-defined]
except Exception as e:  # ImportError u otros
    raise ImportError(
        "The 'stats' module requires SciPy. "
        "Please install it with 'pip install scipy' before importing this module."
    ) from e


ArrayLike = Union[float, int, np.ndarray, Iterable[float]]


def t_cdf(t: ArrayLike, df: int) -> np.ndarray:
    """
    CDF de la distribución t de Student con df grados de libertad.

    Parameters
    ----------
    t : escalar o array-like
        Valores del estadístico t.
    df : int
        Grados de libertad (df > 0).

    Returns
    -------
    cdf : np.ndarray
        Valores de la CDF en los puntos t, con el mismo shape que la entrada.
    """
    df = int(df)
    if df <= 0:
        raise ValueError("df must be positive.")

    t_arr = np.asarray(t, dtype=float)
    # scipy.stats.t.cdf acepta arrays y devuelve un array del mismo shape
    cdf_vals = t_dist.cdf(t_arr, df)
    return np.asarray(cdf_vals, dtype=float)


def t_pvalue(t_stat: ArrayLike, df: int) -> np.ndarray:
    """
    p-value bilateral para un estadístico t con df grados de libertad,
    usando siempre la CDF exacta de SciPy.

    Parameters
    ----------
    t_stat : escalar o array-like
        Estadístico(s) t.
    df : int
        Grados de libertad.

    Returns
    -------
    pvals : np.ndarray
        p-values bilaterales, mismo shape que t_stat.
    """
    
    df = int(df)
    if df <= 0:
        raise ValueError("df must be positive.")
        
    t_abs = np.abs(np.asarray(t_stat, dtype=float))
    
    #cdf_vals = t_cdf(t_abs, df)
    #p_vals = 2.0 * (1.0 - cdf_vals) # p-valor bilateral = 2 * (1 - F(|t|))
    
    # Usamos sf (1 - cdf) directamente para mayor precisión en las colas
    # p-valor bilateral = 2 * P(T > |t|)
    p_vals = 2.0 * t_dist.sf(t_abs, df)
    
    return np.asarray(p_vals, dtype=float)
