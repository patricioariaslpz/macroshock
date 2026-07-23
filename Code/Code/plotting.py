from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_irf_svar(model, shock: int):
    """
    Replica el código de graficado de ImpulseResponse, pero fuera de la clase.
    Usa los atributos:
      - model.variables
      - model.titles
      - model.n_vars
      - model.steps
      - model.bands
      - model.ir_mean, model.ir_low, model.ir_high, etc.
    """
    fig, axes = plt.subplots(nrows=1, ncols=model.n_vars, 
                             figsize=(5 * model.n_vars, 5), 
                             sharex=True)
    if model.n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    if model.shocks is None:
        model.shocks=model.variables
        
    fig.suptitle(
        f"Impulse-Response to {model.shocks[shock-1]}",
        fontsize=23,
        fontname="Times New Roman",
        color="black",
    )

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "black"

    horizon = range(model.steps)
    zero_line = np.zeros(len(horizon))

    for i, var in enumerate(model.variables):
        ax = axes[i]

        ax.plot(horizon, zero_line, color="black", linewidth=0.5)

        if model.bands == "pair":
            ax.fill_between(
                horizon,
                model.ir_low[var],
                model.ir_high[var],
                color="tomato",
                alpha=0.1,
                label="Intervalo de confianza",
            )
            ax.plot(
                horizon,
                model.ir_low[var],
                color="tomato",
                linestyle="--",
                linewidth=1,
            )
            ax.plot(
                horizon,
                model.ir_high[var],
                color="tomato",
                linestyle="--",
                linewidth=1,
            )
            ax.plot(
                horizon,
                model.ir_mean[var],
                color="darkred",
                linewidth=2,
                label=str(var),
            )

        elif model.bands == "many":
            ax.fill_between(
                horizon, model.ir_00[var], model.ir_99[var], color="teal", alpha=0.05
            )
            ax.fill_between(
                horizon, model.ir_02[var], model.ir_97[var], color="teal", alpha=0.1
            )
            ax.fill_between(
                horizon, model.ir_05[var], model.ir_95[var], color="teal", alpha=0.15
            )
            
           
            ax.plot(
                horizon,
                model.ir_mean[var],
                color="teal",
                linewidth=2,
                label=str(var),
            )

        ax.set_title(
            model.titles[i] if model.titles is not None else var,
            fontsize=20,
        )
        ax.set_xlabel("Horizon", fontsize=16)
        if i == 0:
            ax.set_ylabel("Response", fontsize=16)

        ax.grid(False)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])


def plot_forecast_svar(model):
    """
    Replica el código de graficado de Forecast, pero fuera de la clase.
    Usa:
      - model.variables, model.titles
      - model.n_vars, model.horizon, model.past
      - model.bands
      - model.fc_mean, fc_low, fc_high, ..., fc_02, fc_97, etc.
    """
    fig, axes = plt.subplots(nrows=1, ncols=model.n_vars, 
                             figsize=(5 * model.n_vars, 5), 
                             sharex=True)
    if model.n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.suptitle("Forecast", fontsize=23, fontname="Times New Roman", color="black")

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "black"

    horizon = list(range(-model.past + 1, model.horizon + 1))
    zero_line = np.zeros(len(horizon))

    for i, var in enumerate(model.variables):
        ax = axes[i]

        y_min = model.fc_mean[var].min()
        y_max = model.fc_mean[var].max()

        ax.plot(horizon, zero_line, color="black", linewidth=0.5, zorder=0)

        if model.bands == "pair":
            y_min = min(y_min, model.fc_low[var].min())
            y_max = max(y_max, model.fc_high[var].max())
            axes[i].fill_between(
                horizon,
                model.fc_low[var],
                model.fc_high[var],
                color="tomato",
                alpha=0.1,
                label="Intervalo de confianza",
            )
            axes[i].plot(
                horizon,
                model.fc_low[var],
                color="tomato",
                linestyle="--",
                linewidth=1,
            )
            axes[i].plot(
                horizon,
                model.fc_high[var],
                color="tomato",
                linestyle="--",
                linewidth=1,
            )
            axes[i].plot(
                horizon,
                model.fc_mean[var],
                color="darkred",
                linewidth=2,
                label=str(var),
            )

        elif model.bands == "many":
            y_min = min(y_min, model.fc_02[var].min())
            y_max = max(y_max, model.fc_97[var].max())
            axes[i].fill_between(
                horizon, model.fc_00[var], model.fc_99[var], color="teal", alpha=0.05
            )
            axes[i].fill_between(
                horizon, model.fc_02[var], model.fc_97[var], color="teal", alpha=0.1
            )
            axes[i].fill_between(
                horizon, model.fc_05[var], model.fc_95[var], color="teal", alpha=0.15
            )
            
            axes[i].plot(
                horizon,
                model.fc_mean[var],
                color="teal",
                linewidth=2,
                label=str(var),
            )

        # márgenes alrededor de y_min/y_max (idéntico a tu lógica original)
        if y_min < 0:
            y_min = y_min - (0.1 * (y_max - y_min))
        else:
            y_min = y_min - (0.1 * (y_max - y_min))

        if y_max < 0:
            y_max = y_max + (0.1 * (y_max - y_min))
        else:
            y_max = y_max + (0.1 * (y_max - y_min))

        ax.plot([0, 0], [y_min, y_max], color="black", linewidth=1, linestyle="--")

        ax.set_ylim(y_min, y_max)

        ax.set_title(
            model.titles[i] if model.titles is not None else var,
            fontsize=20,
        )
        ax.set_xlabel("Horizon", fontsize=16)
        if i == 0:
            ax.set_ylabel("Value", fontsize=16)

        ax.grid(False)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

def plot_variance_decomp_svar(model, shock: int):
    """
    Grafica la Descomposición de Varianza (FEVD) para una sola variable de
    un modelo SVAR: un stackplot con la contribución de cada shock
    estructural a la varianza del error de pronóstico de esa variable, en
    cada horizonte.

    Parameters
    ----------
    model : objeto SVAR
        Debe tener .variables, .shocks, .steps y .vd_det poblados (p.ej.
        tras llamar model.VarianceDecomp()).
    shock : int
        Índice 1-based de la variable a graficar. Coincide directamente con
        las llaves de model.vd_det (indexado 1..n_vars por variable).
    """
    var_name = model.variables[shock - 1]

    horizon = np.arange(model.steps)

    # Check for assigned shocks
    if model.shocks is None:
        model.shocks = model.variables

    fig, ax = plt.subplots(figsize=(7, 4))

    fig.suptitle(
        f"Forecast Error Variance Decomposition: {var_name}",
        fontsize=13,
        fontname="Times New Roman",
        color="black"
    )

    # Generate a distinct color palette for the structural shocks
    colors = plt.cm.tab10(np.linspace(0, 1, len(model.shocks)))

    # Extract the deterministic FEVD matrix for the target variable.
    # model.vd_det[shock] is shape (steps, n_shocks); multiply by 100 and
    # transpose (.T) so stackplot can read shape (n_shocks, steps).
    y_data = (model.vd_det[shock] * 100).T

    ax.stackplot(
        horizon,
        y_data,
        labels=model.shocks,
        colors=colors,
        alpha=0.9
    )

    ax.set_ylim([0, 100])
    ax.set_xlim([0, model.steps - 1])
    ax.set_xlabel("Horizonte")
    ax.set_ylabel("Varianza Explicada (%)")
    ax.grid(True, axis='y', alpha=0.3)

    ax.legend(
        loc='lower center',
        ncol=len(model.shocks),
        bbox_to_anchor=(0.5, -0.45),
        frameon=False,
        fontsize=11
    )

    fig.tight_layout()
    plt.show()