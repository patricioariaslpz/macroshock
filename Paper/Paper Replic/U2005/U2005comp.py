import pandas as pd
import numpy as np
from macroshock_raw import SVAR
import matplotlib.pyplot as plt

# --- DATA LOADING & SETUP ---
data3 = pd.read_excel("Uhlig2005_Data.xlsx")
data33 = pd.read_excel("U2005_IRFs.xlsx")

df3 = pd.DataFrame()
for i in data3.columns:
    df3[data3[i][0]] = data3[i][1:]

np.set_printoptions(suppress=True)

data3 = df3

variables = ['y', 'pi', 'comm', 'res', 'nbres', 'ff']
lags = 12
rescaling = [100, 100, 100, 100, 100, 1]
identification = 'signs'

signs = [
    [0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0]
]

modelo = SVAR(
    data3,
    variables,
    lags,
    rescaling=rescaling,
    method='signs',
    matrix_signs=signs,
    steps_signs=6,
    alpha=32,
    reps_default=500,
    steps=60
)

modelo.run()  

# --- IMPULSE RESPONSE COMPARISON ---
Imp = modelo.ImpulseResponse(1, show=False)

Imppoint = data33[['y_IRmed', 'pi_IRmed', 'comm_IRmed', 'res_IRmed', 'nbres_IRmed', 'ff_IRmed']]
Implow = data33[['y_IRinf', 'pi_IRinf', 'comm_IRinf', 'res_IRinf', 'nbres_IRinf', 'ff_IRinf']]
Imphigh = data33[['y_IRsup', 'pi_IRsup', 'comm_IRsup', 'res_IRsup', 'nbres_IRsup', 'ff_IRsup']]

fig, axes = plt.subplots(1, modelo.n_vars, figsize=(5 * modelo.n_vars, 5), sharex=False)
if modelo.n_vars == 1:
    axes = [axes] 

fig.suptitle(
    "Impulse-Response to shock 1",
    fontsize=23, 
    fontname="Times New Roman", 
    color="black"
)
    
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"
    
for i in range(modelo.n_vars):
    ax = axes[i]
    horizon = range(modelo.steps)
    zero_line = np.zeros(len(horizon))

    ax.plot(horizon, zero_line, color="black", linewidth=0.5)

    ax.fill_between(
        horizon,
        Imp['low'][variables[i]],
        Imp['high'][variables[i]],
        color="tomato",
        alpha=0.02,
        label="Intervalo de confianza"
    )
    ax.plot(horizon, Imp['low'][variables[i]], color="tomato", linestyle="--", linewidth=1)
    ax.plot(horizon, Imp['high'][variables[i]], color="tomato", linestyle="--", linewidth=1)
    ax.plot(horizon, Imp['point'][variables[i]], color="darkred", linewidth=2, label=str(modelo.variables[i]))
    
    ax.fill_between(
        horizon,
        Implow.values[:, i],
        Imphigh.values[:, i],
        color="teal",
        alpha=0.02,
        label="Intervalo de confianza"
    )
    ax.plot(horizon, Implow.values[:, i], color="teal", linestyle="--", linewidth=1)
    ax.plot(horizon, Imphigh.values[:, i], color="teal", linestyle="--", linewidth=1)
    ax.plot(horizon, Imppoint.values[:, i], color="teal", linewidth=2, label=str(modelo.variables[i]))
    
    ax.set_title(modelo.titles[i] if hasattr(modelo, 'titles') else modelo.variables[i], fontsize=20)
    ax.set_xlabel("Horizon", fontsize=16)
    if i == 0:
        ax.set_ylabel("Response", fontsize=16)

    ax.grid(False)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()