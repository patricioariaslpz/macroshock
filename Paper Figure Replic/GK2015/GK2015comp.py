import pandas as pd
import numpy as np
from macroshock import SVAR
import matplotlib.pyplot as plt

data4= pd.read_excel("GK2015_Data.xlsx")
data44= pd.read_excel("GK2015_IRFs.xlsx")
df4=pd.DataFrame()
for i in data4.columns:
    df4[data4[i][0]]=data4[i][1:]
    
np.set_printoptions(suppress=True)

data4=df4

variables=['gs1','logcpi','logip','ebp']

lags=12

identification='IV'

signs=[[0,0,0,0],
       [0,0,0,0],
       [0,0,0,0],
       [0,0,0,0],
       ]

instrument='ff4_tc'

modelo=SVAR(data4,
            variables,
            lags,
            method=identification,
            iv=instrument,
            resampling=2,
            alpha=5,
            reps=200,
            steps=48
            )

modelo.run()

Imp=modelo.ImpulseResponse(1, show=False)

Imppoint=data44[['gs1_IRp','logcpi_IRp','logip_IRp','ebp_IRp']]
Implow=data44[['gs1_IRinf','logcpi_IRinf','logip_IRinf','ebp_IRinf']]
Imphigh=data44[['gs1_IRsup','logcpi_IRsup','logip_IRsup','ebp_IRsup']]

fig, axes = plt.subplots(1, modelo.n_vars, figsize=(5*modelo.n_vars, 5), sharex=False)
if modelo.n_vars == 1:
    axes = [axes] 

fig.suptitle(f"Impulse-Response to shock {1}",
                 fontsize=23, fontname="Times New Roman", color="black")
    
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
        Imp['low'][modelo.variables[i]],
        Imp['high'][modelo.variables[i]],
        color="tomato",
        alpha=0.02,
        label="Intervalo de confianza"
    )
    ax.plot(horizon, Imp['low'][modelo.variables[i]], color="tomato", linestyle="--", linewidth=1)
    ax.plot(horizon, Imp['high'][modelo.variables[i]], color="tomato", linestyle="--", linewidth=1)
    ax.plot(horizon, Imp['point'][modelo.variables[i]], color="darkred", linewidth=2, label=str(modelo.variables[i]))
    
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
    
    ax.set_title(modelo.titles[i], fontsize=20)
    ax.set_xlabel("Horizon", fontsize=16)
    if i == 0:
        ax.set_ylabel("Response", fontsize=16)

    ax.grid(False)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
