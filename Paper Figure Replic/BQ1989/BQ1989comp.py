import pandas as pd
import numpy as np
from macroshock_raw import SVAR
import matplotlib.pyplot as plt

data2= pd.read_excel("BQ1989_Data.xlsx")
data22= pd.read_excel("BQ1989_IRFs.xlsx",sheet_name=["Shock_1", "Shock_2"])
d1=data22['Shock_1']
d2=data22['Shock_2']

df2=pd.DataFrame()
for i in data2.columns:
    df2[data2[i][0]]=data2[i][1:]

np.set_printoptions(suppress=True)

data2=df2
variables=["y", "u"]
lags=8
identification="long"

modelo=SVAR(data2,
            variables,
            lags,
            method='long',
            steps=40,
            resampling=1,
            alpha=5,
            reps=1000,
            past=20,
            horizon=10
            )

modelo.run()

Imp1=modelo.ImpulseResponse(1, show=False)
Imp2=modelo.ImpulseResponse(2, show=False)

Imp=[Imp1, Imp2]

Imp1bp=d1[['y_IRbar','u_IRbar']]
Imp1bl=d1[['y_IRinf','u_IRinf']]
Imp1bs=d1[['y_IRsup','u_IRsup']]

Imp2bp=d2[['y_IRbar','u_IRbar']]
Imp2bl=d2[['y_IRinf','u_IRinf']]
Imp2bs=d2[['y_IRsup','u_IRsup']]

Imppoint=[Imp1bp,Imp2bp]
Implow=[Imp1bl,Imp2bl]
Imphigh=[Imp1bs,Imp2bs]

for j in range(modelo.n_vars):
    
    fig, axes = plt.subplots(1, modelo.n_vars, figsize=(5*modelo.n_vars, 5), sharex=False)
    if modelo.n_vars == 1:
        axes = [axes] 

    fig.suptitle(f"Impulse-Response to shock {j+1}",
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
            Imp[j]['low'][modelo.variables[i]],
            Imp[j]['high'][modelo.variables[i]],
            color="tomato",
            alpha=0.02,
            label="Intervalo de confianza"
        )
        ax.plot(horizon, Imp[j]['low'][modelo.variables[i]], color="tomato", linestyle="--", linewidth=1)
        ax.plot(horizon, Imp[j]['high'][modelo.variables[i]], color="tomato", linestyle="--", linewidth=1)
        ax.plot(horizon, Imp[j]['point'][modelo.variables[i]], color="darkred", linewidth=2, label=str(modelo.variables[i]))
        
        ax.fill_between(
            horizon,
            Implow[j].values[:, i],
            Imphigh[j].values[:, i],
            color="teal",
            alpha=0.02,
            label="Intervalo de confianza"
        )
        ax.plot(horizon, Implow[j].values[:, i], color="teal", linestyle="--", linewidth=1)
        ax.plot(horizon, Imphigh[j].values[:, i], color="teal", linestyle="--", linewidth=1)
        ax.plot(horizon, Imppoint[j].values[:, i], color="teal", linewidth=2, label=str(modelo.variables[i]))
        
        title_text = modelo.titles[i] if hasattr(modelo, 'titles') else str(modelo.variables[i])
        ax.set_title(title_text, fontsize=20)
        ax.set_xlabel("Horizon", fontsize=16)
        if i == 0:
            ax.set_ylabel("Response", fontsize=16)
    
        ax.grid(False)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

data_fevd_dict = pd.read_excel("BQ1989_FEVD.xlsx")

VD_excel = [data_fevd_dict[['yy','uy']],data_fevd_dict[['yu','uu']]]

VD1 = modelo.VarianceDecomp(shock=1, show=False)
VD2 = modelo.VarianceDecomp(shock=2, show=False)
VD_macro = [VD1, VD2]

for j in range(modelo.n_vars):
    
    fig, axes = plt.subplots(1, modelo.n_vars, figsize=(5*modelo.n_vars, 5), sharex=False)
    if modelo.n_vars == 1:
        axes = [axes] 

    fig.suptitle(f"Variance Decomposition for {modelo.variables[j]}",
                 fontsize=23, fontname="Times New Roman", color="black")
    
    for i in range(modelo.n_vars):
        ax = axes[i]
        horizon = range(modelo.steps)
        var_name = modelo.variables[i]
        
        macro_point = VD_macro[j]['point'][var_name]
        ax.plot(horizon, macro_point, color="darkred", linestyle="-", linewidth=2.5, label="macroshock")
        
        if var_name in VD_excel[j].columns:
            excel_point = VD_excel[j][var_name]
        else:
            col_idx = i + 1 if 'step' in str(VD_excel[j].columns[0]).lower() else i
            excel_point = VD_excel[j].iloc[:, col_idx]

        if np.max(excel_point) > 1.5:
            excel_point = excel_point * 0.01

        ax.plot(horizon, excel_point, color="teal", linestyle=":", marker='*', markersize=5, linewidth=2.5, label="Benchmark BQ1989")

        title_text = modelo.titles[i] if hasattr(modelo, 'titles') else str(var_name)
        ax.set_title(f"{title_text} shock", fontsize=20)
        ax.set_xlabel("Horizon", fontsize=16)
        
        
        
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

modelo.Forecast()