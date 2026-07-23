import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from macroshock_raw import SVAR

data1= pd.read_excel("SW2001_Data.xlsx")
data11= pd.read_excel("SW2001_IRFs.xlsx",sheet_name=["shock_1", "shock_2", "shock_3"])

d1=data11['shock_1']
d2=data11['shock_2']
d3=data11['shock_3']

df1=pd.DataFrame()
for i in data1.columns:
    df1[data1[i][0]]=data1[i][1:]

data=df1

variables=["infl", "unemp","ff"]
lags=4

modelo=SVAR(data,
            variables,
            lags,
            alpha=5,
            past=20,
            reps=1000,
            steps=24
            )

modelo.run()

Imp1=modelo.ImpulseResponse(1, show=False)
Imp2=modelo.ImpulseResponse(2, show=False)
Imp3=modelo.ImpulseResponse(3, show=False)
Imp=[Imp1, Imp2, Imp3]

Imp1bp=d1[['infl_IRbar','unemp_IRbar','ff_IRbar']]
Imp1bl=d1[['infl_IRinf','unemp_IRinf','ff_IRinf']]
Imp1bs=d1[['infl_IRsup','unemp_IRsup','ff_IRsup']]

Imp2bp=d2[['infl_IRbar','unemp_IRbar','ff_IRbar']]
Imp2bl=d2[['infl_IRinf','unemp_IRinf','ff_IRinf']]
Imp2bs=d2[['infl_IRsup','unemp_IRsup','ff_IRsup']]

Imp3bp=d3[['infl_IRbar','unemp_IRbar','ff_IRbar']]
Imp3bl=d3[['infl_IRinf','unemp_IRinf','ff_IRinf']]
Imp3bs=d3[['infl_IRsup','unemp_IRsup','ff_IRsup']]

Imppoint=[Imp1bp,Imp2bp,Imp3bp]
Implow=[Imp1bl,Imp2bl,Imp3bl]
Imphigh=[Imp1bs,Imp2bs,Imp3bs]

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
    
data_fevd = pd.read_excel("SW2001_FEVD.xlsx")
VD_excel = [data_fevd[['infl_infl','unemp_infl','ff_infl']]*0.01,
            data_fevd[['infl_unemp','unemp_unemp','ff_unemp']]*0.01,
            data_fevd[['infl_ff','unemp_ff','ff_ff']]*0.01]

VD1 = modelo.VarianceDecomp(shock=1, show=False)
VD2 = modelo.VarianceDecomp(shock=2, show=False)
VD3 = modelo.VarianceDecomp(shock=3, show=False)
VD_macro = [VD1, VD2, VD3]

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

        ax.plot(horizon, excel_point, color="teal", linestyle=":", marker='*', markersize=5, linewidth=2.5, label="Benchmark SW2001")

        title_text = modelo.titles[i] if hasattr(modelo, 'titles') else str(var_name)
        ax.set_title(f"{title_text} shock", fontsize=20)
        ax.set_xlabel("Horizon", fontsize=16)
        
        max_val = max(np.max(macro_point), np.max(excel_point))
        
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

modelo.Forecast()