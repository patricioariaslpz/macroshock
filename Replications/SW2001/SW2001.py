import pandas as pd
from macroshock_raw import SVAR

# --- DATA LOADING & SETUP ---
data1= pd.read_excel("SW2001_Data.xlsx")
df1=pd.DataFrame()
for i in data1.columns:
    df1[data1[i][0]]=data1[i][1:]

# --- INPUT ---
data=df1
variables=["infl", "unemp","ff"]
lags=4

# --- STOCK & WATSON (2001) MODEL ---
modelo=SVAR(data,
            variables,
            lags,
            alpha=5,
            past=20,
            reps=1000,
            steps=24
            )

# --- RUN THE MODEL ---
modelo.run()

# --- IMPULSE RESPONSE ---
modelo.ImpulseResponse(1)
modelo.ImpulseResponse(2)
modelo.ImpulseResponse(3)

# --- FEVD ---
modelo.VarianceDecomp(1)
modelo.VarianceDecomp(2)
modelo.VarianceDecomp(3)

# --- FORECAST ---
modelo.Forecast()