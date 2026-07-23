import pandas as pd
from macroshock_raw import SVAR

# --- DATA LOADING & SETUP ---
data2= pd.read_excel("BQ1989_Data.xlsx")
df2=pd.DataFrame()
for i in data2.columns:
    df2[data2[i][0]]=data2[i][1:]
    
# --- INPUT ---
data2=df2
variables=["y", "u"]
lags=8

# --- BLANCHARD & QUAH (1989) MODEL ---
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

# --- RUN THE MODEL ---
modelo.run()

# --- IMPULSE RESPONSE ---
modelo.ImpulseResponse(1)
modelo.ImpulseResponse(2)

# --- FEVD ---
modelo.VarianceDecomp(1)
modelo.VarianceDecomp(2)

# --- FORECAST ---
modelo.Forecast()