import pandas as pd
from macroshock_raw import SVAR

# --- DATA LOADING & SETUP ---
data4= pd.read_excel("GK2015_Data.xlsx")
df4=pd.DataFrame()
for i in data4.columns:
    df4[data4[i][0]]=data4[i][1:]

# --- INPUT ---
data4=df4
variables=['gs1','logcpi','logip','ebp']
lags=12
instrument='ff4_tc'

# --- GERTLER & KARADI (2015) MODEL ---
modelo=SVAR(data4,
            variables,
            lags,
            method='IV',
            iv=instrument,
            resampling=2,
            alpha=5,
            reps=200,
            steps=48
            #,DUM='Q'
            )

# --- RUN THE MODEL ---
modelo.run()

# --- IMPULSE RESPONSE ---
modelo.ImpulseResponse(1)
