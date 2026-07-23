import pandas as pd
from macroshock_raw import SVAR

# --- DATA LOADING & SETUP ---
data3 = pd.read_excel("Uhlig2005_Data.xlsx")
df3 = pd.DataFrame()
for i in data3.columns:
    df3[data3[i][0]] = data3[i][1:]
    
# --- INPUT ---
data3 = df3
variables = ['y', 'pi', 'comm', 'res', 'nbres', 'ff']
lags = 12
rescaling = [100, 100, 100, 100, 100, 1]
identification = 'signs'
signs = [
    [ 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0],
    [ 1, 0, 0, 0, 0, 0]
    ]

# --- UHLIG (2005) MODEL ---
modelo = SVAR(
    data3,
    variables,
    lags,
    rescaling=rescaling,
    method='signs',
    matrix_signs=signs,
    steps_signs=6,
    alpha=32,
    reps=500,
    steps=60
    )

# --- RUN THE MODEL ---
modelo.run()  

# --- IMPULSE RESPONSE ---
modelo.ImpulseResponse(1)

# --- FEVD ---
modelo.VarianceDecomp(1)

# --- FORECAST ---
modelo.Forecast()
