import pandas as pd
from macroshock_raw import SVAR

# --- DATA LOADING & SETUP ---
data3 = pd.read_excel("Uhlig2005_Data.xlsx")
data33 = pd.read_excel("U2005_IRFs.xlsx")

df3 = pd.DataFrame()
for i in data3.columns:
    df3[data3[i][0]] = data3[i][1:]
    
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

modelo.ImpulseResponse(1)

modelo.VarianceDecomp()

modelo.Forecast()