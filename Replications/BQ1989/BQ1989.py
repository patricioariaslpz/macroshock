import pandas as pd
from macroshock_raw import SVAR

data2= pd.read_excel("BQ1989_Data.xlsx")

df2=pd.DataFrame()
for i in data2.columns:
    df2[data2[i][0]]=data2[i][1:]

data2=df2

variables=["y", "u"]

lags=8

modelo=SVAR(data2,
            variables,
            lags,
            method='long',
            steps=40,
            resampling=1,
            alpha=5,
            reps_default=100,
            past=20,
            horizon=10
            )

modelo.run()

modelo.ImpulseResponse(1)
modelo.ImpulseResponse(2)

modelo.VarianceDecomp()

modelo.Forecast()