import pandas as pd
from macroshock_raw import SVAR

data1= pd.read_excel("SW2001_Data.xlsx")

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
            reps_default=100,
            steps=24
            )

modelo.run()

modelo.ImpulseResponse(1)
modelo.ImpulseResponse(2)
modelo.ImpulseResponse(3)

modelo.VarianceDecomp()

modelo.Forecast()