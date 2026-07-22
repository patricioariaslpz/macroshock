import pandas as pd
from macroshock_raw import SVAR

data4= pd.read_excel("GK2015_Data.xlsx")
df4=pd.DataFrame()
for i in data4.columns:
    df4[data4[i][0]]=data4[i][1:]

data4=df4

variables=['gs1','logcpi','logip','ebp']

lags=12

identification='IV'

instrument='ff4_tc'

modelo=SVAR(data4,
            variables,
            lags,
            method=identification,
            iv=instrument,
            resampling=2,
            alpha=5,
            reps_default=200,
            steps=48
            #,DUM='Q'
            )

modelo.run()

modelo.ImpulseResponse(1)
