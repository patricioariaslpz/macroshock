#data.py

import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class YXR:
    Y: np.ndarray
    X: np.ndarray
    R: np.ndarray

def YX(model) -> YXR:
    
    model.n_obs, _ = model.data.shape

    Y = model.data[model.lags:,:]
    X = []

    for i in range(model.lags):
        X.append(model.data[(model.lags-i-1) : model.n_obs - model.lags + (model.lags-i-1),:])
    X = np.hstack(X)
    
    if model.const == 1:
        X = np.hstack([np.ones((model.n_obs - model.lags, 1)), X])
    elif model.const == 2:
        trend = np.arange(1, model.n_obs - model.lags + 1).reshape(-1, 1)
        X = np.hstack([np.ones((model.n_obs - model.lags, 1)), trend, X])
    elif model.const == 3:
        trend = np.arange(1, model.n_obs - model.lags + 1).reshape(-1, 1)
        X = np.hstack([np.ones((model.n_obs - model.lags, 1)), trend, trend**2, X])
    
    model.n_obs, _ = X.shape
    
    if model.data_exogenous is not None:
        if model.n_obsex==model.n_obs:
            X_exog=model.data_exogenous
        else:
            X_exog=model.data_exogenous[-model.n_obs:]
            if len(X_exog)!=model.n_obs:
                print('Exogenous variables in different domain')
        X=np.hstack([X,X_exog])
        
    if model.DUM is not None:
        if model.DUM=='Q':
            model.dummies=['Q'+f"{i+1}" for i in range(4)]
            model.n_dum=3
            model.D=np.zeros((model.n_obs+model.horizon,4))
            for j in range(model.n_obs+model.horizon):
                for x in range(4):
                    if j in range(x,model.n_obs+model.horizon,4):
                        model.D[j,x]=1
        if model.DUM=='M':
            model.dummies=['M'+f"{i+1}" for i in range(12)]
            model.n_dum=11
            model.D=np.zeros((model.n_obs+model.horizon,12))
            for j in range(model.n_obs+model.horizon):
                for x in range(12):
                    if j in range(x,model.n_obs+model.horizon,12):
                        model.D[j,x]=1
        model.D=model.D[:,:model.n_dum]
        model.Dhorizon=model.D[model.n_obs:model.n_obs+model.horizon,:]
        X=np.hstack([X,model.D[0:model.n_obs,:]])
    else:
        model.n_dum=0

    model.Y=Y
    model.X=X
    
    if isinstance(model.Y, pd.Series):
        model.Y=model.Y.values.reshape(-1, 1)
    if isinstance(model.X, pd.Series):
        model.X=model.X.values.reshape(-1, 1)
        
    model.Y = model.Y.reshape(-1, 1) if model.Y.ndim == 1 else model.Y
    model.X = model.X.reshape(-1, 1) if model.X.ndim == 1 else model.X
    
    _ , model.n_vars=model.Y.shape
    model.n_obs, model.n_cols= model.X.shape
    
    model.X = np.array(model.X, dtype=float)
    model.Y = np.array(model.Y, dtype=float)
    
    if model.restrictions is None:
        model.R=np.eye(model.n_cols*model.n_vars)
        
    elif model.restrictions=='zeros':
        RR=np.eye(model.n_cols*model.n_vars)
        drop=[]
        
        for i in range(model.n_vars):
            for j in range(model.n_vars):
                if model.R[i][j]==0:
                    for k in range(model.lags):
                        drop.append(model.const*model.n_vars+i+(model.n_vars*j)+(k*model.n_vars*model.n_vars))
        
        model.R=np.delete(RR, drop, axis=1)
                    
    elif model.restrictions=='custom zeros':
        RR=model.R.ravel(order='F')
        RRR=np.eye(len(RR))
        drop=[]
        
        for i in range(len(RR)):
            if RR[i]==0:
                drop.append(i)
                
        model.R=np.delete(RRR, drop, axis=1)
        
    return YXR(Y=model.Y, X=model.X, R=model.R)

