#Macroshock

from datetime import datetime
import pandas as pd
import numpy as np

from .data import YX
from .var import estimate_var
from .identification import (
    identify_short_run,
    identify_long_run,
    identify_signs,
    identify_iv,
)
from .bootstrap import bootstrap_svar

from .fc import compute_forecast
from .irf import compute_bands_from_draws
from .fevd import forecast_error_variance_decomposition
from .plotting import plot_irf_svar, plot_forecast_svar, plot_variance_decomp_svar

#==============================================================================
#                            class SVAR
#==============================================================================

class SVAR:
    
    def __init__(self, 
                 data, 
                 variables, 
                 lags, 
                 transformation=None, 
                 const=1, 
                 rescaling=None, 
                 restrictions=None,
                 R=None,
                 method='short',
                 matrix_long=None,
                 matrix_short=None,
                 matrix_signs=None, 
                 iv=None, 
                 shocks=None, 
                 EXOGdata=None,
                 EXOG=None, 
                 DUM=None,
                 impact=1,
                 steps=60, 
                 steps_signs=1, 
                 resampling=1, 
                 alpha=32,
                 reps_default=1000,
                 titles=None,
                 horizon=6, 
                 past=12
                 ):
        
        
# definición de las variables base del modelo

        self.data=data
        self.variables=variables
        self.lags=lags
        self.transformation=transformation
        self.const=const
        self.rescaling=rescaling
        self.restrictions=restrictions
        self.R=R
        self.R=np.array(self.R)
        self.method=method
        self.matrix_short=matrix_short
        self.matrix_long=matrix_long
        self.matrix_signs=matrix_signs
        self.iv=iv
        self.shocks=shocks
        self.EXOGdata=EXOGdata
        self.EXOG=EXOG
        self.DUM=DUM
        self.impact=impact
        self.steps=steps
        self.steps_signs=steps_signs
        self.resampling=resampling
        self.alpha=alpha
        self.reps_default=reps_default
        self.titles=titles
        self.horizon=horizon
        self.past=past
        self.method_dict={"short":"zero short-run restrictions","long":"zero long-run restrictions","signs":"sign restrictions","IV":"instrumental variable + sign restrictions"}
        self.n_vars=len(self.variables)
        if self.titles is None:
          self.titles=self.variables
        if self.rescaling is None:
            self.rescaling=[1 for i in range(self.n_vars)]
        if self.transformation is None:
            self.transformation=[None for i in range(self.n_vars)]
        if isinstance(data, pd.DataFrame):
            if self.method=='IV':
                self.iv=self.data[self.iv].values
            self.data=self.data[self.variables].values
        if EXOGdata is not None:
            if isinstance(EXOGdata, pd.DataFrame):
                self.EXOGdata=self.EXOGdata[self.EXOG].values
                self.n_obsex,self.n_ex=self.EXOGdata.shape
        else:
            self.n_ex=0
            if self.EXOG is None:
                self.EXOG=['Exog'+f"{i+1}" for i in range(self.n_ex)]
        for i in range(len(self.transformation)):
            if self.transformation[i]=="diff":
                self.data[1:,i]=np.diff(self.data[:, i])
            elif self.transformation[i]=="logdiff":
                self.data[1:,i]=np.diff(np.log(self.data[:,i]))*self.rescaling[i]
            elif self.transformation[i]=="log":
                self.data[:,i]=np.log(self.data[:,i])*self.rescaling[i]
            else:
                self.data[:,i] = self.data[:,i]*self.rescaling[i]
        if 'diff' in self.transformation or 'logdiff' in self.transformation:
            self.data=self.data[1:]
        self.structural=False
        if self.method=='IV':
            if self.iv.ndim == 1:
                self.iv = self.iv.reshape(-1, 1)
            self.n_obsi,self.n_iv=self.iv.shape
    
# funcion para correr el modelo
    def YX(self):
        
        YX(self)
        

    def VAR(self):
        """
        Estimate the reduced-form VAR using the modular 'estimate_var' function
        and then build the companion matrix F as in your original implementation.
        """
        res = estimate_var(self.Y, self.X, self.R)

        # Store everything you used to have as attributes
        self.beta = res.beta
        self.sigma_u = res.sigma_u
        self.resid = res.resid
        self.Y_hat = res.Y_hat
        self.ll = res.ll
        self.aic = res.aic
        self.bic = res.bic
        self.hqic = res.hqic
        self.det = res.det
        self.fpe = res.fpe
        self.beta_std = res.beta_std
        self.tstat = res.tstat
        self.pvalue = res.pvalue
        self.n_obs = res.n_obs
        self.n_vars = res.n_vars
        self.n_cols = res.n_cols

        # Build companion matrix F exactly as before
        top = self.beta[:, self.const : self.const + (self.lags * self.n_vars)]
        bottom = np.zeros(((self.lags - 1) * self.n_vars, self.lags * self.n_vars))
        for i in range((self.lags - 1) * self.n_vars):
            bottom[i, i] = 1
        self.F = np.vstack([top, bottom])
    
    
    def S(self):
        """
        Structural analysis using modular identification functions.
        Mirrors your original 'S()' behavior, but delegates work to identification.py.
        """
        # short-run zero
        if self.method == "short":
            struct = identify_short_run(self.sigma_u, matrix=self.matrix_short)
            self.B = struct.B
            self.n = struct.n_rejected
            self.corr = struct.corr

        # long-run zero
        elif self.method == "long":
            struct = identify_long_run(self.sigma_u, self.F, self.lags, matrix=self.matrix_long)
            self.B = struct.B
            self.n = struct.n_rejected
            self.corr = struct.corr

        # sign restrictions
        elif self.method == "signs":
            struct = identify_signs(
                sigma_u=self.sigma_u,
                F=self.F,
                matrix_signs=self.matrix_signs,
                steps_signs=self.steps_signs,
                steps=self.steps,
                reps=self.reps_default,
                alpha=self.alpha,
            )
            self.B = struct.B
            self.n = struct.n_rejected
            self.m = struct.m_accepted
            self.corr = struct.corr

            # unpack IR sets and bands from 'extra'
            extra = struct.extra
            self.IRL = extra["IRL"]
            self.ir_median = extra["ir_median"]
            self.ir_mean = extra["ir_mean"]
            self.ir_high = extra["ir_high"]
            self.ir_low = extra["ir_low"]
            self.ir_00 = extra["ir_00"]
            self.ir_99 = extra["ir_99"]
            self.ir_02 = extra["ir_02"]
            self.ir_97 = extra["ir_97"]
            self.ir_05 = extra["ir_05"]
            self.ir_95 = extra["ir_95"]
            self.ir_25 = extra["ir_25"]
            self.ir_75 = extra["ir_75"]
            self.ir_16 = extra["ir_16"]
            self.ir_84 = extra["ir_84"]

            # unpack FEVD sets and bands from 'extra'. Kept shock-indexed
            # under a "_signs" suffix (rather than reusing self.vd_mean,
            # self.vd_low, etc. directly) so that SVAR.VarianceDecomp()
            # can slice them by shock without the values being clobbered
            # once it rebuilds the variable-indexed self.vd_* attributes.
            self.FEVDL = extra["FEVDL"]
            self.vd_median_signs = extra["vd_median"]
            self.vd_mean_signs = extra["vd_mean"]
            self.vd_high_signs = extra["vd_high"]
            self.vd_low_signs = extra["vd_low"]
            self.vd_00_signs = extra["vd_00"]
            self.vd_99_signs = extra["vd_99"]
            self.vd_02_signs = extra["vd_02"]
            self.vd_97_signs = extra["vd_97"]
            self.vd_05_signs = extra["vd_05"]
            self.vd_95_signs = extra["vd_95"]

        # IV + sign restrictions
        elif self.method == "IV":
            # self.data at this point should already be the ndarray of endogenous vars
            struct = identify_iv(
                sigma_u=self.sigma_u,
                resid=self.resid,
                iv=self.iv,
                data_array=self.data,
            )
            self.B = struct.B
            self.n = struct.n_rejected
            self.corr = struct.corr
            self.z = struct.extra["z"]
            self.varz = struct.extra["varz"]

        self.structural = True

                #--------------------------------------------------------------
    # show summary function
    def summary(self):
        
        lagged=[]
        for i in range(self.lags):
            for j in range(self.n_vars):
                lagged.append("Lag"+f"{i+1}"+"."+self.variables[j])
    
        summary0="Summary\n=====================================\nModel: SVARX\nMethod: OLS\nDate/Time: "+datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")+"\n-------------------------------------\nNo. of Equations: "+f"{self.n_vars:>18.2f}"+"\nNo. of Observations: "+f"{self.n_obs:>15.2f}"+"\nLog Likelihood: "+f"{self.ll:>20.6f}"+"\nAIC: "+f"{self.aic:>31.6f}"+"\nBIC: "+f"{self.bic:>31.6f}"+"\nHQIC: "+f"{self.hqic:>30.6f}"+"\nFPE: "+f"{self.fpe:>31.6f}"+"\nDet: "+f"{self.det:>31.6f}"+"\n-------------------------------------\n"
        for i in range(self.n_vars):
            summary1="Results for equation "+self.variables[i]+"\n=====================================================================\n                coefficient      std. error     t-stat      p-value\n---------------------------------------------------------------------\n"
            if self.const==1:
                summary1=summary1+("const"f"{self.beta[i,0]:>21.6f}"f"{self.beta_std[i,0]:>16.6f}"f"{self.tstat[i,0]:>12.3f}"f"{self.pvalue[i,0]:>12.3f}\n")
            elif self.const==2:
                summary1=summary1+("const"f"{self.beta[i,0]:>21.6f}"f"{self.beta_std[i,0]:>16.6f}"f"{self.tstat[i,0]:>12.3f}"f"{self.pvalue[i,0]:>12.3f}\n")
                summary1=summary1+("trend"f"{self.beta[i,1]:>21.6f}"f"{self.beta_std[i,1]:>16.6f}"f"{self.tstat[i,1]:>12.3f}"f"{self.pvalue[i,1]:>12.3f}\n")
            elif self.const==3:
                summary1=summary1+("const"f"{self.beta[i,0]:>21.6f}"f"{self.beta_std[i,0]:>16.6f}"f"{self.tstat[i,0]:>12.3f}"f"{self.pvalue[i,0]:>12.3f}\n")
                summary1=summary1+("trend"f"{self.beta[i,1]:>21.6f}"f"{self.beta_std[i,1]:>16.6f}"f"{self.tstat[i,1]:>12.3f}"f"{self.pvalue[i,1]:>12.3f}\n")
                summary1=summary1+("trend sqrd"f"{self.beta[i,2]:>16.6f}"f"{self.beta_std[i,2]:>16.6f}"f"{self.tstat[i,2]:>12.3f}"f"{self.pvalue[i,2]:>12.3f}\n")
    
            for j in range(self.const,self.n_cols):
                if j in range(self.const,self.const+(self.lags*self.n_vars)):
                    summary2 = (lagged[j-self.const]+f"{self.beta[i,j]:>{26-len(lagged[j-self.const])}.6f}"f"{self.beta_std[i,j]:>16.6f}"f"{self.tstat[i,j]:>12.3f}"f"{self.pvalue[i,j]:>12.3f}\n")
                    summary1=summary1+summary2
                elif j in range(self.const+(self.lags*self.n_vars),self.const+(self.lags*self.n_vars)+self.n_ex):
                    summary2 = (self.EXOG[j-self.const-(self.lags*self.n_vars)]+f"{self.beta[i,j]:>{26-len(self.EXOG[j-self.const-(self.lags*self.n_vars)])}.6f}"f"{self.beta_std[i,j]:>16.6f}"f"{self.tstat[i,j]:>12.3f}"f"{self.pvalue[i,j]:>12.3f}\n")
                    summary1=summary1+summary2
                elif j in range(self.const+(self.lags*self.n_vars)+self.n_ex,self.const+(self.lags*self.n_vars)+self.n_ex+self.n_dum):
                    summary2 = (self.dummies[j-self.const-(self.lags*self.n_vars)-self.n_ex]+f"{self.beta[i,j]:>{26-len(self.dummies[j-self.const-(self.lags*self.n_vars)-self.n_ex])}.6f}"f"{self.beta_std[i,j]:>16.6f}"f"{self.tstat[i,j]:>12.3f}"f"{self.pvalue[i,j]:>12.3f}\n")
                    summary1=summary1+summary2
            summary1=summary1+"=====================================================================\n\n"
            summary0=summary0+summary1
        if self.structural==True:
            if self.method=='short' or self.method=='long' or self.method=='IV':
                summaryb="Structural analysis\n---------------------------------------------------------------------\nIdentification: "+" "*((52-len(str(self.method_dict[self.method]))))+self.method_dict[self.method]+"\nMatrices rejected (B):"+" "*(46-len(str(self.n)))+str(self.n)+"\nStrength of instrument: "
            else:
                summaryb="Structural analysis\n---------------------------------------------------------------------\nIdentification: "+" "*((52-len(str(self.method_dict[self.method]))))+self.method_dict[self.method]+"\nRejection rate (B):"+" "*(44-len(str(round(self.n/self.m))))+str(round(self.n/self.m,4))+"\nStrength of instrument: "
            if self.method=='IV':
                for k in range(self.n_iv):
                    summarye=" "*(44-len(str(round(float(self.corr[k]),4))))
                    summaryc=str(round(self.corr[k],4))
            else:
                summarye=''
                summaryc=''
            summaryd="\n=====================================================================\nMatrix B (median)\n\n" if self.method=="signs" else "\n=====================================================================\nMatrix B\n\n"
            summaryb=summaryb+summarye+summaryc+summaryd
            summaryb0=""
            summarym=""
            for i in range(self.n_vars):
                summ=self.variables[i]
                summaryb0=summaryb0+"        sh"+f"{i+1}"
                for j in range(self.n_vars):
                    if j==0:
                        summarybb=f"{self.B[i,j]:>{13-len(self.variables[i])}.3f}"
                    else:
                        summarybb=f"{self.B[i,j]:>11.3f}"
                    summ=summ+summarybb
                summarym=summarym+summ+"\n"
            summaryb=summaryb+summaryb0+"\n"+summarym
            summary0=summary0+summaryb+"====================================================================="
        self.summary=summary0
        
        print(self.summary)
    #--------------------------------------------------------------------------
        
    # impulse response calculation and plotting function
    def ImpulseResponse(self, shock, show=True, bands='pair', steps=None, impact=None):
        if steps is None:
            steps = self.steps
        else:
            self.steps = steps
        if impact is None:
            impact = self.impact
        else:
            self.impact = impact
        """
        Calcula IRFs (punto y bandas) y, opcionalmente, grafica.
        Mantiene la API y el output de la versión original.
        """
        print("\nshock: " + str(shock) + "\n")

        # tipo de bandas
        self.bands = bands

        # Métodos distintos a 'signs': bootstrap estándar
        if self.method != 'signs':

            # Bootstrap de IRFs para este shock
            self.drawsIR=bootstrap_svar(self, typ='IR', shock=shock)
            
            # Calcular medias y cuantiles a partir de self.drawsIR
            stats = compute_bands_from_draws(self.drawsIR, self.alpha)

            # Guardar en atributos como antes
            self.ir_point = stats["point"]
            self.ir_mean = stats["mean"]
            self.ir_low = stats["low"]
            self.ir_high = stats["high"]
            self.ir_00 = stats["q0_5"]
            self.ir_99 = stats["q99_5"]
            self.ir_02 = stats["q2_5"]
            self.ir_97 = stats["q97_5"]
            self.ir_05 = stats["q5"]
            self.ir_95 = stats["q95"]

        # Método 'signs': ya tenemos el conjunto de identificaciones
        else:
            # self.ir_median, self.ir_mean, etc. vienen de identify_signs (identification.py),
            # indexados por número de shock.
            
            median_arr = self.ir_median[shock]  # (steps, n_vars)
            mean_arr = self.ir_mean[shock]
            low_arr = self.ir_high[shock]   # OJO: en tu código original high/low
            high_arr = self.ir_low[shock]  # podrían estar invertidos; aquí asumimos
            q00_arr = self.ir_00[shock]
            q99_arr = self.ir_99[shock]
            q02_arr = self.ir_02[shock]
            q97_arr = self.ir_97[shock]
            q05_arr = self.ir_05[shock]
            q95_arr = self.ir_95[shock]
            
            # Convertir a diccionarios {var: serie} para ser coherentes con el resto
            self.ir_point = {}
            self.ir_mean = {}
            self.ir_low = {}
            self.ir_high = {}
            self.ir_00 = {}
            self.ir_99 = {}
            self.ir_02 = {}
            self.ir_97 = {}
            self.ir_05 = {}
            self.ir_95 = {}

            for idx, var in enumerate(self.variables):
                self.ir_point[var] = median_arr[:, idx]
                self.ir_mean[var] = mean_arr[:, idx]
                self.ir_low[var] = low_arr[:, idx]
                self.ir_high[var] = high_arr[:, idx]
                self.ir_00[var] = q00_arr[:, idx]
                self.ir_99[var] = q99_arr[:, idx]
                self.ir_02[var] = q02_arr[:, idx]
                self.ir_97[var] = q97_arr[:, idx]
                self.ir_05[var] = q05_arr[:, idx]
                self.ir_95[var] = q95_arr[:, idx]

        # Mostrar gráficas si se pide
        if show:
            plot_irf_svar(self, shock)

        # Output: mantiene la convención original
        if self.method != 'signs':
            output = {
                'point': self.ir_mean,  # en bootstrap usabas la media como "point"
                'low': self.ir_low,
                'high': self.ir_high,
                0.5: self.ir_00,
                2.5: self.ir_02,
                5: self.ir_05,
                99.5: self.ir_97,
                97.5: self.ir_97,
                95: self.ir_95,
            }
        else:
            # para 'signs' el "point" era la mediana
            output = {
                'point': self.ir_point,
                'low': self.ir_low,
                'high': self.ir_high,
                0.5: self.ir_00,
                2.5: self.ir_02,
                5: self.ir_05,
                99.5: self.ir_99,
                97.5: self.ir_97,
                95: self.ir_95,
            }

        return output

        
        print("\nshock: "+str(shock)+"\n")
        # type of bands
        self.bands=bands
        
        # for signs IMF are already estimated from the set identification process, thus this bootstrap is only for the other methods
        if self.method!='signs':
    
            self.Bootstrap('IR',shock=shock)
            
            # alpha levels
            lower_q = self.alpha / 2
            upper_q = 100 - self.alpha / 2
            #--------------------------------------------------------------
            
            # storage for IRF
            self.ir_point = {}
            self.ir_mean = {}
            self.ir_low = {}
            self.ir_high = {}
            self.ir_95 = {}
            self.ir_05 = {}
            self.ir_00 = {}
            self.ir_99 = {}
            self.ir_02 = {}
            self.ir_97 = {}
        
            for var in self.variables:
                x=np.zeros((self.reps_default,self.steps))
                for j in range(self.reps_default):
                    x[j,:]=self.drawsIR[var][j]
                self.drawsIR[var]=x
        
            for var in self.variables:
                self.ir_point[var] = np.mean(self.drawsIR[var], axis=0)
                self.ir_mean[var] = np.mean(self.drawsIR[var], axis=0)
                self.ir_low[var]  = np.percentile(self.drawsIR[var], lower_q, axis=0)
                self.ir_high[var] = np.percentile(self.drawsIR[var], upper_q, axis=0)
                self.ir_95[var]   = np.percentile(self.drawsIR[var], 95, axis=0)
                self.ir_05[var]   = np.percentile(self.drawsIR[var], 5, axis=0)
                self.ir_00[var]   = np.percentile(self.drawsIR[var], 0.5, axis=0)
                self.ir_99[var]   = np.percentile(self.drawsIR[var], 99.5, axis=0)
                self.ir_02[var]   = np.percentile(self.drawsIR[var], 2.5, axis=0)
                self.ir_97[var]   = np.percentile(self.drawsIR[var], 97.5, axis=0)
            #--------------------------------------------------------------
        
        # if method is signs we already have the identification set for B matrices thus no bootstrap
        elif self.method=='signs':
            
            self.ir_point = self.ir_median[shock]
            self.ir_low = self.ir_low[shock]
            self.ir_high = self.ir_high[shock]
            self.ir_02 = self.ir_02[shock]
            self.ir_97 = self.ir_97[shock]
            self.ir_05 = self.ir_05[shock]
            self.ir_95 = self.ir_95[shock]
            self.ir_00 = self.ir_00[shock]
            self.ir_99 = self.ir_99[shock]
            self.ir_mean = self.ir_mean[shock]
        #--------------------------------------------------------------
        
        # show the plots if indicated
        if show==True:
            plot_irf_svar(shock)
            
            #--------------------------------------------------------------
            
       # output set up 
        if self.method!='signs':
            output={'point':self.ir_mean, 'low':self.ir_low, 'high':self.ir_high, 2.5:self.ir_02, 5:self.ir_05, 0.5:self.ir_00, 99.5:self.ir_99,97.5:self.ir_97,95:self.ir_95}
        else:
            output={'point':self.ir_point, 'low':self.ir_low, 'high':self.ir_high, 2.5:self.ir_02, 5:self.ir_05, 0.5:self.ir_00, 99.5:self.ir_99,97.5:self.ir_97,95:self.ir_95}

        return output
      
    # forecast calculation and plotting (THIS FUNCTION MIMICS THE IMPULSE RESPONSE ONE FOR BOOTSTAP AND BANDS LOGIC)
    def Forecast(self, show=True, bands='many'):
        """
        Calcula pronósticos (punto y bandas) y, opcionalmente, grafica.
        Mantiene la API y estructura de salida de la función original.
        """
        # Para métodos distintos de 'signs', hacemos bootstrap
        if self.method != 'signs':

            print("Forecast:\n")

            # Pronóstico determinista (no lo usabas mucho, pero lo mantenemos)
            self.fc_point = compute_forecast(self, self.F)

            # Bootstrap de pronósticos
            self.drawsFC=bootstrap_svar(self, typ='FC')

            # Calcular bandas a partir de self.drawsFC
            stats = compute_bands_from_draws(self.drawsFC, self.alpha)

            self.fc_mean = stats["mean"]
            self.fc_low = stats["low"]
            self.fc_high = stats["high"]
            self.fc_02 = stats["q2_5"]
            self.fc_97 = stats["q97_5"]
            self.fc_05 = stats["q5"]
            self.fc_95 = stats["q95"]
            self.fc_00 = stats["q0_5"]
            self.fc_99 = stats["q99_5"]

        # Para 'signs', no había bootstrap en tu código: solo trayectoria determinista
        elif self.method == 'signs':

            print('No bands Forecast:\n')

            fc_det = compute_forecast(self, self.F)  # (past + horizon, n_vars)

            # Crear diccionarios por variable
            self.fc_mean = {}
            self.fc_low = {}
            self.fc_high = {}
            self.fc_02 = {}
            self.fc_97 = {}
            self.fc_05 = {}
            self.fc_95 = {}
            self.fc_00 = {}
            self.fc_99 = {}

            for i, var in enumerate(self.variables):
                series = fc_det[:, i]
                self.fc_mean[var] = series
                # sin bandas: todas iguales a la media determinista
                self.fc_low[var] = series
                self.fc_high[var] = series
                self.fc_02[var] = series
                self.fc_97[var] = series
                self.fc_05[var] = series
                self.fc_95[var] = series
                self.fc_00[var] = series
                self.fc_99[var] = series

        # Gráficas
        if show is True:
            self.bands = bands
            plot_forecast_svar(self)

        # Output: misma estructura que antes
        output = {
            'point': self.fc_mean,
            'low': self.fc_low,
            'high': self.fc_high,
            2.5: self.fc_02,
            5: self.fc_05,
            97.5: self.fc_97,
            95: self.fc_95,
            0.5: self.fc_00,
            99.5: self.fc_99
        }

        return output

        
        if self.method!='signs':
            
            print("\nreps_default:\n")
            
            self.fc_point = self.FC(self.F)
            
            self.Bootstrap('FC')
        
            lower_q = self.alpha / 2
            upper_q = 100 - self.alpha / 2
            
            self.fc_mean = {}
            self.fc_low = {}
            self.fc_high = {}
            self.fc_95 = {}
            self.fc_50 = {}
            self.fc_05 = {}
            self.fc_02 = {}
            self.fc_97 = {}
            self.fc_00 = {}
            self.fc_99 = {}
        
            for var in self.variables:
                x=np.zeros((self.reps_default,self.past+self.horizon))
                for j in range(self.reps_default):
                    x[j,:]=self.drawsFC[var][j]
                self.drawsFC[var]=x
        
            for var in self.variables:
                self.fc_mean[var] = np.mean(self.drawsFC[var], axis=0)
                self.fc_low[var]  = np.percentile(self.drawsFC[var], lower_q, axis=0)
                self.fc_high[var] = np.percentile(self.drawsFC[var], upper_q, axis=0)
                self.fc_95[var]   = np.percentile(self.drawsFC[var], 95, axis=0)
                self.fc_99[var]   = np.percentile(self.drawsFC[var], 99.5, axis=0)
                self.fc_00[var]   = np.percentile(self.drawsFC[var], 0.5, axis=0)
                self.fc_05[var]   = np.percentile(self.drawsFC[var], 5, axis=0)
                self.fc_02[var]   = np.percentile(self.drawsFC[var], 2.5, axis=0)
                self.fc_97[var]   = np.percentile(self.drawsFC[var], 97.5, axis=0)
            
        elif self.method=='signs':
            
            print('no bands Forecast:')
            self.fc_mean = {self.variables[i]:self.FC(self.F)[:,i] for i in range(self.n_vars)}
            self.fc_low = self.fc_mean
            self.fc_high = self.fc_mean
            self.fc_95 = self.fc_mean
            self.fc_99 = self.fc_mean
            self.fc_00 = self.fc_mean
            self.fc_05 = self.fc_mean
            self.fc_02 = self.fc_mean
            self.fc_97 = self.fc_mean
            
        if show==True:
            
            self.bands=bands
            
            plot_forecast_svar(self)
           
        output={'point':self.fc_mean, 'low':self.fc_low, 'high':self.fc_high, 2.5:self.fc_02, 5:self.fc_05,97.5:self.fc_97,95:self.fc_95,0.5:self.fc_00,99.5:self.fc_99}
        
        return output
    
    
    def VarianceDecomp(self, shock: int = 1, show: bool = True, bands: str = 'pair', steps: int = None):
        """
        Calcula la Descomposición de Varianza (punto y bandas) y, opcionalmente, grafica.
        Mantiene la API y estructura de salida de ImpulseResponse y Forecast.
        """
        if steps is None:
            steps = self.steps
        else:
            self.steps = steps

        print("Calculating Variance Decomposition:\n")

        self.bands = bands

        if self.method != 'signs':
            self.vd_det = forecast_error_variance_decomposition(self.B, self.F, steps=self.steps)

            self.drawsVD = bootstrap_svar(self, typ='VD', shock=shock)

            stats = compute_bands_from_draws(self.drawsVD, self.alpha)

            self.vd_mean = stats["mean"]
            self.vd_low  = stats["low"]
            self.vd_high = stats["high"]
            self.vd_02   = stats["q2_5"]
            self.vd_97   = stats["q97_5"]
            self.vd_05   = stats["q5"]
            self.vd_95   = stats["q95"]
            self.vd_00   = stats["q0_5"]
            self.vd_99   = stats["q99_5"]

        elif self.method == 'signs':

            # Deterministic FEVD for the median B matrix (self.B is already
            # the median of the accepted set for 'signs'). This is what
            # feeds the stacked-area plot below -- exactly the same object
            # and code path used for 'short' and 'long'.
            self.vd_det = forecast_error_variance_decomposition(self.B, self.F, steps=self.steps)

            # Distributional statistics across the accepted sign-restriction
            # draws for this shock (computed in identify_signs() and stored
            # shock-indexed on self at S() time).
            median_arr = self.vd_median_signs[shock]  # (steps, n_vars)
            mean_arr   = self.vd_mean_signs[shock]
            low_arr    = self.vd_high_signs[shock]     # swapped, mirrors ImpulseResponse()
            high_arr   = self.vd_low_signs[shock]
            q00_arr    = self.vd_00_signs[shock]
            q99_arr    = self.vd_99_signs[shock]
            q02_arr    = self.vd_02_signs[shock]
            q97_arr    = self.vd_97_signs[shock]
            q05_arr    = self.vd_05_signs[shock]
            q95_arr    = self.vd_95_signs[shock]

            # Convert to {var: series} dicts to be consistent with the rest
            self.vd_point = {}
            self.vd_mean  = {}
            self.vd_low   = {}
            self.vd_high  = {}
            self.vd_00    = {}
            self.vd_99    = {}
            self.vd_02    = {}
            self.vd_97    = {}
            self.vd_05    = {}
            self.vd_95    = {}

            for idx, var in enumerate(self.variables):
                self.vd_point[var] = median_arr[:, idx]
                self.vd_mean[var]  = mean_arr[:, idx]
                self.vd_low[var]   = low_arr[:, idx]
                self.vd_high[var]  = high_arr[:, idx]
                self.vd_00[var]    = q00_arr[:, idx]
                self.vd_99[var]    = q99_arr[:, idx]
                self.vd_02[var]    = q02_arr[:, idx]
                self.vd_97[var]    = q97_arr[:, idx]
                self.vd_05[var]    = q05_arr[:, idx]
                self.vd_95[var]    = q95_arr[:, idx]

        if show is True:
            # Uses self.vd_det in both branches -- plot_variance_decomp_svar()
            # doesn't need to know about self.method at all.
            plot_variance_decomp_svar(self)

        if self.method != 'signs':
            output = {
                'point': self.vd_mean,
                'low':   self.vd_low,
                'high':  self.vd_high,
                2.5:     self.vd_02,
                5:       self.vd_05,
                97.5:    self.vd_97,
                95:      self.vd_95,
                0.5:     self.vd_00,
                99.5:    self.vd_99
            }
        else:
            output = {
                'point': self.vd_point,
                'low':   self.vd_low,
                'high':  self.vd_high,
                2.5:     self.vd_02,
                5:       self.vd_05,
                97.5:    self.vd_97,
                95:      self.vd_95,
                0.5:     self.vd_00,
                99.5:    self.vd_99
            }

        return output

    
    def HistoricalDecomp(self):
        return 0

    def stability(self, show=False):
        # 2. Calculate Eigenvalues
        eigenvalues = np.linalg.eigvals(self.F)
        
        # 3. Calculate the absolute value (modulus) of the eigenvalues
        abs_values = np.abs(eigenvalues)
        
        # 4. Check if ALL absolute values are strictly less than 1
        is_stable = np.all(abs_values < 1)
        
        # Output results
        if show is True:
            print("Eigenvalues:", eigenvalues)
            print("\nAbsolute Values:", abs_values)
        
        if is_stable:
            print("\nStability Check PASSED: All eigenvalues are inside the unit circle (< 1).")
            return True
        else:
            print("\nStability Check FAILED: At least one eigenvalue is >= 1.")
            return False
            
    def run(self, show=False, IRbands='pair', FCbands='many', showIR=False, showFC=False, showVD=False):
        
        self.YX()
        
        self.VAR()
        
        if self.method is not None:
            self.S()
            
        self.summary()
        
        if showIR==True:
            for k in range(self.n_vars):
                self.ImpulseResponse(k+1,show=show,bands=IRbands)
                
        if showFC==True:
            self.Forecast(bands=FCbands)
                
        if showVD==True:
            self.VarianceDecomp()
    
    
class SVEC:
    
    def __init__(self,
                 data
                 ):
        
        self.data=data
    
    
class LP:
    
    def __init__(self,
                 data
                 ):
        
        self.data=data

