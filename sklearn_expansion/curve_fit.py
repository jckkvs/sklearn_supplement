from inspect import signature
import optuna 
from optuna.trial import Trial
import scipy
from sklearn.metrics import mean_squared_error
warnings.simplefilter('ignore')

def curve_fit_optuna(f, xdata , ydata, p0=None):    
    def objective(trial):
        sig = signature(f)
        parameters = sig.parameters
        optimize_parameters = list(parameters.keys())[1:]
        p0 = [trial.suggest_float(f"p0_{i}", -10**2, 10**2) for i in range(len(optimize_parameters))]
        try:
            a, b = curve_fit(f, xdata, ydata, method="dogbox", p0=p0)
            y_predict = f(xdata, *a)
            mse = mean_squared_error(ydata, y_predict)
        except:
            mse = np.inf        
        return mse

    sig = signature(func)
    parameters = sig.parameters
    optimize_parameters = list(parameters.keys())[1:]    
    study = optuna.create_study()
    optuna.logging.set_verbosity(100)
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    best_p0 = list(best_params.values())
    best_mse = np.inf
    best_a, best_b = None, None
    best_idx = 0

    max_trial = 1000
    patience = 0
    for cnt in range(max_trial):        
        try:
            p0 = [float(scipy.stats.norm.rvs(scale=(abs(best_p0[i])**0.5) * (1 - cnt/max_trial) * 30, loc=best_p0[i], size=1)) for i in range(len(optimize_parameters))]
            a, b = curve_fit(f, xdata, ydata, method="dogbox", p0=p0)    
            y_predict = f(xdata, *a)
            mse = mean_squared_error(ydata, y_predict)
        except:
            mse = np.inf

        if mse < best_mse:
            best_mse = mse
            best_a = a
            best_b = b
            best_idx = cnt
            patience = 0
        else:
            patience += 1
            if patience >= max_trial/3:
                break

    print(best_mse)
    print(best_idx)
            
    return best_a, best_b