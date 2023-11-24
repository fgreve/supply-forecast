
# Parámetros que se ocupan para simular 1 pn (-pn1)
params = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 1,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': False, 
    'min_winsor': 5, 
    'max_winsor': 90, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'  
}


# CV=False : Simulación con params / CV=True : Simulación con Cross Validation usando model_type
CV=False

# Modelos a simular en Cross Validation (-cv)
model_type = {'linear_log': ('linear', False, False, True, False), 
              'linear_window': ('linear', False, True, False, False),
              'linear_winsor': ('linear', True, False, False, False), 
              'linear_tukey': ('linear', False, False, False, True),
              'flat_log': ('flat', False, False, True, False), 
              'flat_window': ('flat', False, True, False, False),
              'flat_winsor': ('flat', True, False, False, False), 
              'flat_tukey': ('flat', False, False, False, True)
             }

# Parámetros a simular según la rotación del PN.
# Al sacar outliers se puede caer en errores, por lo que se van intentanto distintos parmétros:
# Para el caso de FM


# Paramétros definidos por rotación: MPARAMS
params_fm1 = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 3,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': True, 
    'min_winsor': 0, 
    'max_winsor': 80, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'  
}
params_fm2 = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 3,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': False, 
    'min_winsor': 0, 
    'max_winsor': 90, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'  
}
params_fm3 = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 3,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': False, 
    'min_winsor': 0, 
    'max_winsor': 90, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'  
}

params_mm1 = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 3,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': True, 
    'min_winsor': 0, 
    'max_winsor': 80, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'  
}
params_mm2 = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 3,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': False, 
    'min_winsor': 0, 
    'max_winsor': 90, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'  
}
params_mm3 = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 3,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': False, 
    'min_winsor': 0, 
    'max_winsor': 90, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'  
}

params_sm1 = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 3,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': True, 
    'min_winsor': 0, 
    'max_winsor': 80, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'  
}
params_sm2 = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 3,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': False, 
    'min_winsor': 0, 
    'max_winsor': 90, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'  
}
params_sm3 = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 3,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': False, 
    'min_winsor': 0, 
    'max_winsor': 90, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'   
}

params_nm1 = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 3,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': True, 
    'min_winsor': 0, 
    'max_winsor': 80, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'   
}
params_nm2 = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 3,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': False, 
    'min_winsor': 0, 
    'max_winsor': 90, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'   
}
params_nm3 = { 
    'growth': 'flat', 
    'first_date': 'all',                          
    'interval_width': 0.95, 
    'changepoint_range': 0.95, 
    'changepoint_prior_scale': 2.0, 
    'seasonality_mode': 'additive',
    'hyperparameter_tuning': False, 
    'cross_vals': 3,
    'drop_outliers_windows': False, 
    'window': 12,
    'drop_outliers_winsorization': False, 
    'min_winsor': 0, 
    'max_winsor': 90, 
    'add_regressor_FH': False, 
    'output': 'prediction', 
    'n_changepoints': 'auto'  
}

parametros = {
    'FM': {
        1: params_fm1,
        2: params_fm2,
        3: params_fm3,
    },
    'MM': {
        1: params_mm1,
        2: params_mm2,
        3: params_mm3
    },  
    'SM': {
        1: params_sm1,
        2: params_sm2,
        3: params_sm3
    },
    'NM': {
        1: params_nm1,
        2: params_nm2,
        3: params_nm3
    }
}
