from estimators_utils import *

best_estimators = [ElasticNet(), BayesianRidge(), ARDRegression(),
                   HuberRegressor(), QuantileRegressor(solver="highs"), RANSACRegressor(random_state=1),
                   TheilSenRegressor(random_state=1), ]

# ------------------ BaianRidge -------------------

names_BayesianRidge = [  "n_iter", "tol", "alpha_1", "alpha_2", 
                        "lambda_1", "lambda_2",]
space_BayesianRidge = [
    space.Integer(100, 3000, name="n_iter"),
    space.Real(1e-6, 1e-1, name="tol"),
    space.Real(1e-9, 1e-3, name="alpha_1"),
    space.Real(1e-9, 1e-3, name="alpha_2"),
    space.Real(1e-9, 1e-3, name="lambda_1"),
    space.Real(1e-9, 1e-3, name="lambda_2"),]

# ------------------ ElasticNet -------------------
names_ElasticNet = ['alpha', 'l1_ratio', 'max_iter', 
                    'random_state', 'selection', 'tol']
space_ElasticNet = [
    space.Real(1e-1, 1, name="alpha"),
    space.Real(1e-1, 1, name="l1_ratio"),
    space.Integer(1000, 3000, name="max_iter"),
    space.Integer(1, 2, name="random_state"),
    space.Categorical(['cyclic', 'random'], name='selection'),
    space.Real(1e-6, 1e-1, name="tol"),]

# ------------------ ARDRegression -------------------

names_ARDRegression = [ 'alpha_1', 'alpha_2', 'lambda_1','lambda_2',
                        'n_iter','threshold_lambda', 'tol']

space_ARDRegression = [
    space.Real(1e-7, 1, name="alpha_1"),
    space.Real(1e-7, 1, name="alpha_2"),
    space.Real(1e-7, 1, name="lambda_1"),
    space.Real(1e-7, 1, name="lambda_2"),
    space.Integer(300, 450, name="n_iter "),
    space.Real(9500.0, 10500.0, name="threshold_lambda"),
    space.Real(1e-5, 1e-1, name="tol"),]

# ------------------ HuberRegressor -------------------

names_HuberRegressor = [ 'alpha', 'epsilon', 'fit_intercept','max_iter',
                        'tol','warm_start']

space_HuberRegressor = [
    space.Real(1e-6, 1, name="alpha"),
    space.Real(1, 2, name="epsilon"),
    space.Categorical([True, False], name="fit_intercept"),
    space.Integer(100, 175, name="max_iter"),
    space.Real(1e-6, 1, name="tol"),
    space.Categorical([True, False], name="warm_start"),]

# ------------------ QuantileRegressor -------------------

names_QuantileRegressor = []

space_QuantileRegressor = []

# ------------------ RANSACRegressor -------------------

names_RANSACRegressor = []

space_RANSACRegressor = []

e = RANSACRegressor().get_params(deep=False)

print(e.keys())
print(e.values())
print(1e-1)


estimators_hp = [
    {   
        'Estimator'     : BayesianRidge,
        'Params_names'  : names_BayesianRidge,
        'Space'         : space_BayesianRidge},
    {   
        'Estimator'     : ElasticNet,
        'Params_names'  : names_ElasticNet,
        'Space'         : space_ElasticNet},
    {   
        'Estimator'     : ARDRegression,
        'Params_names'  : names_ARDRegression,
        'Space'         : space_ARDRegression},
    {   
        'Estimator'     : HuberRegressor,
        'Params_names'  : names_HuberRegressor,
        'Space'         : space_HuberRegressor},
    # {   
    #     'Estimator'     : QuantileRegressor,
    #     'Params_names'  : names_QuantileRegressor,
    #     'Space'         : space_QuantileRegressor},
    ]
