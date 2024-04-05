from estimators_utils import *

best_estimators = [ElasticNet(), BayesianRidge(), ARDRegression(),
                   HuberRegressor(), QuantileRegressor(solver="highs"), RANSACRegressor(random_state=1),
                   TheilSenRegressor(random_state=1), ]

names_baysianridge = [ "n_iter", "tol", "alpha_1", "alpha_2", 
                        "lambda_1", "lambda_2",]

space_baysianridge = [
    space.Integer(100, 3000, name="n_iter"),
    space.Real(1e-6, 1e-1, name="tol"),
    space.Real(1e-9, 1e-3, name="alpha_1"),
    space.Real(1e-9, 1e-3, name="alpha_2"),
    space.Real(1e-9, 1e-3, name="lambda_1"),
    space.Real(1e-9, 1e-3, name="lambda_2"),]

names_elasticnet = ['alpha', 'l1_ratio', 'max_iter', 
                    'random_state', 'selection', 'tol']
space_elasticnet = [
    space.Real(1e-1, 1, name="alpha"),
    space.Real(1e-1, 1, name="l1_ratio"),
    space.Integer(1000, 3000, name="max_iter"),
    space.Integer(1, 2, name="random_state"),
    space.Categorical(['cyclic', 'random'], name='selection'),
    space.Real(1e-6, 1e-1, name="tol"),]


e = ElasticNet().get_params(deep=False)

print(e.keys())
print(e.values())
print(1e-1)



estimators_hp = [
    {   
        'Estimator'     : BayesianRidge,
        'Params_names'  : names_baysianridge,
        'Space'         : space_baysianridge},
    {   
        'Estimator'     : ElasticNet,
        'Params_names'  : names_elasticnet,
        'Space'         : space_elasticnet},
    ]
