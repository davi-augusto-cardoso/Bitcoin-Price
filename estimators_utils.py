from ml_utils import *

from skopt import space

best_estimators = [ElasticNet(), BayesianRidge(), ARDRegression(),
                   HuberRegressor(), QuantileRegressor(solver="highs"), RANSACRegressor(random_state=1),
                   TheilSenRegressor(random_state=1), ]


all_linears = [ElasticNet(), BayesianRidge(), 
               ARDRegression(), Lars(random_state=1), 
               LassoLars(random_state=1), SGDRegressor(random_state=1), PassiveAggressiveRegressor(), 
               HuberRegressor(), QuantileRegressor(solver="highs"),
               RANSACRegressor(random_state=1), TheilSenRegressor(random_state=1)]

all_robust_linear = [HuberRegressor(), QuantileRegressor(solver="highs"), 
                     RANSACRegressor(random_state=1), TheilSenRegressor(random_state=1)]

all_estimator = [SVR(), NuSVR(), LinearSVR(random_state=1), 
                  ElasticNet(), BayesianRidge(), ARDRegression(), PassiveAggressiveRegressor(random_state=1),
                  GaussianProcessRegressor(),
                  DecisionTreeRegressor(), 
                  RandomForestRegressor(random_state=1), AdaBoostRegressor(), GradientBoostingRegressor(random_state=1),
                  HuberRegressor(), QuantileRegressor(solver="highs"), RANSACRegressor(random_state=1), TheilSenRegressor(random_state=1), 
                  KNeighborsRegressor()]

hp_baysianridge_names = [ "n_iter", "tol", "alpha_1", "alpha_2", 
                         "lambda_1", "lambda_2",]

hp_baysianridge = [
    space.Integer(100, 3000, name="n_iter"),
    space.Real(1e-6, 1e-1, name="tol"),
    space.Real(1e-9, 1e-3, name="alpha_1"),
    space.Real(1e-9, 1e-3, name="alpha_2"),
    space.Real(1e-9, 1e-3, name="lambda_1"),
    space.Real(1e-9, 1e-3, name="lambda_2"),]


Estimators = {
    "Neural Networks" : list([dict({
        "Estimator" : MLPRegressor(),
        "Params"    : dict({
            'activation'        :   space.Categorical(['identity', 'logistic', 'tanh', 'relu'], name = 'activation'),
            'alpha'             :   space.Real(1e-9, 1e-3, name='alpha'),
            'hidden_layer_sizes':   space.Integer(300, 800, name = 'hidden_layer_sizes'),
            'max_iter'          :   space.Integer(300, 1200, name='max_iter'),
            'solver'            :   space.Categorical(['lbfgs','adam'], name='solver'),
            'tol'               :   space.Real(1e-8, 1e-3, name = 'tol'),
            'beta_1'            :   space.Real(0.1, 0.99, name='beta_1'),
            'beta_2'            :   space.Real(0.01, 0.99, name='beta_2'),
            'random_state'      :   space.Integer(0, 1, name = 'random_state'),
        })
    }),])
}
