from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# The variables specified below ar optimal values after tuning

def get_linear_model():
    return LinearRegression()

def get_ridge_model(alpha = 500):
    return Ridge(alpha = alpha)

def get_lasso(alpha = 550):
    return Lasso(alpha = alpha)

def get_random_forest(n_estimators = 500, max_depth = 100):
    return RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth)

def get_bagging(n_estimators = 100):
    return BaggingRegressor(n_estimators = n_estimators)

def get_adaboost(alpha = 0.1, n_estimators = 300, loss='exponential'):
    return AdaBoostRegressor(learning_rate=alpha, n_estimators=n_estimators, loss=loss)

def get_gradient_boost():
    return GradientBoostingRegressor()

def get_mlp_regressor():
    return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter= 2000, alpha=0.1, learning_rate="adaptive")
