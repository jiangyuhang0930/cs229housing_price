from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

# The variables specified below ar optimal values after tuning
def get_linear_model():
    return LinearRegression()

def get_ridge_model(alpha = 2):
    return Ridge(alpha = alpha)

def get_lasso_model(alpha = 40):
    return Lasso(alpha = alpha)

def get_random_forest(n_estimators = 1000, max_depth = None):
    return RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth)

def get_adaboost(alpha = 1, n_estimators = 1000):
    return AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=None), learning_rate=alpha, n_estimators=n_estimators)

def get_gradient_boost(alpha=0.3, n_estimators=100):
    return GradientBoostingRegressor(learning_rate=alpha, n_estimators=n_estimators)

def get_mlp_regressor(layer_sizes=(64, 64), alpha=0.5):
    return MLPRegressor(hidden_layer_sizes=layer_sizes, max_iter= 5000, alpha=alpha)
