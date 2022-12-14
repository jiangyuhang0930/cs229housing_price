import numpy as np
import my_models
from sklearn.model_selection import train_test_split
import feature_selection
import matplotlib.pyplot as plt
seed = 1999
np.random.seed(seed)
def evaluate_relative_error(y_pred, y):
    return np.mean(np.abs((y-y_pred) / y_pred))

def evaluate_mse(y_pred, y):
    return np.mean((y-y_pred)**2) * (1/1e6)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def predict(model, X_test, y_test):
    y_pred = model.predict(X_test)
    R2 = model.score(X_test, y_test)
    rel_error = evaluate_relative_error(y_pred, y_test)
    mse = evaluate_mse(y_pred, y_test)
    return R2, rel_error, mse

def split_data(X, y):
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.25, random_state = seed)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size = 0.4, random_state = seed) 
    return X_train, y_train, X_val, y_val, X_test, y_test

def tune_hyperparameter(parameter_list, X, y):
    for parameter in parameter_list:
        print(parameter)
        X_train, y_train, X_val, y_val, _, _ = split_data(X, y)
        model = my_models.get_mlp_regressor(layer_sizes=parameter)
        train_model(model, X_train, y_train)
        print("Validation Result: " + str(predict(model, X_val, y_val)))

def feature_selection_experiment(k_list, X, y, method='regular'):
    X_train, y_train, X_val, y_val, _, _ = split_data(X, y)  
    for k in k_list:
        print(k)
        X_reduced_train, selector = feature_selection.extra_tree_rfe_selection(X_train, y_train, k)
        X_reduced_val = selector.transform(X_val)
        model = my_models.get_gradient_boost()
        train_model(model, X_reduced_train, y_train)
        print("Validation Result: " + str(predict(model, X_reduced_val, y_val)))

def plot_feature_importance(X, y, k=10):
    X_train, y_train, _, _, _, _ = split_data(X, y)
    X_reduced_train, selector = feature_selection.extra_tree_rfe_selection(X_train, y_train, k)   
    chosen_column = ['Total area', 'First floor area', 'Garage area', 'Basement area', 'Basement area T1',
        'Year built', 'Overall quality', 'Garage capacity', 'External quality', 'Kitchen quality']
    model = my_models.get_extra_tree()
    train_model(model, X_reduced_train, y_train)
    print(model.feature_importances_)
    print(chosen_column)
    plt.figure(figsize=(11,5))
    plt.barh(chosen_column, sorted(model.feature_importances_), color ='maroon')
    plt.rc('ytick', labelsize=12) 
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance of Top 10 Featues Using Extra Tree')
    plt.savefig('importance.png')