import numpy as np

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

def select_feature(X_train, selection_function):
    return selection_function(X_train)