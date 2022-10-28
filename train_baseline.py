from tkinter.ttk import Separator
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def load_iowa_data(data_path, write=False):
    X = pd.read_csv(data_path + '.csv', sep=',')
    # separator features from targets
    X, y = seperate_feature_target(X, 'SalePrice')

    # used to fill in empty values
    X, y = fillin_missing_values(X, y)
    X = str_to_categorical(X)
    if write:
        X['SalePrice'] = y
        X.to_csv(data_path + '_processed.csv')
    return X, y

def seperate_feature_target(X, target_name):
    y = X[target_name]
    X = X.drop(columns=[target_name])
    return X, y    

def fillin_missing_values(X, y):
    X_means = X.mean().round(0)    
    X_modes = X.mode()
    # some columns are numeric but we want to treat it as string 
    exception_columns = ['MSSubClass']
    for column in X:
        if pd.api.types.is_numeric_dtype(X[column]) and column not in exception_columns:
            X[column] = X[column].fillna(X_means[column])
        else:
            X[column] = X[column].fillna(X_modes[column])
    y = y.fillna(y.mean())
    return X, y

def str_to_categorical(X):
    for column in X:
        if not pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].astype('category')
    categorical_columns = X.select_dtypes(['category']).columns
    X[categorical_columns] = X[categorical_columns].apply(lambda x: x.cat.codes)
    return X

def evaluate_relative_error(y_pred, y):
    return np.mean(np.abs(y-y_pred) / y_pred)

def evaluate_mse(y_pred, y):
    return np.mean((y-y_pred)**2 / y_pred)

def load_bejing_data(data_path):
    pass
    ## TODO ##

def load_london_data(data_path):
    pass
    ## TODO ##

def load_boston_data(data_path):
    pass
    ## TODO ##

if __name__ == "__main__":
    seed = 1999
    np.random.seed(seed)

    train_path = './iowa_data/train'
    test_path = './iowa_data/test'

    X, y = load_iowa_data(data_path=train_path)
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    model = LinearRegression().fit(X_train, y_train)
    # this reports the R^2 coefficient
    y_pred = model.predict(X_test)
    R2 = model.score(X_test, y_test)
    rel_error = evaluate_relative_error(y_pred, y_test)
    mse = evaluate_mse(y_pred, y_test)
    print("R^2: " + str(R2))
    print("Average Relative Error: " + str(rel_error))
    print("MSE: " + str(mse))

