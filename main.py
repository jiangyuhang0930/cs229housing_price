import data_generator
import train_predict
import sys
import my_models
import feature_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

try:
    data_dir = sys.argv[1]
except RuntimeError as e:
    print('ERROR: {}'.format(e))
    sys.exit(2)

seed = 1999
np.random.seed(seed)
data_path = data_dir + '/ames'
cleaned_dataset = data_generator.DataGenerator(data_path=data_path)
X, y = cleaned_dataset.X, cleaned_dataset.y
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.25, random_state = seed)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size = 0.4, random_state = seed)

# hyperparameter tuning log
# alpha values for ridge regression 
# parameter_list = [0.1, 1, 2, 5, 10]

# alpha values for lasso regression 
# parameter_list = [10, 20, 30, 40, 50]

# max_depth values for random forest
# parameter_list = [10, 50, 100, 500, None]
# n_estimator values for random forest
# parameter_list = [100, 200, 500, 1000]

# learning_rate values for adaboost
# parameter_list = [0.1, 0.2, 0.5, 1, 2]
# n_estimator values for adaboost
# parameter_list = [50, 100, 200, 500, 1000]

# learning_rate values for gradient boost
# parameter_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
# n_estimator values for gradient boost
# parameter_list = [50, 100, 200, 500, 1000]

# layer_size for MLP
# parameter_list = [(32, 32), (64, 32), (64, 64)]
# learning_rate values for MLP
# parameter_list = [0.1, 0.2, 0.5, 1]
# for parameter in parameter_list:
#     print(parameter)
model = my_models.get_gradient_boost()
train_predict.train_model(model, X_train, y_train)
print(train_predict.predict(model, X_val, y_val))
# print(train_predict.predict(model, X_test, y_test))

# X_train_selected, selector = feature_selection.f_selection(X_train, y_train, k)
# X_val_selected = selector.transform(X_val)
# model = my_models.get_linear_model()
# train_predict.train_model(model, X_train_selected, y_train)
# print(train_predict.predict(model, X_val_selected, y_val))