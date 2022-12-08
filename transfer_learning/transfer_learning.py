import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import my_models
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")

melb_features = pd.read_csv("melb_features.csv")
ames_features = pd.read_csv("ames_features.csv")
melb_target = pd.read_csv("melb_target.csv").squeeze()
ames_target = pd.read_csv("ames_target.csv").squeeze()

idx = np.arange(6000)
np.random.shuffle(idx)
rand_idx = idx[:500]
# melb_features = melb_features.iloc[rand_idx,:]
# melb_target = melb_target[rand_idx]

ames_X_train, ames_X_test, ames_y_train, ames_y_test = train_test_split(
        ames_features, ames_target, test_size=0.2)
melb_X_train, melb_X_test, melb_y_train, melb_y_test = train_test_split(
        melb_features, melb_target, test_size=0.2)

def evaluate_relative_error(y_pred, y):
    return np.mean(np.abs(y-y_pred) / y_pred)

def evaluate_mse(y_pred, y):
    return np.mean((y-y_pred)**2) / 1e6

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def predict(model, X_test, y_test):
    y_pred = model.predict(X_test)
    R2 = model.score(X_test, y_test)
    rel_error = evaluate_relative_error(y_pred, y_test)
    mse = evaluate_mse(y_pred, y_test)
    return R2, rel_error, mse

model1 = my_models.get_gradient_boost()
model1.fit(ames_X_train, ames_y_train)

model2 = my_models.get_lasso()
model3 = my_models.get_random_forest()
model4 = my_models.get_ridge_model()
model5 = my_models.get_bagging()
models = [model1, model2, model3, model4, model5]

predictions_on_train = []
for model in models:
    model.fit(ames_X_train, ames_y_train)
    predictions_on_train.append(model.predict(melb_X_train))
predictions = np.reshape(predictions_on_train, (-1,5))

prediction_on_test = []
for model in models:
    prediction_on_test.append(model.predict(melb_X_test))
prediction_on_test = np.reshape(prediction_on_test, (-1,5))

pred_and_raw = np.hstack((predictions, melb_X_train))
# Goal: fit a model on pred_and_raw better than 31% error rate
test_and_raw = np.hstack((prediction_on_test, melb_X_test))

# linear_model = my_models.get_linear_model()
# linear_model.fit(pred_and_raw, melb_y_train)


# lin_raw = my_models.get_linear_model()
# lin_raw.fit(melb_X_train, melb_y_train)


# mlp_model = MLPRegressor(hidden_layer_sizes=(64, 16), max_iter= 20000, verbose = 0, alpha=1e-1, learning_rate="constant")
# mlp_model.fit(pred_and_raw, melb_y_train)

# # Using 10+5 features
# gb_model = my_models.get_gradient_boost()
# gb_model.fit(pred_and_raw, melb_y_train)

# # Using only 10 features
# gb_model2 = my_models.get_gradient_boost()
# gb_model2.fit(melb_X_train, melb_y_train)


rf_model = my_models.get_random_forest()
rf_model.fit(pred_and_raw, melb_y_train)

rf_model2 = my_models.get_random_forest()
rf_model2.fit(melb_X_train, melb_y_train)

# print("Using linear model on 10+5 features:")
# print(predict(linear_model, test_and_raw, melb_y_test))
# print("Using linear model on only 10 features:")
# print(predict(lin_raw, melb_X_test, melb_y_test))
# # print("Using 10+5 on mlp:")
# # print(predict(mlp_model, test_and_raw, melb_y_test))
# print("Using gradient boost on 10+5 features:")
# print(predict(gb_model, test_and_raw, melb_y_test))
# print("Using gradient boost on only 10 features:")
# print(predict(gb_model2, melb_X_test, melb_y_test))
print("Using random forest on 10+5 features:")
print(predict(rf_model, test_and_raw, melb_y_test))
print("Using random forest on only 10 features:")
print(predict(rf_model2, melb_X_test, melb_y_test))

# param_grid = {'hidden_layer_sizes': [(1024, 256), (512, 128), (256, 64)],   
#               'alpha': [1e-5, 1e-4, 1e-3]}  
# grid = GridSearchCV(mlp_model, param_grid, refit = True, verbose = 1)
# grid.fit(pred_and_raw, melb_y_train)
# print(grid.best_params_) 
# print(predict(grid, test_and_raw, melb_y_test))

rf_model3 = my_models.get_random_forest()
param_grid = {'max_depth': [int(x) for x in np.linspace(10, 100, num = 10)],
'min_samples_split': [2,5,10], "n_estimators": [int(x) for x in np.linspace(100, 2100, num = 10)],
'min_samples_leaf': [1,2]}
grid = RandomizedSearchCV(rf_model3, param_grid, verbose = 1)
grid.fit(pred_and_raw, melb_y_train)
print("Best parameters are:")
print(grid.best_params_) 
print("Using tuned model on 10+5 features:")
print(predict(grid, test_and_raw, melb_y_test))

