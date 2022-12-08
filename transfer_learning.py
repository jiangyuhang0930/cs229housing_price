import pandas as pd
import numpy as np
import data_generator
import train_predict
from sklearn.preprocessing import MinMaxScaler
import my_models
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")

# Read in the melb and ames dataset
melb_features = pd.read_csv("melb_features.csv")
ames_features = pd.read_csv("ames_features.csv")
melb_target = pd.read_csv("melb_target.csv").squeeze()
ames_target = pd.read_csv("ames_target.csv").squeeze()
ames_X_train, ames_X_test, ames_y_train, ames_y_test = train_test_split(
            ames_features, ames_target, test_size=0.2)

# Number of trials to run
NUM_TRIALS = 20

# Create data generator object and use functions in data_generator.py to read in the dataset
melb_raw = data_generator.DataGenerator("melb_data")  # melb whole dataset used for no transfer learning

# Arrays to store model evaluation metrics
no_transfer, lin1, lin2, mlp, gb1, gb2, rf1, rf2 = ([0,0,0] for i in range(8))

for i in range(NUM_TRIALS):
    print("======= Running Iteration {} =======".format(i+1))

    # Choose random index each time to mimic small dataset for transfer learning settings
    idx = np.arange(6000)
    np.random.shuffle(idx)
    rand_idx = idx[:200]

    # Without transfer learning part
    normal_gradient_boost = my_models.get_gradient_boost()
    X, y = melb_raw.X.iloc[rand_idx,:], melb_raw.y[rand_idx]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_predict.train_model(normal_gradient_boost, X_train, y_train)
    no_transfer_pred = train_predict.predict(normal_gradient_boost, X_val, y_val)
    for j in range(3):
        no_transfer[j] += no_transfer_pred[j] / NUM_TRIALS

    # Selecting only a small portion of melb data for transfer learning
    melb_rand_features = melb_features.iloc[rand_idx,:]
    melb_rand_target = melb_target[rand_idx]

    melb_X_train, melb_X_test, melb_y_train, melb_y_test = train_test_split(
            melb_rand_features, melb_rand_target, test_size=0.2)

    # Build, pretrain, and stack the 5 models
    model1 = my_models.get_gradient_boost()
    model1.fit(ames_X_train, ames_y_train)

    model2 = my_models.get_lasso()
    model3 = my_models.get_random_forest()
    model4 = my_models.get_ridge_model()
    model5 = my_models.get_bagging()
    models = [model1, model2, model3, model4, model5]

    # Collect the predictions given by the 5 models as input data for the later stage
    predictions_on_train = []
    for model in models:
        model.fit(ames_X_train, ames_y_train)
        predictions_on_train.append(model.predict(melb_X_train))
    predictions_on_train = np.reshape(predictions_on_train, (-1,5))

    prediction_on_test = []
    for model in models:
        prediction_on_test.append(model.predict(melb_X_test))
    prediction_on_test = np.reshape(prediction_on_test, (-1,5))

    pred_and_raw = np.hstack((predictions_on_train, melb_X_train))
    test_and_raw = np.hstack((prediction_on_test, melb_X_test))

    # Training final model T
    linear_model = my_models.get_linear_model()
    linear_model.fit(pred_and_raw, melb_y_train)
    lin_pred_1 = train_predict.predict(linear_model, test_and_raw, melb_y_test)
    for j in range(3):
        lin1[j] += lin_pred_1[j] / NUM_TRIALS

    lin_raw = my_models.get_linear_model()
    lin_raw.fit(melb_X_train, melb_y_train)
    lin_pred_2 = train_predict.predict(lin_raw, melb_X_test, melb_y_test)
    for j in range(3):
        lin2[j] += lin_pred_2[j] / NUM_TRIALS

    mlp_model = MLPRegressor(hidden_layer_sizes=(64, 16), max_iter= 20000, verbose = 0, alpha=1e-1, learning_rate="constant")
    mlp_model.fit(pred_and_raw, melb_y_train)
    mlp_pred = train_predict.predict(mlp_model, test_and_raw, melb_y_test)
    for j in range(3):
        mlp[j] += mlp_pred[j] / NUM_TRIALS

    # Using 10+5 features
    gb_model = my_models.get_gradient_boost()
    gb_model.fit(pred_and_raw, melb_y_train)
    gb_pred_1 = train_predict.predict(gb_model, test_and_raw, melb_y_test)
    for j in range(3):
        gb1[j] += gb_pred_1[j] / NUM_TRIALS

    # Using only 10 features
    gb_model2 = my_models.get_gradient_boost()
    gb_model2.fit(melb_X_train, melb_y_train)
    gb_pred_2 = train_predict.predict(gb_model2, melb_X_test, melb_y_test)
    for j in range(3):
        gb2[j] += gb_pred_2[j] / NUM_TRIALS

    rf_model = my_models.get_random_forest()
    rf_model.fit(pred_and_raw, melb_y_train)
    rf_pred_1 = train_predict.predict(rf_model, test_and_raw, melb_y_test)
    for j in range(3):
        rf1[j] += rf_pred_1[j] / NUM_TRIALS

    rf_model2 = my_models.get_random_forest()
    rf_model2.fit(melb_X_train, melb_y_train)
    rf_pred_2 = train_predict.predict(rf_model2, melb_X_test, melb_y_test)
    for j in range(3):
        rf2[j] += rf_pred_2[j] / NUM_TRIALS

print("Using gradient boost without transfer learning on the whole dataset of 200 instances:")
print(no_transfer[0], no_transfer[1], no_transfer[2])
print("Using linear model on 10+5 features:")
print(lin1[0], lin1[1], lin1[2])
print("Using linear model on only 10 features:")
print(lin2[0], lin2[1], lin2[2])
print("Using 10+5 on mlp:")
print(mlp[0], mlp[1], mlp[2])
print("Using gradient boost on 10+5 features:")
print(gb1[0], gb1[1], gb1[2])
print("Using gradient boost on only 10 features:")
print(gb2[0], gb2[1], gb2[2])
print("Using random forest on 10+5 features:")
print(rf1[0], rf1[1], rf1[2])
print("Using random forest on only 10 features:")
print(rf2[0], rf2[1], rf2[2])


################################################################################################
# Below is used earlier for hyperparameter tuning. These are commented out after tuning finished

# param_grid = {'hidden_layer_sizes': [(1024, 256), (512, 128), (256, 64)],   
#               'alpha': [1e-5, 1e-4, 1e-3]}  
# grid = GridSearchCV(mlp_model, param_grid, refit = True, verbose = 1)
# grid.fit(pred_and_raw, melb_y_train)
# print(grid.best_params_) 
# print(predict(grid, test_and_raw, melb_y_test))

# rf_model3 = my_models.get_random_forest()
# param_grid = {'max_depth': [int(x) for x in np.linspace(10, 100, num = 10)],
# 'min_samples_split': [2,5,10], "n_estimators": [int(x) for x in np.linspace(100, 2100, num = 10)],
# 'min_samples_leaf': [1,2]}
# grid = RandomizedSearchCV(rf_model3, param_grid, verbose = 1)
# grid.fit(pred_and_raw, melb_y_train)
# print("Best parameters are:")
# print(grid.best_params_) 
# print("Using tuned model on 10+5 features:")
# print(predict(grid, test_and_raw, melb_y_test))

