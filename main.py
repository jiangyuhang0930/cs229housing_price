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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = seed)
# TODO: tune the 0.2
for k in [10, 20, 30, 40, 50, 60, 70, 80]:
    print(k)
    X_train_selected, selector = feature_selection.f_selection(X_train, y_train, k)
    X_val_selected = selector.transform(X_val)
    model = my_models.get_gradient_boost()
    train_predict.train_model(model, X_train_selected, y_train)
    print(train_predict.predict(model, X_val_selected, y_val))