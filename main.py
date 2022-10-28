import data_generator
import train_predict
import sys
import myModels
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
train_path = data_dir + '/train'
test_path = data_dir + '/test'
cleaned_dataset = data_generator.DataGenerator(train_path)
X, y = cleaned_dataset.X, cleaned_dataset.y
X_train, X_val, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = seed)
# TODO: tune the 0.2
