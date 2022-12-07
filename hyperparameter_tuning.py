import mymodels
import augment
import plot_confusion_matrix
import sys
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt

""" This file is used for hyperparameter tuning, before one has had any model fitted.
    In general, it takes no input arguments, perform hyperparameter tuning on a model
    the user chooses to tune, and write the optimal hyperparameter values to a txt file.
    Here an example is given to tune the ResNet-50 model with Adam solver.
"""

EPOCHS = 20
early_stopping = EarlyStopping(monitor = 'val_acc',patience = 5,restore_best_weights=True)
# Example of tuning ResNet 50 with adam optimizer: 
tuner = kt.RandomSearch(mymodels.tune_resnet50_adam,  # Change this when one needs to tune a new model
                     objective='val_acc',
                     overwrite = False,
                     max_trials = 1)
tuner.search(augment.train_images_aug, augment.train_labels_aug, epochs = 15, 
validation_data = (augment.valid_images, augment.valid_labels))
# Output the best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print(best_hp.values)
# Write the dictionary to a .txt file
# Remember to change this file name whenever the tuner is applied on a new model
with open("hpvalues_resnet_adam.txt", 'w') as f: 
    for key, value in best_hp.values.items(): 
        f.write('%s:%s\n' % (key, value))

