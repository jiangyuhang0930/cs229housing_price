import mymodels
import augment
import plot_confusion_matrix
import numpy as np
import tensorflow as tf
import sys
from keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

""" This file is used only after hyperparameter training, when one wants to train a model and
    generate the training metric (eg. training loss over epoch). One must already have the hyperparameter
    values ready to fill in. This file generates a 3*1 plot consisting of validation & training
    loss over epochs, validation & training AUC over epochs, and one confusion matrix on validation
    set. Two arguments must be given to save the model and plot.

    An example is given below using the ResNet50-sgd model.
"""

IMAGE_SIZE = [176,208]
BATCH_SIZE = 32
CLASS_LIST  = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
METRICS = [tf.keras.metrics.AUC(name='auc'), "acc"]

if len(sys.argv) < 3:
    print("Usage: ")
    print("   $python3 evaluate_and_plot.py  <model.h5> <image_path.png>")
model_path = sys.argv[1]
save_path = sys.argv[2]

model = mymodels.build_resnet50_sgd()   # Change this when fitting a new model
EPOCHS         = 25
early_stopping = EarlyStopping(monitor = 'val_acc',patience = 5,restore_best_weights=True)
checkpoint_cb  = ModelCheckpoint(model_path, save_best_only=True)

# Fitting the model
history = model.fit(
    augment.train_images_aug, augment.train_labels_aug,
    epochs           = EPOCHS,
    validation_data  = (augment.valid_images, augment.valid_labels),
    verbose          = 1,
    callbacks        = [checkpoint_cb, early_stopping],
)

# get class predictions on validation set
y_prob = model.predict(augment.valid_images)
y_pred = y_prob.argmax(axis=-1)

# get actual classes
y_actual = np.argmax(augment.valid_labels, axis=-1)

# Plotting training metrics
plot_confusion_matrix.plot_training_metrics(history,model,augment.valid_images,augment.valid_labels,y_actual,y_pred,['mild','moderate','normal','very-mild'],save_path)
