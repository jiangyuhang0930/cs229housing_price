import mymodels
import augment
import plot_confusion_matrix
import numpy as np
import tensorflow as tf
import sys
from keras.optimizers import SGD, Adam

""" This file is used when one already has the model fitted and saved to .h5 file. It generates
    ONLY a confusion matrix plot. There're three user provided arguments: existing .h5 file path,
    path to save the plots, and an optional third argument. If the optional third argument is given,
    then perform the evulation on test set. Otherwise (by default), perform on validation set.
"""

IMAGE_SIZE = [176,208]
BATCH_SIZE = 32
CLASS_LIST  = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
METRICS = [tf.keras.metrics.AUC(name='auc'), "acc"]

if len(sys.argv) < 3:
    print("Usage: ")
    print("   $python3 evaluate_and_plot.py  <model.h5> <image_path.png> [Bool:perform on test]")
model_path = sys.argv[1]
save_path = sys.argv[2]
on_test = False
if len(sys.argv) > 3:
    on_test = True

# Load all parameters from existing model
model = tf.keras.models.load_model(model_path)
sgd = SGD(learning_rate=0.0027, decay=1e-6, momentum=0.95, nesterov=True)
adam = Adam()
model.compile(optimizer=sgd,
                        loss=tf.losses.CategoricalCrossentropy(), 
                        metrics=METRICS)

if on_test:
    # get class predictions on test set
    y_prob = model.predict(augment.test_images)
    y_pred = y_prob.argmax(axis=-1)

    # get actual classes
    y_actual = np.argmax(augment.test_labels, axis=-1)

    # plot training metrics
    plot_confusion_matrix.plot_confusion_matrix(model,augment.test_images,augment.test_labels,y_actual,y_pred,['mild','moderate','normal','very-mild'],save_path)

else:
    # get class predictions on validation set
    y_prob = model.predict(augment.valid_images)
    y_pred = y_prob.argmax(axis=-1)

    # get actual classes
    y_actual = np.argmax(augment.valid_labels, axis=-1)

    # plot training metrics
    plot_confusion_matrix.plot_confusion_matrix(model,augment.valid_images,augment.valid_labels,y_actual,y_pred,['mild','moderate','normal','very-mild'],save_path)

