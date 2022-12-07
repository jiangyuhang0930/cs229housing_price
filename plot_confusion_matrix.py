from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Inspired from https://www.kaggle.com/code/kenconstable/alzheimer-s-multi-class-classification/notebook
def plot_training_metrics(train_hist,model,test_gen,test_label,y_actual,y_pred,classes,save_path):
    """
    Input: trained model history, model, test image generator, actual and predicted labels, 
    class list (to be shown on axes), and path to save the plot
    Output: Plots loss vs epochs, accuracy vs epochs, confusion matrix, written to save_path

    Notice that the difference between test_label and y_actual is that label is the one-hot
    representation (eg. [0,1,0,0]), where y_actual is a number {0,1,2,3} outputed by the softmax 
    layer. In general, y_actual = np.argmax(test_labels, axis=-1)
    """
    
    # Evaluate the results:
    test_loss, test_metric, *anythingelse = model.evaluate(test_gen,test_label,verbose = False)
    results       = round(test_metric,2)*100 
    results_title ="\n Model AUC on Test Data:{}%".format(results)
    print(results_title.format(results))
    print(len(results_title) * "-")
    
    # print classification report
    print(classification_report(y_actual, y_pred, target_names=classes))

    f1 = round(f1_score(y_actual, y_pred, average="weighted"),2)*100

    # extract data from training history for plotting
    history_dict    = train_hist.history
    loss_values     = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    auc_values      = history_dict['auc']
    val_auc_values  = history_dict['val_auc']
    epochs          = range(1, len(history_dict['auc']) + 1)

    # get the min loss and max accuracy for plotting
    max_auc = np.max(val_auc_values)
    min_loss = np.min(val_loss_values)
    
    # create plots
    plt.subplots(figsize=(12,4))
    
    # plot loss by epochs
    plt.subplot(1,3,1)
    plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
    plt.plot(epochs, val_loss_values, 'cornflowerblue', label = 'Validation loss')
    plt.title('Validation Loss by Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.axhline(y=min_loss,color='darkslategray', linestyle='--')
    plt.legend()

    # plot accuracy by epochs
    plt.subplot(1,3,2)
    plt.plot(epochs, auc_values, 'bo',label = 'Training AUC')
    plt.plot(epochs, val_auc_values, 'cornflowerblue', label = 'Validation AUC')
    # plt.plot(epochs,[results/100]*len(epochs),'darkmagenta',linestyle = '--',label='Test AUC')
    plt.title('Validation AUC by Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.axhline(y=max_auc,color='darkslategray', linestyle='--')
    plt.legend()
    
    # calculate Confusion Matrix
    cm = confusion_matrix(y_actual, y_pred)
    # create confusion matrix plot
    plt.subplot(1,3,3)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.BuPu)
    plt.title("Confusion Matrix \n F1 score:{}% \n AUC: {}%".format(f1, results))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # loop through matrix, plot each 
    threshold = cm.max() / 2.
    for r, c in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(c, r, format(cm[r, c], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[r, c] > threshold else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)

def plot_confusion_matrix(model,test_gen,test_label,y_actual,y_pred,classes,save_path):
    ''' Plot only the confusion matrix: this is for evaluation on validation
    or test set. Write to save_path.

    Notice: this function does not need history as an input.
    '''
    test_loss, test_metric, *anythingelse = model.evaluate(test_gen,test_label,verbose = False)
    results       = round(test_metric,2)*100
    f1 = round(f1_score(y_actual, y_pred, average="weighted"),2)*100
    # calculate Confusion Matrix
    cm = confusion_matrix(y_actual, y_pred)
    # create confusion matrix plot
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.BuPu)
    plt.title("Confusion Matrix \n F1 score:{}% \n AUC: {}%".format(f1, results))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # loop through matrix, plot each 
    threshold = cm.max() / 2.
    for r, c in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(c, r, format(cm[r, c], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[r, c] > threshold else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
