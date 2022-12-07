import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

""" Adapted from https://github.com/Jacobheyman702/Alzheimer_Image_classifier-/blob/master/Image_analysis.ipynb

    Application of keras ImageDataGenerator() to perform image augmentation on the training set only
    of the two minor classes: moderate demented and mild demented.

    train_images_aug, valid_images, and test_images are generator objects that will
    be used in almost all other python files. train_labels, etc. are the labels in 
    one-hot representation. These are useful when plotting the confusion matrix.
"""

IMAGE_SIZE = [176,208]
BATCH_SIZE = 32
CLASS_LIST  = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
METRICS = [tf.keras.metrics.AUC(name='auc'), "acc"]

train_directory = "./Alzheimer_s Dataset/train"
# Training / Validation split
train_data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_data_generator = ImageDataGenerator(rescale=1./255)
test_directory = "./Alzheimer_s Dataset/test"
data_train= train_data_generator.flow_from_directory( 
        train_directory, 
        subset= 'training',
        batch_size = 5121, 
        target_size=IMAGE_SIZE, 
)

data_valid = train_data_generator.flow_from_directory( 
        train_directory, 
        subset= 'validation',
        batch_size = 1023, 
        target_size=IMAGE_SIZE, 
)

data_test= test_data_generator.flow_from_directory( 
        test_directory, 
        target_size=IMAGE_SIZE,
        batch_size = 1279,
        shuffle = False)

train_images, train_labels = next(data_train)

test_images, test_labels = next(data_test)

valid_images, valid_labels = next(data_valid)

#create image generators
hflip = ImageDataGenerator(rescale=1./255,horizontal_flip=True,validation_split=0.2)
vflip = ImageDataGenerator(rescale=1./255,vertical_flip=True,validation_split=0.2)
hshift = ImageDataGenerator(rescale=1./255,width_shift_range=(-40,40),validation_split=0.2)
vshift = ImageDataGenerator(rescale=1./255,height_shift_range=(-40,40),validation_split=0.2)
rotation = ImageDataGenerator(rescale=1./255,rotation_range=90,validation_split=0.2)
shear = ImageDataGenerator(rescale=1./255,shear_range=45,validation_split=0.2)
hvflip = ImageDataGenerator(rescale=1./255,horizontal_flip=True,vertical_flip=True,validation_split=0.2)
bright = ImageDataGenerator(rescale=1./255,brightness_range=(.2,1),validation_split=0.2)
bright_rot = ImageDataGenerator(rescale=1./255,brightness_range=(.2,1),rotation_range=90,validation_split=0.2)


augmentors = [hflip,vflip,hshift,vshift,rotation,shear,hvflip,bright,bright_rot]

#iterate through augmentors and generate augmented image sets
mild_augmented_train = []
mod_augmented_train = []

for augmentor in augmentors[:2]:

    aug_mild_train = augmentor.flow_from_directory( 
        train_directory, 
        subset= 'training',
        target_size=IMAGE_SIZE,
        batch_size = 717,  
        classes= ['MildDemented'])

    mild_augmented_train.append(aug_mild_train)

for augmentor in augmentors:    

    aug_moderate_train = augmentor.flow_from_directory( 
        train_directory, 
        subset = "training",
        target_size=IMAGE_SIZE, 
        classes= ['ModerateDemented'])
    
    mod_augmented_train.append(aug_moderate_train)    

#extract image matricies from generators and separate out images matricies from labels
train_mild_aug = [next(mild) for mild in mild_augmented_train]  
train_mod_aug = [next(mod) for mod in mod_augmented_train]

mild_aug_images_train = [images[0] for images in train_mild_aug]
mod_aug_images_train = [images[0] for images in train_mod_aug]

#concat all matricies together
concat_mild_images_train = np.vstack(mild_aug_images_train)
concat_mod_images_train = np.vstack(mod_aug_images_train)

#create label matricies 
concat_mild_labels_train = np.array([[1.,0.,0.,0.] for i in range(len(concat_mild_images_train))])
concat_mod_labels_train = np.array([[0.,1.,0.,0.] for i in range(len(concat_mod_images_train))])

#concat back to original training data
train_images_aug = np.concatenate((train_images,concat_mild_images_train,concat_mod_images_train))
train_labels_aug = np.concatenate((train_labels,concat_mild_labels_train,concat_mod_labels_train))