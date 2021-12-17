
#Followed model structure from this tutorial
#https://medium.com/@pallawi.ds/semantic-segmentation-with-u-net-train-and-test-on-your-custom-data-in-keras-39e4f972ec89

#Found keras loss functions from below
#https://github.com/maxvfischer/keras-image-segmentation-loss-functions
#https://github.com/maxvfischer/keras-image-segmentation-loss-functions/blob/master/losses/multiclass_losses.py#L107

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.layers.merge import concatenate
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from getTiles import image_gen
from getTiles import read_image

#from tensorflow.keras.optimizers import SGD
def iou_coef(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, dtype=tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return 1 - iou
    

#Keras
def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    targets = tf.cast(targets, dtype=tf.float32)
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(K.dot(targets, inputs))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


def weighted_categorical_crossentropy(weights):
    #https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss
    
    
def prepUnetModel(n_classes=8, image_size=64, n_features=4, w_decay = 0.0005):
    #Build U-Net model
    inputs = Input((image_size, image_size, n_features))

    s = Lambda(lambda x: x) (inputs)

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',kernel_regularizer=l2(w_decay)) (c9)

    outputs = Conv2D(n_classes, (1, 1), padding="same", activation="softmax") (c9)

    return inputs, outputs

    
def compileUnetModel(n_classes, image_size, n_features, class_weights, learning_rate=0.001, w_decay=0.0005):
    inputs, outputs = prepUnetModel(n_classes=n_classes, image_size=image_size, 
                                    n_features=n_features, w_decay = w_decay)
    model = Model(inputs=[inputs], outputs=[outputs])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=weighted_categorical_crossentropy(class_weights),
                  optimizer=optimizer,
                  metrics=[iou_coef])
    return model
    
