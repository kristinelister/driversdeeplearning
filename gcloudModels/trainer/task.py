from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

#Import utility functions
from . import modelFunctions
from . import getTiles

import argparse
import tensorflow as tf

#Import other modules
import random
import glob
import warnings
import numpy as np
import subprocess

def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting models')
    parser.add_argument(
        '--trainingFilePath',
        type=str,
        default='gcloudInputs/trainingPlots',
        help='local or GCS location for training tiles')
    parser.add_argument(
        '--validationFilePath',
        type=str,
        default='gcloudInputs/validationPlots',
        help='local or GCS location for validation tiles')
    parser.add_argument(
        '--bandFilePath',
        type=str,
        default='gcloudInputs/landsatTilesPostYear',
        help='local or GCS location for landsat tiles')
    parser.add_argument(
        '--gsInputBucket',
        type=str,
        default='gs://drivers2-bucket/gcloudInputs',
        help='gs bucket with inputs, default=gs://drivers2-bucket/gcloudInputs')
    parser.add_argument(
        '--class_weights',
        type=list,
        default=[1, 10, 10, 30, 30, 30, 30,30],
        help='number of classes, default=8')
    parser.add_argument(
        '--n_classes',
        type=int,
        default=8,
        help='number of classes, default=8')
    parser.add_argument(
        '--n_features',
        type=int,
        default=4,
        help='number of features, default=4')
    parser.add_argument(
        '--image_size',
        type=int,
        default=64,
        help='number of features, default=64')
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=1,
        help='number of times to go through the data, default=10')
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning_rate',
        default=.001,
        type=float,
        help='learning rate for gradient descent, default=.0001'),
    parser.add_argument(
        '--modelName',
        default='model',
        type=str,
        help='name for model folder, default=model')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args

def train_and_evaluate(args):
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job_dir argument.

    Args:
      args: dictionary of arguments - see get_args() for details
    """

    #Check if job dir exists, if not create it
    if not os.path.isdir(args.job_dir):
        os.makedirs(args.job_dir)
    
    #Download training files
    subprocess.check_output('gsutil -m cp -r {} {}'.format(args.gsInputBucket,args.job_dir),shell=True)
    
    #Join job dir and folder paths
    trainingFilePath = os.path.join(args.job_dir,args.trainingFilePath)
    validationFilePath = os.path.join(args.job_dir,args.validationFilePath)
    bandFilePath = os.path.join(args.job_dir,args.bandFilePath)
    
    #Get [[plotID, year]] list for training plots
    trainingFileNames = glob.glob(trainingFilePath+'/*.tif')
    trainingFileNames.sort()
    trainingPlots = [[int(x.split('_')[1]),int(x.split('_')[2].split('.')[0])] for x in trainingFileNames]
    
    #Get [[plotID, year]] list for validation plots
    validationFileNames = glob.glob(validationFilePath+'/*.tif')
    validationFileNames.sort()
    validationPlots = [[int(x.split('_')[1]),int(x.split('_')[2].split('.')[0])] for x in validationFileNames]
    
    #Define generator for training and validation data
    train_sequence_generator = getTiles.image_gen(trainingPlots,trainingFilePath,bandFilePath,
            args.n_classes,
            args.n_features,
            args.batch_size,
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True,bands='all')


    validation_sequence_generator = getTiles.image_gen(validationPlots,validationFilePath,bandFilePath,
            args.n_classes,
            args.n_features,
            batch_size=args.batch_size,
            rotation_range=0,
            horizontal_flip=False,
            vertical_flip=False,bands='all')
            

    # Create the Keras Model
    model = modelFunctions.compileUnetModel(args.n_classes, args.image_size, args.n_features, args.class_weights, learning_rate=args.learning_rate, w_decay=0.0005)

    # Setup TensorBoard callback.
    tensorboard = tf.keras.callbacks.TensorBoard(
        os.path.join(args.job_dir, 'keras_tensorboard'),
        histogram_freq=1)
        
    #Callbacks for training
    decay = args.learning_rate/args.n_epochs
    def lr_time_based_decay(epoch):
            return args.learning_rate * 1 / (1 + decay * epoch)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_time_based_decay,verbose=1)
    
    earlystopper = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)

    # Train model
    model.fit(train_sequence_generator, 
                    validation_data=validation_sequence_generator,validation_steps=5, 
                    epochs=args.n_epochs,steps_per_epoch=20,
                    callbacks=[lr_schedule,earlystopper,tensorboard],
                    verbose=1)


    export_path = os.path.join(args.job_dir, 'model')
    model.save(export_path)
    print('Model exported to: {}'.format(export_path))


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)