import os
import sys
import time
import rasterio
import glob
import numpy as np
import tensorflow as tf
from shutil import copyfile
import random
import warnings
import pandas as pd
from tensorflow.keras.utils import to_categorical
from random import sample, choice
from skimage.transform import rotate

def time_series_image_gen(
        plot_ids,
        label_file_path,
        band_file_path,
        dates,
        n_classes=8,
        n_features = 4,
        batch_size=3,
        image_size = 64,
        rotation_range=0,
        horizontal_flip=False,
        vertical_flip=False,
        seed=42
):
    """
    Adapted image generator from
    github.com/jensleitloff/CNN-Sentinel/blob/master/py/image_functions.py.
    Instead of the whole image, now each pixel is labelled
    with a one-hot-vector.
    The output arrays are now 4D: tiles are picked from different images
    that should be ordered chronologially in the 'dates' argument.
    Args:
        path_to_training_data (str):
            Indicates the directory that contains the subdirectories
            for different dates.
        dates (list):
            List of strings with the dates of the different pictures
            in format 'YYYY'.
        n_classes (int):
            Number of classes that are considered in the classification.
        batch_size (int):
            Size of the batch that the generator yields in one iteration.
        rotation_range (int):
            Rotation angle by which image is rotated at most (in degrees).
        horizontal_flip (bool):
            If True, allows chance of flipping the image horizontally.
        vertical_flip (bool):
            If True, allows chance of flipping the image vertically.
    Returns:
        X (np.array):
            Numpy array with shape
            (batch_size, timesteps, img_size, img_size, n_features)
            that contains a batch of sequences of landsat image tiles.
        Y_one_hot (np.array):
            Numpy array with shape
            (batch_size, img_size, img_size, n_classes)
            that contains a batch of one-hot-encoded labels corresponding to
            the data in X.
    """
    random.seed = seed
    np.random.seed = seed
    timesteps = len(dates)
    label_files = []
    band_files = []
    label_file_path = os.path.join(label_file_path,'plot_{}_{}.tif')
    band_file_path = os.path.join(band_file_path,'Landsat7_{}_{}.tif')
    for plot_id in plot_ids:
        for date in dates:
            label_files.append(label_file_path.format(plot_id,date))
            band_files.append(band_file_path.format(plot_id,date))
    while True:
        X = np.empty((batch_size, len(dates), image_size, image_size, n_features))
        Y = np.empty((batch_size, image_size, image_size))
        Y_one_hot = np.empty((batch_size, image_size, image_size, n_classes))

        # draw samples
        batchPlotIds = sample(plot_ids, batch_size)
        
        
        for i,batchPlot in enumerate(batchPlotIds):
            for j,date in enumerate(dates):
                with rasterio.open(band_file_path.format(batchPlot,date)) as bandsrc:
                    bandDataRaw = bandsrc.read()
                    bandData = np.nan_to_num(bandDataRaw, copy=True)
        
                bandData = np.transpose(bandData, (1, 2, 0))
                X[i,j] = bandData
                        
            with rasterio.open(label_file_path.format(batchPlot)) as src:
                label = src.read()
            Y[i] = label

        # image augmentation and one-hot-encoding
        # make sure to apply the same augmentation to every date
        # 1. outer loop: sample nr in batch
        # 2. random decision of image augmentation params
        # 3. inner loop: apply augmentation with params to all dates
        for s in range(batch_size):
            if horizontal_flip:
                # randomly flip image up/down
                if choice([True, False]):
                    for d in range(timesteps):
                        for f in range(n_features):
                            X[s, d, ..., f] = np.flipud(X[s, d, ..., f])
                    Y[s, ...] = np.flipud(Y[s, ...])
            if vertical_flip:
                # randomly flip image left/right
                if choice([True, False]):
                    for d in range(timesteps):
                        for f in range(n_features):
                            X[s, d, ..., f] = np.fliplr(X[s, d, ..., f])
                    Y[s, ...] = np.fliplr(Y[s, ...])
            # rotate image by random angle between
            # -rotation_range <= angle < rotation_range
            if rotation_range !=0:
                angle = np.random.uniform(low=-abs(rotation_range),
                                          high=abs(rotation_range))
                for d in range(timesteps):
                    for f in range(n_features):
                        X[s, d, ..., f] = rotate(X[s, d, ..., f],
                                                 angle,
                                                 mode='reflect',
                                                 order=1,
                                                 preserve_range=True
                                                 )
                Y[s, ...] = rotate(Y[s, ...],
                                   angle,
                                   mode='reflect',
                                   order=1,
                                   preserve_range=True
                                   )
            Y_encoded = to_categorical(Y[s, ...], n_classes, dtype ="float64")
            
            for c in range(n_classes):
                Y_one_hot[s, ..., c] = Y_encoded[:,:,c]
        yield (X, Y_one_hot)

        
def time_series_image(plot_id, label_file_path, band_file_path, dates):
    timesteps = len(dates)
    X = np.empty((len(dates), image_size, image_size, n_features))
    for i,date in enumerate(dates):
        with rasterio.open(band_file_path.format(plot_id,date)) as bandsrc:
            bandDataRaw = bandsrc.read()
            bandData = np.nan_to_num(bandDataRaw, copy=True)
        X[i] = np.transpose(bandData, (1, 2, 0))

    with rasterio.open(label_file_path.format(plot_id)) as src:
        label = src.read()
    return X, label


def image_gen(
        idDateList,
        label_file_path,
        band_file_path,
        n_classes=8,
        n_features = 4,
        batch_size=3,
        image_size = 64,
        rotation_range=0,
        labelFileFormat = 'plot_{}_{}.tif',
        bandFileFormat = 'Landsat7_{}_{}.tif',
        horizontal_flip=False,
        vertical_flip=False,
        seed=42,
        bands = 'all'
):
#get batch sample using 
#X,label = next(train_sequence_generator)
    """
    Adapted image generator from
    github.com/jensleitloff/CNN-Sentinel/blob/master/py/image_functions.py.
    Instead of the whole image, now each pixel is labelled
    with a one-hot-vector.
    The output arrays are now 4D: tiles are picked from different images
    that should be ordered chronologially in the 'dates' argument.
    Args:
        path_to_training_data (str):
            Indicates the directory that contains the subdirectories
            for different dates.
        n_classes (int):
            Number of classes that are considered in the classification.
        batch_size (int):
            Size of the batch that the generator yields in one iteration.
        rotation_range (int):
            Rotation angle by which image is rotated at most (in degrees).
        horizontal_flip (bool):
            If True, allows chance of flipping the image horizontally.
        vertical_flip (bool):
            If True, allows chance of flipping the image vertically.
    Returns:
        X (np.array):
            Numpy array with shape
            (batch_size, img_size, img_size, n_features)
            that contains a batch of landsat image tiles.
        Y_one_hot (np.array):
            Numpy array with shape
            (batch_size, img_size, img_size, n_classes)
            that contains a batch of one-hot-encoded labels corresponding to
            the data in X.
    """
    random.seed = seed
    np.random.seed = seed
    label_files = []
    band_files = []
    label_file_path = os.path.join(label_file_path,labelFileFormat)
    band_file_path = os.path.join(band_file_path,bandFileFormat)
    for plot_id, date in idDateList:
        label_files.append(label_file_path.format(plot_id,date))
        band_files.append(band_file_path.format(plot_id,date))
    while True:
        X = np.zeros((batch_size, image_size, image_size, n_features))
        Y = np.zeros((batch_size, image_size, image_size))
        Y_one_hot = np.zeros((batch_size, image_size, image_size, n_classes))

        # draw samples
        batchPlotIds = sample(idDateList, batch_size)
        
        
        for i,idDate in enumerate(batchPlotIds):
            with rasterio.open(band_file_path.format(idDate[0],idDate[1])) as bandsrc:
                if bands=='all':
                    bandDataRaw = bandsrc.read()
                else:
                    bandDataRaw = bandsrc.read(bands)    
                bandData = np.nan_to_num(bandDataRaw, copy=True)
        
            bandData = np.transpose(bandData, (1, 2, 0))
            X[i] = bandData

            with rasterio.open(label_file_path.format(idDate[0],idDate[1])) as src:
                label = src.read()
            Y[i] = label

        # image augmentation and one-hot-encoding
        # make sure to apply the same augmentation to every date
        # 1. outer loop: sample nr in batch
        # 2. random decision of image augmentation params
        # 3. inner loop: apply augmentation with params to all dates
        for s in range(batch_size):
            if horizontal_flip:
                # randomly flip image up/down
                if choice([True, False]):
                    for f in range(n_features):
                        X[s, ..., f] = np.flipud(X[s, ..., f])
                    Y[s, ...] = np.flipud(Y[s, ...])
            if vertical_flip:
                # randomly flip image left/right
                if choice([True, False]):
                    for f in range(n_features):
                        X[s, ..., f] = np.fliplr(X[s, ..., f])
                    Y[s, ...] = np.fliplr(Y[s, ...])
            # rotate image by random angle between
            # -rotation_range <= angle < rotation_range
            if rotation_range !=0:
                angle = np.random.uniform(low=-abs(rotation_range),
                                          high=abs(rotation_range))
                for f in range(n_features):
                    X[s, ..., f] = rotate(X[s, ..., f],
                                             angle,
                                             mode='reflect',
                                             order=1,
                                             preserve_range=True
                                             )
                Y[s, ...] = rotate(Y[s, ...],
                                   angle,
                                   mode='reflect',
                                   order=1,
                                   preserve_range=True
                                   )
            Y_encoded = to_categorical(Y[s, ...], n_classes, dtype ="float64")
            
            for c in range(n_classes):
                Y_one_hot[s, ..., c] = Y_encoded[:,:,c]
        yield (X, Y_one_hot)

        
def read_image(idDate, label_file_path, band_file_path, bands = 'all'):
    with rasterio.open(band_file_path.format(idDate[0],idDate[1])) as bandsrc:
        if bands=='all':
            bandDataRaw = bandsrc.read()
        else:
            bandDataRaw = bandsrc.read(bands)
        bandData = np.nan_to_num(bandDataRaw, copy=True)
    X = np.transpose(bandData, (1, 2, 0))

    with rasterio.open(label_file_path.format(idDate[0],idDate[1])) as src:
        label = src.read()
    return X, label
    
def read_satellite(idDate, band_file_path, bands = 'all'):
    with rasterio.open(band_file_path.format(idDate[0],idDate[1])) as bandsrc:
        if bands=='all':
            bandDataRaw = bandsrc.read()
        else:
            bandDataRaw = bandsrc.read(bands)
        bandData = np.nan_to_num(bandDataRaw, copy=True)
    X = np.transpose(bandData, (1, 2, 0))

    return X
            