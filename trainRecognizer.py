from configuration import emotionConfiguration as config
from utils import loadData, transposeData, plotHistory, ImageToArrayPreprocessor, makeOneMatrix
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
import numpy as np
import tensorflow as tf
import pickle
import os

# Loading the data
# trainData, trainLabels, valData, valLabels, testData, testLabels = loadData(trainPath, 
#                                                                     validationPath, testPath, numClasses)

# trainData, trainLabels = transposeData(trainData, trainLabels)
# valData, valLabels = transposeData(valData, valLabels)
# testData, testLabels = transposeData(testData, testLabels)

trainAug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10, zoom_range=0.1,
                            horizontal_flip=True, rescale=1 / 255.0, fill_mode="nearest")
valAug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()
# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.train_hdf5, config.batchSize,
                                aug=trainAug, preprocessors=[iap], classes=config.numClasses)
valGen = HDF5DatasetGenerator(config.vald_hdf5, config.batchSize,
                            aug=valAug, preprocessors=[iap], classes=config.numClasses)

trainData, trainLabels = makeOneMatrix(trainGen)
valData, valLabels = makeOneMatrix(valGen)

numEpochs = 15
batchSize = 64
learningRate = 0.0001

# our model = 2304-672-192-64-20-6
# our model = 2304-512-128-20-6
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(672, activation='relu', input_dim= 48 * 48))
model.add(tf.layers.BatchNormalization())
model.add(tf.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(192, activation='relu'))
model.add(tf.layers.BatchNormalization())
model.add(tf.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.layers.BatchNormalization())
model.add(tf.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(20, activation='relu'))
model.add(tf.layers.BatchNormalization())
model.add(tf.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(config.numClasses, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
history = model.fit(trainData, trainLabels, epochs=numEpochs, batch_size=batchSize,
                    validation_data=(valData, valLabels))

# history = model.fit_generator(trainGen.generator(), steps_per_epoch=trainGen.numImages // config.batchSize,
#                                 validation_data=valGen.generator(),
#                                 validation_steps=valGen.numImages // config.batchSize,
#                                 epochs= numEpochs,
#                                 max_queue_size=config.batchSize * 2,
#                                 max_queue_size=config.batchSize * 2,
#                                 verbose=1))

# close the databases
trainGen.close()
valGen.close()
# print result
plotHistory(history)