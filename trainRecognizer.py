from configuration import emotionConfiguration as config
from utils import plotHistory, ImageToArrayPreprocessor, makeOneMatrix, plot_confusion_matrix
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

trainAug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10, zoom_range=0.1,
                            horizontal_flip=True, rescale=1 / 255.0, fill_mode="nearest")
valAug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
testAug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()
# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.train_hdf5, config.batchSize,
                                aug=trainAug, preprocessors=[iap], classes=config.numClasses)
valGen = HDF5DatasetGenerator(config.vald_hdf5, config.batchSize,
                            aug=valAug, preprocessors=[iap], classes=config.numClasses)
testGen = HDF5DatasetGenerator(config.test_hdf5, config.batchSize,
                            aug=testAug, preprocessors=[iap], classes=config.numClasses)

trainData, trainLabels = makeOneMatrix(trainGen)
valData, valLabels = makeOneMatrix(valGen)
testData, testLabels = makeOneMatrix(testGen)

numEpochs = 1
batchSize = 64
learningRate = 0.0001

# model 1 = 2304-672-192-64-20-6
# model 2 = 2304-512-128-20-6
# model 3 = 2304-90-32-6
regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(150, activation='sigmoid', input_dim= 48 * 48))
model.add(tf.layers.BatchNormalization() )
# model.add(tf.layers.Dropout(0.5))

# model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
# model.add(tf.layers.BatchNormalization())
# model.add(tf.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(config.numClasses, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
history = model.fit(trainData, trainLabels, epochs=numEpochs, batch_size=batchSize,
                    validation_data=(valData, valLabels))

predicted = model.predict_classes(testData)
confusionMatrix = confusion_matrix(testLabels, predicted)
classesNames = ["Angry", "Fear", "Happy", "Sad", "surprised", "sleepy"]
plot_confusion_matrix(confusionMatrix, classes=classesNames,
                      title='Confusion matrix, without normalization')
confusionMatrix = np.zeros((6, 6), np.int32)
for i, j in zip(testLabels, predicted):
    confusionMatrix[int(i), int(j)] += 1
print(confusionMatrix)

print(classification_report(testAug.classes, predicted, target_names=classesNames))

# close the databases
trainGen.close()
valGen.close()
testGen.close()

# print result
plotHistory(history)