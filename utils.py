import h5py
from configuration import emotionConfiguration as config
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def loadData(trainPath, valPath, testPath, numClasses):
    db_train = h5py.File(trainPath)
    db_val = h5py.File(valPath)
    db_test = h5py.File(testPath)
    

    trainData = np.array(db_train["images"][:])
    trainLabels = np.array(db_train["labels"][:])

    valData = np.array(db_val["images"][:])
    valLabels = np.array(db_val["labels"][:])

    testData = np.array(db_test["images"][:])
    testLabels = np.array(db_test["labels"][:])

    trainData, trainLabels = standarize(trainData, trainLabels, numClasses)
    valData, valLabels = standarize(valData, valLabels, numClasses)
    testData, testLabels = standarize(testData, testLabels, numClasses)
    
    return trainData, trainLabels, valData, valLabels, testData, testLabels

def standarize(data, labels, numClasses):
    # A trick when you want to flatten a matrix X of shape (a,b,c,d) 
    # to a matrix X_flatten of shape (b∗∗c∗∗d, a) is to use:
    # X_flatten = X.reshape(a, -1).T      
    # # X.T is the transpose of X

    data_flatten = data.reshape(data.shape[0], -1).T # vectorize all images
    # standardize our dataset
    data = data_flatten / 255
    # Convert training and test labels to one hot matrices
    labels = np.eye(numClasses)[labels.reshape(-1)].T

    return data, labels

def transposeData(data, label):
    data = np.transpose(data)
    label = np.transpose(label)
    return data, label

def plotHistory(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		# store the image data format
		self.dataFormat = dataFormat

	def preprocess(self, image):
		# apply the Keras utility function that correctly rearranges
		# the dimensions of the image
		return tf.keras.preprocessing.image.img_to_array(image, data_format=self.dataFormat)

def makeOneMatrix(Gen):
    # initialize the image matrix and label
    features = np.zeros(shape=(Gen.numImages, 48, 48, 1))
    labels = np.zeros(shape=(Gen.numImages, config.numClasses))
    i = 0
    for inputs_batch, labels_batch in Gen.generator():
        features[i * config.batchSize : (i + 1) * config.batchSize] = inputs_batch
        labels[i * config.batchSize : (i + 1) * config.batchSize] = labels_batch
        i += 1
        if i * config.batchSize >= Gen.numImages:
            break

    features = np.reshape(features, (Gen.numImages, 48 * 48))
    return features, labels