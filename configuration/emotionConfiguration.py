from imutils import paths

datasetPath = "dataset/fer2013/fer2013.csv" # path to csv file
numClasses = 6
# define the path to the output training, validation, and testing
train_hdf5 = "dataset/hdf5/train.hdf5"
vald_hdf5 = "dataset/hdf5/val.hdf5"
test_hdf5 = "dataset/hdf5/test.hdf5"
batchSize = 64