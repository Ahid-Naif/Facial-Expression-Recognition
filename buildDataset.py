# import the necessary packages
from configuration import emotionConfiguration as config
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
import numpy as np

# open the input file for reading (skipping the header), then
# initialize the list of data and labels for the training,
# validation, and testing sets
print("[INFO] loading input data...")
f = open(config.datasetPath)
lines = f.readlines()

(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])

# loop over the rows in the input file
for i in range(0, len(lines)):
	if i != 0:
		line = lines[i]
	else:
		continue
	# extract the label, image, and usage from the row
	(label, image, usage) = line.strip().split(",")
	label = int(label)

	# if we are ignoring the "disgust" class there will be 6 total
	# class labels instead of 7
	if config.numClasses == 6:
		# skip "anger" class
		if label == 1:
			continue

		# if label has a value greater than zero, subtract one from
		# it to make all labels sequential (not required, but helps
		# when interpreting results)
		if label > 0:
			label -= 1

	# reshape the flattened pixel list into a 48x48 (grayscale)
	# image
	image = np.array(image.split(" "), dtype="uint8")
	image = image.reshape((48, 48))

	# check if we are examining a training image
	if usage == "Training":
		trainImages.append(image)
		trainLabels.append(label)

	# check if this is a validation image
	elif usage == "PrivateTest":
		valImages.append(image)
		valLabels.append(label)

	# otherwise, this must be a testing image
	else:
		testImages.append(image)
		testLabels.append(label)

# construct a list pairing the training, validation, and testing
# images along with their corresponding labels and output HDF5
# files
datasets = [
	(trainImages, trainLabels, config.train_hdf5),
	(valImages, valLabels, config.vald_hdf5),
	(testImages, testLabels, config.test_hdf5)]

# loop over the dataset tuples
for (images, labels, outputPath) in datasets:
	# create HDF5 writer
	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter((len(images), 48, 48), outputPath)

	# loop over the image and add them to the dataset
	for (image, label) in zip(images, labels):
		writer.add([image], [label])

	# close the HDF5 writer
	writer.close()

# close the input file
f.close()