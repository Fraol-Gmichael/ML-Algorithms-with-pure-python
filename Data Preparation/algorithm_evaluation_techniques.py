from scale_and_normalize import *
import random
dataset = load_csv_file("dataset/pima-indians-diabetes-data.csv", float)

def train_test_split(dataset, split=0.6):
	datasetN = dataset[1:]
	random.shuffle(datasetN)

	train_len = int((len(datasetN) * split)+1)
	train = [dataset[0]]
	train.extend(datasetN[:train_len])
	test = datasetN[train_len:]

	return train, test

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
	datasetN = dataset[1:]
	random.shuffle(datasetN)
	cross_result = []
	fold_size = int(len(datasetN)/folds)
	for each in range(folds):
		start = each * fold_size
		end = start + fold_size
		if end > len(datasetN):
			end = len(datasetN)
		test = datasetN[start:end]
		train = datasetN[:start]
		train.extend(datasetN[end:])
		cross_result.append([test, train])

	return cross_result
print(cross_validation_split(dataset))
