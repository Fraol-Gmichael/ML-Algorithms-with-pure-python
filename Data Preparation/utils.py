import csv
from math import sqrt
import time
import random

def convert_dataset_to(dataset, dtype=float):
	dict_data  = {}
	data = [dataset[0]]

	for row in dataset[1:]:
		data.append([dtype(float(each.strip())) for each in row])

	return data

def load_csv_file(file_name, dtype=None):
	dataset = []
	with open(file_name, 'r') as file:
		lines = csv.reader(file)
		for row in lines:
			if row:
				dataset.append(row)
		if dtype:
			return convert_dataset_to(dataset, dtype)
		
		return dataset


def classifier(column):
	col = {}
	index = 0
	for each in column:
		if each not in col:
			col[each] = index
			index+=1
	ans = []
	for each in column:
		ans.append(col[each])

	return ans


def dataset_min_max(dataset):
	min_max = []
	min_max_dict = {}
	for i, _ in enumerate(dataset[0]):
		column = [each[i] for each in dataset[1:]]
		min_val, max_val = min(column), max(column)
		min_max.append([min_val, max_val])
		min_max_dict[_] = [min_val, max_val]

	return min_max

def normalize(dataset):
	datasetN = dataset.copy()
	min_max = dataset_min_max(datasetN)
	for i, row in enumerate(datasetN[1:]):
		datasetN[i+1] = [(each-min_max[index][0])/(min_max[index][1]-min_max[index][0]) for index, each in enumerate(row)]

	return datasetN

def column_means(dataset):
	mean_vals = []
	for i, _ in enumerate(dataset[0]):
		column = [each[i] for each in dataset[1:]]
		col_mean= sum(column) / len(column)
		mean_vals.append(col_mean)

	return mean_vals


def column_std(dataset):
	means = column_means(dataset)
	std = []
	denum = len(dataset) - 2
	for i, _ in enumerate(dataset[0]):
		column = [each[i] for each in dataset[1:]]
		mean = means[i]
		numerator = sum([(each-mean)**2 for each in column])
		std.append(sqrt(numerator/denum))

	return std

def standardize_dataset(dataset):
	means = column_means(dataset)
	std = column_std(dataset)
	datasetN = dataset.copy()
	for i, row in enumerate(datasetN[1:]):
		datasetN[i+1] = [(each-means[index])/std[index] for index, each in enumerate(row)]

	return datasetN

def to_csv(dataset, file_name=f"dataset/new-file {int(time.time())}.csv"):
	with open(file_name, 'w', newline='') as file:
		writer = csv.writer(file)
		for row in dataset:
			writer.writerow(row)


def accuracy_metric(actual, predicted):
	if len(actual) != len(predicted):
		raise Exception("Actual and predicted are not equal in length")
	correct = 0
	total = len(actual)

	for act, pre in zip(actual, predicted):
		if act == pre:
			correct += 1

	accuracy = correct/total

	return accuracy * 100

def confusion_matrix(actual, predicted):
	if len(actual) != len(predicted):
		raise Exception("Actual and predicted are not equal in length")
	correct = 0
	n = []
	p = []
	
	tn, tp, fn, fp = 0, 0, 0, 0 
	for act, pre in zip(actual, predicted):
		if act == pre:
			if pre == 1:
				tp += 1
			else:
				tn += 1
		else:

			if pre == 0:
				fn += 1
			else:
				fp += 1 
	n.append(tn), n.append(fn)
	p.append(tp), p.append(fp)
	
	return [n, p]

def mean_abs_error(actual, predicted):
	sum_error = sum([abs(act - pre) for act, pre in zip(actual, predicted)])
	total = len(predicted)
	return sum_error / total

def root_mean_squared_error(actual, predicted):
	sum_error = sum([(act - pre)**2 for act, pre in zip(actual, predicted)])
	total = len(predicted)
	return sqrt(sum_error / total)

def mean_squared_error(actual, predicted):
	return root_mean_squared_error(actual, predicted)**2

def train_test_split(dataset, split=0.6):
	datasetN = dataset[1:]
	random.shuffle(datasetN)
	train_len = int((len(datasetN) * split)+1)
	train = datasetN[:train_len]
	test = datasetN[train_len:]

	return train, test

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
	datasetN = dataset[1:]
	random.shuffle(datasetN)
	cross_result = []
	fold_size = int(len(datasetN)/folds)
	for each in range(folds):
		#train, test =  [dataset[0]],  [dataset[0]]
		start = each * fold_size
		end = start + fold_size
		test = datasetN[start:end]
		train = datasetN[:start]
		train.extend(datasetN[end:])
		cross_result.append([train, test])

	return cross_result


def mean(values):
	return sum(values) / float(len(values))

def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

