import csv
from math import sqrt
import time
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




if __name__ == '__main__':

	dataset = load_csv_file('dataset/pima-indians-diabetes-data.csv')
	dataset = convert_dataset_to(dataset)
	to_csv(dataset)
	std = standardize_dataset(dataset)
	print(std[:3])