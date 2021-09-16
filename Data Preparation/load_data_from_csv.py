import csv
from math import sqrt
def convert_dataset_to(dataset, dtype=float):
	dict_data  = {}
	data = [dataset[0]]

	for row in dataset[1:]:
		data.append([dtype(float(each.strip())) for each in row])

	return data

def load_csv_file(file_name):
	dataset = []
	with open(file_name, 'r') as file:
		lines = csv.reader(file)
		for row in lines:
			if row:
				dataset.append(row)
		return dataset

