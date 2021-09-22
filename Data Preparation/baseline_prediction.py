from utils import *
from random import randrange

def random_algorithm(train, test):
	output_values = [row[-1] for row in train]
	unique = list(set(output_values))
	predicted = []
	for row in test:
		index = randrange(len(unique))
		predicted.append(unique[index])

	return predicted

def zero_rule_algorithm_classification(train, test):
	output_values = [row[-1] for row in train]
	prediction = max(set(output_values), key=output_values.count) 

	return [prediction for i in range(len(test))]

def zero_rule_algorithm_regression(train, test):
	output_values = [row[-1] for row in train]
	prediction = sum(output_values) /len(output_values)

	return [prediction for i in range(len(test))]


if __name__ == '__main__':

	dataset = load_csv_file("dataset/pima-indians-diabetes-data.csv", float)

	train, test = train_test_split(dataset, 0.8)

	predicted = random_algorithm(train, test)
	score = accuracy_metric([each[-1] for each in test], predicted)
	print(score)
	