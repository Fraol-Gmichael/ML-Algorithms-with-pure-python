from utils import *
from random import randrange

def random_algorithm(train, test):
	trainN, testN = train[1:], test[1:]
	output_values = [row[-1] for row in trainN]
	unique = list(set(output_values))
	predicted = [test[0][-1]]
	for row in testN:
		index = randrange(len(unique))
		predicted.append(unique[index])

	return predicted

def zero_rule_algorithm_classification(train, test):
	trainN, testN = train[1:], test[1:]
	output_values = [row[-1] for row in trainN]
	prediction = max(set(output_values), key=output_values.count)
	predicted = [test[0][-1]] 
	[predicted.append(prediction) for i in range(len(testN))]

	return predicted

def zero_rule_algorithm_regression(train, test):
	trainN, testN = train[1:], test[1:]
	output_values = [row[-1] for row in trainN]
	prediction = sum(output_values) /len(output_values)
	predicted = [test[0][-1]] 
	[predicted.append(prediction) for i in range(len(testN))]

	return predicted


dataset = load_csv_file("dataset/pima-indians-diabetes-data.csv", float)

train, test = train_test_split(dataset, 0.8)

predicted = zero_rule_algorithm_regression(train, test)
score = mean_abs_error([each[-1] for each in test], predicted)
print(score)