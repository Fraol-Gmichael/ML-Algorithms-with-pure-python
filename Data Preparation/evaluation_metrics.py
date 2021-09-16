from scale_and_normalize import *
import random

dataset = load_csv_file("dataset/pima-indians-diabetes-data.csv", float)


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


actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,1,0,0,1,0,1,1,1]
conf_mat = confusion_matrix(actual, predicted)


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

if __name__ == '__main__':
	actual = [0.1, 0.2, 0.3, 0.4, 0.5]
	predicted = [0.11, 0.19, 0.29, 0.41, 0.5]

	print(mean_squared_error(actual, predicted))
