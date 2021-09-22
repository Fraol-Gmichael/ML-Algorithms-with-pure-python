import sys
sys.path.append("C:/Users/gyon/Desktop/ML _ SRC/Data Preparation")
import baseline_prediction as bp
import utils

dataset = utils.load_csv_file('datasets/pima-indians-diabetes-data.csv', dtype=float)


def evaluate_algotithm_normal(dataset, algorithm, acc, split):
	train, test = utils.train_test_split(dataset, split)
	test_set = [each[-1] for each in test]
	predicted = algorithm(train, test)
	accuracy = acc(test_set, predicted)

	return accuracy


def evaluate_algotithm_cross_val(dataset, algorithm, acc, fold=3):
	cross = utils.cross_validation_split(dataset, fold)
	result = {}
	fold_n = 1
	for train, test in cross:
		test_set = [each[-1] for each in test]
		predicted = algorithm(train, test)
		accuracy = acc(test_set, predicted)
		result[f'Fold {fold_n}'] = accuracy
		fold_n += 1
	
	return result

if __name__ == '__main__':
	train_t = evaluate_algotithm_normal(dataset, bp.random_algorithm, utils.accuracy_metric, 0.8)
	cross = evaluate_algotithm_cross_val(dataset, bp.random_algorithm, utils.accuracy_metric, 5)
	print(cross)
