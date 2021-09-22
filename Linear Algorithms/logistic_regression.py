import sys
sys.path.append("C:/Users/gyon/Desktop/ML _ SRC/Data Preparation")
import baseline_prediction as bp
import utils
import test_harness
from math import exp

dataset = utils.load_csv_file('datasets/pima-indians-diabetes-data.csv', dtype=float)
dataset = utils.normalize(dataset)
train, test = utils.train_test_split(dataset)


class LogisticRegression:
	def __init__(self,  l_rate=0.0001, n_epoch=5000):
		self.l_rate = l_rate
		self.n_epoch = n_epoch
		self.fitted = False

	def fit(self, train):
		self.coefficients = self.coefficients_sgd(train)
		self.fitted = True

	def predict(self, test_X):
		if self.fitted:
			return [round(self.predict_row(row_x, self.coefficients)) for row_x in test_X]

	# Y = 1 / 1 + exp(a0 + a1x1 + a2x2 ....)
	def predict_row(self, row, coefficients):

		yhat = coefficients[0]
		for i in range(len(row)-1):
			yhat += coefficients[i+1] * row[i]

		return 1 / (1 + exp(-yhat))

	def score(self, actual, predicted):
		return utils.accuracy_metric(actual, predicted)

	def coefficients_sgd(self, train):
		coef = [0.0 for i in range(len(train[0]))]
		for epoch in range(self.n_epoch):
			sum_error = 0
			for row in train:
				yhat = self.predict_row(row, coef)
				error = yhat - row[-1]
				sum_error += error*error 
				coef[0] = coef[0] - self.l_rate * error
				for i in range(len(row)-1):
					coef[i + 1] = coef[i + 1] - self.l_rate * error * row[i]

			#print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.l_rate, sum_error))
		
		return coef

clf = LogisticRegression()
clf.fit(train)
actual = [each[-1] for each in test]
pred = clf.predict([each[:-1] for each in test])
score = clf.score(actual, pred)
print(score)
