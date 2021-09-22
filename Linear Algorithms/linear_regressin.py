import sys
sys.path.append("C:/Users/gyon/Desktop/ML _ SRC/Data Preparation")
import baseline_prediction as bp
import utils
import test_harness

dataset = utils.load_csv_file('datasets/swedish.csv', dtype=float)

def column(col, dataset):	
	return [row[col] for row in dataset]

class LinearRegression:
	def __init__(self):
		self.fitted = False

	def fit(self, train_X, train_y):
		self.b0, self.b1 = self.coefficients(train_X, train_y)
		self.fitted = True

	def predict(self, test_X):
		if self.fitted:
			return [(self.b0 + self.b1*x) for x in test_X]		

	def rmse(self, actual, predicted):
		return utils.root_mean_squared_error(actual, predicted)

	def coefficients(self, X, Y):
		b1 = self.covariance(X, Y) / self.variance(X)
		b0 = self.mean(Y) - b1*self.mean(X)

		return b0, b1

	def covariance(self, X, Y):
		mean_x = self.mean(X)
		mean_y = self.mean(Y)
		return sum([(x-mean_x)*(y-mean_y) for x, y in zip(X, Y)])	


	# mean
	def mean(self, values):
		return sum(values) / float(len(values))

	# variance
	def variance(self, values):
		mean_x = self.mean(values)
		return sum([(x-mean_x)**2 for x in values])



def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]

	return yhat

dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
coef = [0.4, 0.8]

for row in dataset:
	yhat = predict(row, coef)
	print(f"Expected: {row[-1]}, Predicted: {yhat}")


dataset = utils.load_csv_file('datasets/winequality.csv', dtype=float)
train, test = utils.train_test_split(dataset)
