from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
import scipy.optimize as opt

data = loadtxt("ex2data1.txt", delimiter=',')

X = data[:, 0:2]

Y = data[:, 2]

pos = where(Y == 1)

neg = where(Y == 0)

scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')

show()

def sigmoid(X):
	denominator = 1.0 + e ** (-1.0 * X)
	d = 1.0 / denominator
	return d

def logistic_cost(X, Y, theta):
	m = X.shape[0]
	theta = reshape(theta, (len(theta), 1))

	J = (1.0/m)*(-transpose(y).dot(log(sigmoid(X.dot(theta)))) - transpose(1-y).dot(log(1-sigmoid(X.dot(theta)))))
	return J

def gradient(X, Y, theta):
	h = sigmoid(numpy.dot(X, theta))
	error = h - Y

	grad = numpy.dot(error, X)/y.size
	return grad

#using bfgs_min function from numpy

theta = 0.1 * numpy.random.randn(3)
X_1 = numpy.append(numpy.ones(X.shape[0], 1), X, axis=1)
theta_1 = opt.fmin_bfgs(cost, theta, fprime=grad, args=(X_1, y))


	






