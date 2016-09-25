from numpy import loadtxt
from numpy import zeros
from numpy import ones
from numpy import array
from numpy import linspace
from numpy import logspace
from pylab import scatter
from pylab import show
from pylab import title
from pylab import xlabel
from pylab import ylabel
from pylab import plot
from pylab import contour

#loading the data
data = loadtxt("ex1data1.txt", delimiter=',')

#plotting the data

scatter(data[:, 0], data[:, 1], marker='o', c='b')
title('Profits distribution')
xlabel('Population of City in 10,000\'s')
ylabel('Profit in 10,000\'s')
show()

#implementing gradient descent

X = data[:, 0]
Y = data[:, 1]

#to find number of training examples
m = Y.size

# adding a column of ones to accomodate for theta zero

it = ones(shape=(m, 2))
it[:, 1] = X
theta = zeros(shape=(2,1))

iterations = 1500
alpha = 0.01


def costFunction(X, Y, theta):
	m = Y.size
	prediction = X.dot(theta).flatten()

	sqerrors = (prediction - Y) ** 2

	cost = (1.0 / (2 * m)) * sqerrors.sum()

	return cost

def gradient_descent(X, Y, theta, alpha, num_iterations):
	m = Y.size
	print(X.shape)
	costHistory = zeros(shape=(num_iterations, 1))

	for i in range(0, num_iterations):
		prediction = X.dot(theta).flatten()

		errors_x1 = (prediction - Y) * X[:, 0]
		errors_x2 = (prediction - Y) * X[:, 1]

		theta[0][0] = theta[0][0] - alpha*(1.0/m)*errors_x1.sum()
		theta[1][0] = theta[1][0] - alpha*(1.0/m)*errors_x2.sum()

		costHistory[i, 0] = costFunction(X, Y, theta)
		print(costHistory[i, 0])

	return theta, costHistory	

gradient_descent(it, Y, theta, alpha, iterations)



