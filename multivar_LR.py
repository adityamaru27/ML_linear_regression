def normalizeData(X):
	nColumns = X.size[1]
	mean_r = []
	sd_r = []

	X_norm = X

	for i in range(nColumns):
		m = mean(x[:, i])
		s = std(x[:, i])

		mean_r.append(m)
		sd_r.append(s)

		X_norm = (X_norm[:, i] - mean) / sd

def costFunction(X, Y, theta):
	m = Y.size

	predictions = X.dot(theta)
	error = predictions - Y

	J_cost = (1/2*m)*error.T.dot(error)


def gradientDescent(X, Y, theta, iterations, alpha):
	cost_history = zeros(shape=(iterations, 1))
	for i in range(iterations):
		for a in range(theta.size):
			temp = X[:, a]
			temp.shape = (m, 1)

			predictions = X.dot(theta)
			error = (predictions - Y) * temp

			theta[a][0] = theta[a][0] - alpha*(1/m)*error.sum()

		cost_history[i][0] = costFunction(X, Y, theta)









