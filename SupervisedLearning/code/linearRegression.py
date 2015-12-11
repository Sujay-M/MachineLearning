import numpy as np

def input_data():
	n,m = raw_input().split(' ')
	n = int(n)
	m = int(m)
	X_train = np.empty((m,n))
	y = np.empty((m,1))
	for i in range(m):
		l = raw_input().split(' ')
		for j in range(n):
			X_train[i][j] = float(l[j])
		y[i] = float(l[n])
	p = int(raw_input())
	X_predict = np.empty((p,n))
	for i in range(p):
		l = raw_input().split(' ')
		for j in range(n):
			X_predict[i][j] = float(l[j])
	return n,m,X_train,y,p,X_predict

def cost(X,y,theta):
	h = np.dot(theta.T,X.T).T
	c = (h-y)**2
	return np.sum(c)

def gradientDescent(X,y,theta,m,iterations,alpha):
	for x in xrange(1,iterations+1):
		h = np.dot(theta.T,X.T).T
		gradient = np.dot((h-y).T,X).T
        theta = theta - gradient*alpha
        # print 'iteration = {} cost = {}'.format(x,cost(X,y,theta))
	return theta

def addColumn(A):
	r,c = A.shape
	B = np.ones((r,1),dtype=np.float64)
	return np.append(B, A, axis=1)

#get data from standard input
n,m,X_train,y,p,X_predict = input_data()

#initialize theta vector to 1 with a theta0 component
theta = np.ones((n+1,1),dtype=np.float64)

#add x0 column with all ones
X_train = addColumn(X_train)

#choose number of iterations and alpha
alpha = .1; iterations = 250

#apply gradient descent to find theta
theta = gradientDescent(X_train,y,theta,m,iterations,alpha)

#now theta is found, do the predictions
X_predict = addColumn(X_predict)
predictions = np.dot(theta.T,X_predict.T).T

for val in predictions:
    print val[0]