import numpy as np
import mnistData

def hypothesis(xi,theta):
    h = np.exp(xi.T.dot(theta.T))    
    h = h/(np.sum(h)+1)
    return h
    

def gradientAcsent(X,y,theta,alpha,m):
    for i in xrange(m):
        indi = np.zeros(X[i].shape,dtype='float64')
        if y[i][0]<9:
            indi[y[i][0]] = 1.0
        h = hypothesis(X[i],theta)
        for j in xrange(k-1):
            gradient = indi[j]-h[j]
            gradient = alpha*(X[i]*gradient)/m
            theta[j] = theta[j]+gradient
    return theta

def likelihood(y,X,theta,m):
    l = 0.0
    for i in xrange(m):
        h = hypothesis(X[i],theta)
        if y[i][0]==k-1:
            l += 1-np.sum(h)
        else:
            l += h[y[i][0]]
    return l


def addColumn(A):
    r,c = A.shape
    B = np.ones((r,1),dtype=np.float64)
    return np.append(B, A, axis=1)

def predict(x,theta):
    m,_ = x.shape
    p = np.zeros((m,1),dtype='uint8')
    for i in xrange(m):
        h = hypothesis(x[i],theta)
        s = np.sum(h)
        l = np.argmax(h)
        if (1-s)<h[l]:
            p[i][0] = l
        else:
            p[i][0] = 9
    return p
        
        

m,rows,cols,xTrain,yTrain = mnistData.getTrainData()
mTest,_,_,xTest,yTest = mnistData.getTestData()
k = 10
xTrain = xTrain.reshape((m,rows*cols))
xTrain = addColumn(xTrain)
xTrain = xTrain/255
xTest = xTest.reshape((mTest,rows*cols))
xTest = addColumn(xTest)
xTest = xTest/255
theta = np.zeros((k-1,rows*cols+1))
iterations = 250
alpha = .2
for i in xrange(iterations):
    theta = gradientAcsent(xTrain,yTrain,theta,alpha,m)
    print 'iteration = {}'.format(i+1)
pTrain = predict(xTrain,theta)
print 'train error = {}'.format(np.sum(pTrain!=yTrain)/float(m))
pTest = predict(xTest,theta)
print 'test error = {}'.format(np.sum(pTest!=yTest)/float(mTest))

