import numpy as np
import mnistData
from neuralNetworks import NeuralNetwork

def createY(y,k,m):
    Y = np.zeros((m,k),dtype=int)
    temp = np.array(range(m))
    Y[temp,y[:,0]] = 1
    return Y

m,rows,cols,x_train,y = mnistData.getTrainData()
y_train = createY(y,10,m)
x_train = x_train.reshape((m,rows*cols))

mtest,rtest,ctest,x_test,y_t = mnistData.getTestData()
y_test = createY(y_t,10,mtest)
x_test = x_test.reshape((mtest,rtest*ctest))

nn = NeuralNetwork('mnist',rows*cols,1,[50],10,.1)
nn.train(x_train,y_train,250,.1)

count = 0
for x,y in zip(x_test,y_t):
    res = nn.predict(x)
    if res==y[0]:
        count+=1
print count
