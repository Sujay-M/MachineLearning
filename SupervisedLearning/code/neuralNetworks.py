import numpy as np

class NeuralNetwork:
    
    def __init__(self,name,nFeatures,nHiddenLayers,nParams,k,epsilon):
        
        self.nFeatures = [nFeatures]+nParams+[k]
        self.nHiddenLayers = nHiddenLayers
        self.k = k
        self.name = name
        self.theta = [(np.random.rand(self.nFeatures[i-1]+1,self.nFeatures[i])*2*epsilon-epsilon) for i in range(1,nHiddenLayers+2)]

    def logit(self,z):

        return 1/(1+np.exp(-z))
        
    
    def forwardPropogation(self,x):
        
        
        activation = np.append([1],x)
        result = [activation]
        for i in xrange(self.nHiddenLayers+1):
            t = self.theta[i]
            activation = np.append([1],self.logit((t.T.dot(activation.T)).T))
            result.append(activation)
        return result
        
    def backPropogation(self,x,y):
        
        DELTA = []
        activations = self.forwardPropogation(x)
        delta = activations[-1][1:]-y
        DELTA.append(delta)
        index = self.nHiddenLayers
        while index>0:
            delta = self.theta[index][1:,:].dot(delta)*(activations[index][1:]*(1-activations[index][1:]))
            DELTA.append(delta)
            index -= 1
        return DELTA,activations

    def predict(self,x):
        
        result = self.forwardPropogation(x)
        return np.argmax(result[-1][1:])
    
    def train(self,X,Y,iterations,alpha):
        
        m,_ = X.shape
        print 'm = ',m 
        for i in xrange(iterations):
            print 'iteration = ',i
            DELTA = [np.zeros(self.theta[k].shape) for k in range(len(self.theta))]
            for x,y in zip(X,Y):
                delta,activations = self.backPropogation(x,y)
                for j in xrange(self.nHiddenLayers+1):
                    DELTA[j] += np.array([activations[j]]).T.dot(np.array([delta[self.nHiddenLayers-j]]))
            for j in xrange(len(DELTA)):
                self.theta[j] = self.theta[j] - alpha*DELTA[j]/m

    def loadConfiguration():

        return

    def saveConfiguration():

        return
        
