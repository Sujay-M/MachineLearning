import struct
import numpy as np

def getLabels(fname):
    with open(fname,'rb') as fL:
        fL.seek(4)
        m = struct.unpack('>i',fL.read(4))[0]
        y = np.empty((m,1),'int8')
        for i in xrange(m):
            y[i] = struct.unpack('>b',fL.read(1))
        return m,y

def getImages(fname):
    with open(fname,'rb') as fI:
        fI.seek(4)
        m = struct.unpack('>i',fI.read(4))[0]
        rows = struct.unpack('>i',fI.read(4))[0]
        cols = struct.unpack('>i',fI.read(4))[0]
        x = np.empty((m,rows,cols),'uint8')
        nPixels = rows*cols
        pat = '>' + 'B'*nPixels
        for i in xrange(m):
            x[i] = np.array(struct.unpack(pat,fI.read(nPixels))).reshape((rows,cols))
        return m,rows,cols,x

def getTrainData():
    fI = '../../../datasets/mnist/train-images.idx3-ubyte'
    fL = '../../../datasets/mnist/train-labels.idx1-ubyte'    
    m,rows,cols,x = getImages(fI)    
    _,y = getLabels(fL)    
    return m,rows,cols,x,y

def getTestData():
    fI = '../../datasets/mnist/t10k-images.idx3-ubyte'
    fL = '../../datasets/mnist/t10k-labels.idx1-ubyte'
    m,rows,cols,x = getImages(fI)    
    _,y = getLabels(fL)
    return m,rows,cols,x,y
