import jax
import jax.random as random
import jax.numpy as np
import pandas as pd
import time

def f(x):
    '''Function we want to train our ANN for'''
    return np.exp(-10*np.abs(x))

def generateData(size, fileName):
    '''Generate new training data using the function f'''
    key = random.PRNGKey(int(time.time()))

    xData = random.uniform(key, shape=(size,1), minval=-0.5, maxval=0.5)
    #xData = random.normal(key, shape=(size,1))
    yData = f(xData)
    
    data = np.concatenate((xData, yData), axis=1)
    df = pd.DataFrame(data=data, columns=['x', 'y'])

    df.to_pickle('./pickle_files/' + fileName + '.pkl')


def ReLU(x):
    return np.maximum(0,x)

def initializeParam(layers):
    '''Initilaze the weights and bias of each neuron in the network's layers'''
    key = random.PRNGKey(int(time.time()))
    
    # create the first layer
    weights = np.array(random.uniform(key, shape=(layers[0],1), minval=-1, maxval=1))
    bias = np.array(random.uniform(key, shape=(layers[0],1), minval=-1, maxval=1))
    
    # create a list of all the weights and biases as a tuple for each layer
    allWeights = [weights]
    allBias = [bias]

    # go through each layer after the first
    for layer in range(len(layers)-1):
        # generate weights based on the number of neurons in previous layer
        weights = random.uniform(key, shape=(layers[layer+1],layers[layer]), minval=-1, maxval=1)
        bias = np.array(random.uniform(key, shape=(layers[layer+1],1), minval=-1, maxval=1))
        
        allWeights.append(weights)
        allBias.append(bias)
    
    #return param
    return allWeights, allBias

def loss(pred, actual):
    '''Calculate the residual with a loss function'''
    return np.power(pred - actual, 2).mean()

def forwardPass(W, b, x):
    '''Go through the network once'''
    values = x

    # go through each layer
    for i in range(len(W)):
        values = np.dot(W[i], values)
        values = values + b[i]
        values = np.apply_along_axis(ReLU, 0, values)
    
    # give back the predicted value
    return values[0][0]

def predict(W, b, x, y):
    '''Apply the network to the input data and compute the loss'''
    pred = forwardPass(W, b, x)
    res = loss(pred, y)
    return res

def backwardPass(W_grad, b_grad, W, b):
    '''Update weights and bias so that the network learns'''
    for i in range(len(W)):
        W[i] -= np.multiply(W[i], W_grad[i])
        b[i] -= np.multiply(b[i], b_grad[i])

    return W, b
    
fileName = 'q4_dataSet'

# get data
generateData(100, fileName)
df = pd.read_pickle('./pickle_files/' + fileName + '.pkl')
xTrain = df['x'].tolist()
yTrain = df['y'].tolist()

# create ANN
layers = np.array([3, 1])
W, b = initializeParam(layers)

# train the ANN
for i in range(len(xTrain)):
    # forward propagate
    W_grad, b_grad = jax.grad(predict, (0, 1))(W, b, np.array(xTrain[i]), np.array(yTrain[i]))
    
    # backward propagate
    W, b = backwardPass(W_grad, b_grad, W, b)
    
# test the ANN
print(predict(W, b, np.array(0), 1))







