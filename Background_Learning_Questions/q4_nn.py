import jax
import jax.random as random
import jax.numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def f(x):
    '''Function we want to train our ANN for'''
    return np.exp(-10*np.abs(x))
    #return x

def generateData(size, fileName):
    '''Generate new training data using the function f'''
    key = random.PRNGKey(int(time.time()))

    xData = random.uniform(key, shape=(size,1), minval=-1, maxval=1)
    yData = f(xData)
    
    data = np.concatenate((xData, yData), axis=1)
    df = pd.DataFrame(data=data, columns=['x', 'y'])

    df.to_pickle('./pickle_files/' + fileName + '.pkl')

def initializeParam(layers):
    '''Initilaze the weights and bias of each neuron in the network's layers'''
    key = random.PRNGKey(int(time.time()))
    keys = random.split(key) 

    # create the first layer
    weights = np.array(random.uniform(keys[0], shape=(layers[0],1), minval=-1, maxval=1))
    bias = np.array(random.uniform(keys[1], shape=(layers[0],1), minval=-1, maxval=1))
    
    # create a list of all the weights and biases as a tuple for each layer
    allWeights = [weights]
    allBias = [bias]

    # go through each layer after the first
    for layer in range(len(layers)-1):
        # generate weights based on the number of neurons in previous layer
        weights = random.uniform(key, shape=(layers[layer+1],layers[layer]), minval=0, maxval=1)
        bias = np.array(random.uniform(key, shape=(layers[layer+1],1), minval=-1, maxval=1))
        
        allWeights.append(weights)
        allBias.append(bias)
    
    #return param
    return allWeights, allBias

def ReLU(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2) + 1)

def forwardPass(W, b, x):
    '''Go through the network once'''
    values = x
    
    # go through each layer
    for i in range(len(W)):
        #values = sigmoid(np.dot(W[i], values) + b[i])
        values = np.dot(W[i], values) + b[i]
        if (i != len(W)-1):
            #values = np.apply_along_axis(sigmoid, 0, values)
            #values = np.apply_along_axis(ReLU, 1, values)
            values = ReLU(values)
    
    # give back the predicted value
    return values[0][0]
    

def loss(pred, actual):
    '''Calculate the residual with a loss function'''
    return np.power(pred-actual, 2)

def predict(W, b, x, y):
    '''Apply the network to the input data and compute the loss'''
    pred = forwardPass(W, b, x)
    res = loss(pred, y)
    return res

def backwardPass(W_grads, b_grads, W, b, lr):
    '''Update weights and bias so that the network learns'''
    for i in range(len(W)):
        W[i] = W[i] - lr * W_grads[i]
        b[i] = b[i] - lr * b_grads[i]
    
fileName = 'q4_dataSet'

# get data
generateData(10000, fileName)
df = pd.read_pickle('./pickle_files/' + fileName + '.pkl')
xTrain = df['x'].tolist()
yTrain = df['y'].tolist()

# create ANN
layers = np.array([10,5,1])
W, b = initializeParam(layers)

# train the ANN
learningRate = 0.01
for i in range(len(xTrain)):
    # forward propagate
    W_grads, b_grads = jax.grad(predict, (0,1))(W, b, np.array(xTrain[i]), np.array(yTrain[i]))
    
    # backward propagate
    backwardPass(W_grads, b_grads, W, b, learningRate)
    

    if (i % 100 == 0):
        print('loss: ', predict(W, b, np.array(xTrain[i]), np.array(yTrain[i])))
        print('guess: ' + str(forwardPass(W, b, np.array(0))))

    if (i==9900):
        break
 
yPred = []
xPred = []
for i in range(len(xTrain)):
    if (i>9900 and i < 10000):
        yPred.append(forwardPass(W, b, xTrain[i]))
        xPred.append(xTrain[i])

print(len(yPred))
print('guess: ' + str(forwardPass(W, b, np.array(0))))
    
# test the ANN
plt.scatter(xTrain, yTrain)
plt.scatter(xPred, yPred)
plt.show()






