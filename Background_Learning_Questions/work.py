import jax as jnp
import numpy as np
import time

# model parameters
s = 20
m = 100000
x = 1

key = jnp.random.PRNGKey(int(time.time()))
dicerolls = jnp.random.randint(key, shape=(m,), minval=1, maxval=s+1)/(s+1)

partialSum = 0
N = 0
sumTimes = 0
iterations = 0

for i in range(len(dicerolls)):
    if (partialSum <= x):
        partialSum += dicerolls[i]
        sumTimes += 1
    else:
        N += sumTimes
        sumTimes = 0
        partialSum = 0
        
        iterations += 1

approx = N/iterations
print("approximation of e: " + str(approx))
