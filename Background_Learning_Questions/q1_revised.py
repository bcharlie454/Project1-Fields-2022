import jax as jnp
import numpy as np
import time

# model parameters
s = 20
m = 10000000
x = 1


key = jnp.random.PRNGKey(int(time.time()))

# m must be divisible by s
diceRolls = jnp.random.randint(key, shape=(int(m/s),s), minval=1, maxval=s+1)/(s+1)

diceRolls = jnp.numpy.cumsum(diceRolls, axis=1)

diceRolls = jnp.numpy.where(diceRolls > 1, 0, diceRolls)

rollCount = jnp.numpy.count_nonzero(diceRolls, axis=1) + 1

approx = jnp.numpy.mean(rollCount)

print(approx)

