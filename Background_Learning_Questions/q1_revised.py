import jax.numpy as jnp
import jax
import numpy as np
import time

# model parameters
s = 20
m = 100000000
x = 1


key = jax.random.PRNGKey(int(time.time()))

# m must be divisible by s
diceRolls = jax.random.randint(key, shape=(int(m/s),s), minval=1, maxval=s+1)
diceRolls = diceRolls/s - 1/(2*s)

#diceRolls = jax.random.uniform(key, shape=(int(m/s),s), minval=0, maxval=1)

diceRolls = jnp.cumsum(diceRolls, axis=1)

# randomly assigning ties does a really bad job
#diceRolls = jnp.where(diceRolls == x, jax.random.choice(key, jnp.array([0.5, 1.5])), diceRolls)
rows, cols = jnp.where(diceRolls == x)
tieCount = len(rows)
#diceRolls = jnp.delete(diceRolls, rows, axis=0)

diceRolls = jnp.where(diceRolls > x, 0, diceRolls)

rollCount = jnp.count_nonzero(diceRolls, axis=1) + 1

approx = (jnp.sum(rollCount) - tieCount/2)/(len(rollCount))
#approx = jnp.mean(rollCount)
print(approx)

