import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt

dt = 1
dx = 1
n = 1000000

# part 1
key = jax.random.PRNGKey(int(time.time()))

X = jax.random.choice(key, jnp.array([1,-1]), shape=(n,))

W = jnp.zeros(shape=())

W = jnp.append(W, X)

W = jnp.cumsum(W)

# part 2a
a = jax.random.uniform(key, minval=10, maxval=100)*(-1)
b = jax.random.uniform(key, minval=10, maxval=100)

print(np.where(np.logical_or(W > b,W < a))[0][0])

#plt.plot(jnp.arange(0,len(W)*dt, step=dt), W)
#plt.xlabel("time")
#plt.ylabel("W(t)")
#plt.show()
