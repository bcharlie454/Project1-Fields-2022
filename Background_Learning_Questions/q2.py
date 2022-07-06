import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt

dt = 0.2
dx = 1
n = 1000000

key = jax.random.PRNGKey(int(time.time()))

X = jax.random.choice(key, jnp.array([1,-1]), shape=(n,))

W = jnp.zeros(shape=())

W = jnp.append(W, X)

W = jnp.cumsum(W)

print(W[-1])

plt.plot(jnp.arange(0,len(W)*dt, step=dt), W)
plt.xlabel("time")
plt.ylabel("W(t)")
plt.show()
