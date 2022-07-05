import jax as jnp

key = jnp.random.PRNGKey(0)
x = jnp.random.normal(key, (10,))
print(x)
