import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_shape: int, out_shape: int, key: PRNGKeyArray):
        wkey, bkey = jax.random.split(key, num=2)
        self.weight = jax.random.uniform(wkey, (out_shape, in_shape))
        self.bias = jax.random.uniform(bkey, (out_shape,))

    def __call__(self, x):
        return self.weight @ x + self.bias
