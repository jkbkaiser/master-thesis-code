import equinox as eqx
import jax
import jax.numpy as jnp

import src.shared.jax.modules as mod


def test_linear():
    key = jax.random.key(seed=42)

    linear_layer = mod.Linear(in_shape=3, out_shape=2, key=key)

    new_weight = jnp.arange(6).reshape((2, 3))
    linear_layer = eqx.tree_at(lambda tree: tree.weight, linear_layer, new_weight)

    new_bias = jnp.ones((2,))
    linear_layer = eqx.tree_at(lambda tree: tree.bias, linear_layer, new_bias)

    inp = jnp.arange(3)
    out = linear_layer(inp)

    assert jnp.all(jnp.equal(out, jnp.array([6, 15])))
