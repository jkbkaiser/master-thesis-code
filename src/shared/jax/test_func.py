import jax
import jax.numpy as jnp

import src.shared.jax.func as fnc


def test_relu():
    arr = jax.random.uniform(jax.random.key(seed=42), (10, 5))
    out = fnc.relu(arr)
    comp = jax.nn.relu(arr)

    assert jnp.allclose(out, comp, rtol=0.0001)


def test_sigmoid():
    arr = jax.random.uniform(jax.random.key(seed=42), (10, 5))
    out = fnc.sigmoid(arr)
    comp = jax.nn.sigmoid(arr)

    assert jnp.allclose(out, comp, rtol=0.0001)


def test_softmax():
    arr = jax.random.uniform(jax.random.key(seed=42), (10, 5))
    out = fnc.softmax(arr, axis=1)
    comp = jax.nn.softmax(arr, axis=1)

    assert jnp.allclose(1, out.sum(axis=1), rtol=0.005)
    assert jnp.allclose(out, comp, rtol=0.0001)


def test_log_softmax():
    arr = jax.random.uniform(jax.random.key(seed=42), (4, 2))

    out = fnc.log_softmax(arr, axis=1)
    comp = jax.nn.log_softmax(arr, axis=1)

    assert jnp.allclose(out, comp, rtol=0.0001)
