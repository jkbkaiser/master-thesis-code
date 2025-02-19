import jax.numpy as jnp


def relu(x):
    return jnp.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def softmax(x, axis: int | tuple[int] = -1):
    x_exp = jnp.exp(x)
    x_exp_s = x_exp.sum(axis=axis, keepdims=True)
    return x_exp / x_exp_s


def log_softmax(x, axis: int | tuple[int] = -1):
    """
    Computes the logarithm of the softmax using the logsumexp trick, which
    can be derived as follows:

    p_i = exp(x_i) / sum(exp(x))
    log(p_i) = log[ exp(x_i) / sum(exp(x)) ]
             = log[ exp(x_i) ] - log [ sum(exp(x)) ]
             = x_i - log [ exp(x_j) + exp(x_m) + ... ]
             = x_i - log [ (exp(x_j) + exp(x_m) + ...) * exp(c) / exp(c) ]
             = x_i - log [ (exp(x_j - c) + exp(x_m - c) + ...) * exp(c) ]
             = x_i - (log [ (exp(x_j - c) + exp(x_m - c) + ...) ] + c)
             = x_i - c - log [ (exp(x_j - c) + exp(x_m - c) + ...) ]

    Here we choose c = max(x).
    """
    c = jnp.max(x, axis=axis, keepdims=True)
    x_norm = x - c
    return x_norm - jnp.log(jnp.exp(x_norm).sum(axis=axis, keepdims=True))
