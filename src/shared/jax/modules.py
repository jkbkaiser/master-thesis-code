import flax.nnx as nnx
from jax import lax


class Unfold(nnx.Module):
    def __init__(self, kernel_size: tuple[int, int], padding: tuple[int, int], stride: tuple[int, int]):
        self.kernel_size = kernel_size
        self.padding = [
            (padding[0], padding[0]),
            (padding[1], padding[1]),
        ]
        self.stride = stride

    def __call__(self, x):
        patches = lax.conv_general_dilated_patches(
            lhs=x,
            filter_shape=self.kernel_size,
            window_strides=self.stride,
            padding=self.padding,
        )
        flattened = patches.reshape(patches.shape[0], patches.shape[1], -1)
        return flattened
