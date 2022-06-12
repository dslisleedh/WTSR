import tensorflow as tf
import tensorflow_addons as tfa
from einops import rearrange
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, List


class InstanceNorm2D(nn.Module):

    @nn.compact
    def __call__(self, x):
        shape = x.shape
        x = nn.GroupNorm(num_groups=shape[-1])(x)
        return x


class DownsamplingConv2D(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters,
            kernel_size=3,
            strides=2,
            padding='SAME'
    ):
        super(DownsamplingConv2D, self).__init__()

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=n_filters,
                                   kernel_size=(kernel_size, kernel_size),
                                   strides=(strides, strides),
                                   padding=padding,
                                   activation='linear',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02),
                                   use_bias=False
                                   ),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.LeakyReLU(.2)
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class TFCritic(tf.keras.layers.Layer):
    def __init__(self, dims: int = 32):
        super(TFCritic, self).__init__()
        self.dims = dims

        self.forward = tf.keras.Sequential([
            # 50
            tf.keras.layers.Conv2D(filters=self.dims // 2,
                                   kernel_size=(5, 5),
                                   padding='VALID',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=.02)
                                   ),
            tf.keras.layers.LeakyReLU(.2),
            # 48
            DownsamplingConv2D(self.dims),
            # 24
            DownsamplingConv2D(self.dims * 2),
            # 12
            DownsamplingConv2D(self.dims * 4),
            # 6
            DownsamplingConv2D(self.dims * 8),
            # 3
            tf.keras.layers.Conv2D(1,
                                   kernel_size=(3, 3),
                                   padding='VALID',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=.02)
                                   ),
            #1
            tf.keras.layers.Flatten()
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class FlaxCritic(nn.Module):
    n_filters: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.n_filters // 2,
                    kernel_size=(3, 3),
                    padding='VALID'
                    )(x)
        x = jax.nn.leaky_relu(x, negative_slope=.2)
        for i in range(4):
            x = nn.Conv(self.n_filters * (2 ** i),
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='SAME'
                        )(x)
            x = InstanceNorm2D()(x)
            x = jax.nn.leaky_relu(x, negative_slope=.2)
        x = nn.Conv(1,
                    kernel_size=(3, 3),
                    padding='VALID'
                    )(x)
        x = jnp.squeeze(x)
        return x
