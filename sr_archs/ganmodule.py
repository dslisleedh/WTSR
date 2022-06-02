import tensorflow as tf
from einops import rearrange
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, List


@tf.function
def tf_compute_critic_loss(true_logit, fake_logit):
    loss = tf.reduce_mean(
        fake_logit
    ) - tf.reduce_mean(
        true_logit
    )
    return loss


@tf.function
def tf_compute_gen_loss(fake_logit):
    loss = -tf.reduce_mean(
        fake_logit
    )
    return loss


def flax_compute_critic_loss(true_logit, fake_logit):
    loss = jnp.mean(
        fake_logit
    ) - tf.reduce_mean(
        true_logit
    )
    return loss


def flax_compute_gen_loss(fake_logit):
    loss = -jnp.mean(
        fake_logit
    )
    return loss


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
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(filters=n_filters,
                                   kernel_size=(kernel_size, kernel_size),
                                   strides=(strides, strides),
                                   padding=padding,
                                   activation='linear',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02)
                                   ),
            tf.keras.layers.LeakyReLU(.2)
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class Critic(tf.keras.layers.Layer):
    def __init__(self, dims: int = 32):
        super(Critic, self).__init__()
        self.dims = dims

        self.forward = tf.keras.Sequential([
            # 48
            tf.keras.layers.Conv2D(filters=self.dims // 2,
                                   kernel_size=(3, 3),
                                   padding='same',
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
            tf.keras.layers.Conv2D(1,
                                   kernel_size=(3, 3),
                                   padding='SAME',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=.02)
                                   )
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class FlaxCritic(nn.Module):
    n_filters: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.n_filters // 2,
                    kernel_size=(3, 3),
                    padding='SAME'
                    )
        x = jax.nn.leaky_relu(x, negative_slope=.2)
        for i in range(3):
            x = nn.LayerNorm()(x)
            x = nn.Conv(self.n_filters * (2 ** i),
                        kernel_size=(3, 3),
                        padding='SAME'
                        )(x)
            x = jax.nn.leaky_relu(x, negative_slope=.2)
        x = nn.Conv(1,
                    kernel_size=(3, 3),
                    padding='SAME'
                    )
        return x