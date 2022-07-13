import tensorflow as tf
import tensorflow.keras.backend as K
from sr_archs.model_utils import PixelShuffle
from typing import List


def edge_padding2d(x, h_pad, w_pad):
    if h_pad[0] != 0:
        x_up = tf.gather(x, indices=[0], axis=1)
        x_up = tf.concat([x_up for _ in range(h_pad[0])], axis=1)
        x = tf.concat([x_up, x], axis=1)
    if h_pad[1] != 0:
        x_down = tf.gather(tf.reverse(x, axis=[1]), indices=[0], axis=1)
        x_down = tf.concat([x_down for _ in range(h_pad[1])], axis=1)
        x = tf.concat([x, x_down], axis=1)
    if w_pad[0] != 0:
        x_left = tf.gather(x, indices=[0], axis=2)
        x_left = tf.concat([x_left for _ in range(w_pad[0])], axis=2)
        x = tf.concat([x_left, x], axis=2)
    if w_pad[1] != 0:
        x_right = tf.gather(tf.reverse(x, axis=[2]), indices=[0], axis=2)
        x_right = tf.concat([x_right for _ in range(w_pad[1])], axis=2)
        x = tf.concat([x, x_right], axis=2)
    return x


class LocalAvgPool2D(tf.keras.layers.Layer):
    def __init__(self,
                 local_size: List[int]
                 ):
        super(LocalAvgPool2D, self).__init__()
        self.local_size = local_size

    def call(self, inputs, *args, **kwargs):
        if K.learning_phase():
            return tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)

        _, h, w, _ = inputs.get_shape().as_list()
        kh = min(h, self.local_size[0])
        kw = min(w, self.local_size[1])
        inputs = tf.pad(inputs,
                        [[0, 0],
                         [1, 0],
                         [1, 0],
                         [0, 0]]
                        )
        inputs = tf.cumsum(tf.cumsum(inputs, axis=2), axis=1)
        s1 = tf.slice(inputs,
                      [0, 0, 0, 0],
                      [-1, kh, kw, -1]
                      )
        s2 = tf.slice(inputs,
                      [0, 0, w - kw + 1, 0],
                      [-1, kw, -1, -1]
                      )
        s3 = tf.slice(inputs,
                      [0, h - kh + 1, 0, 0],
                      [-1, -1, kw, -1]
                      )
        s4 = tf.slice(inputs,
                      [0, h - kh + 1, w - kw + 1, 0],
                      [-1, -1, -1, -1]
                      )
        local_gap = (s4 + s1 - s2 - s3) / (kh * kw)

        _, h_, w_, _ = local_gap.get_shape().as_list()
        h_pad, w_pad = [(h - h_) // 2, (h - h_ + 1) // 2], [(w - w_) // 2, (w - w_ + 1) // 2]
        local_gap = edge_padding2d(local_gap, h_pad, w_pad)
        return local_gap

    def get_config(self):
        config = super(LocalAvgPool2D, self).get_config()
        config.update({
            'local_size': self.local_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SimpleGate(tf.keras.layers.Layer):
    def __init__(self):
        super(SimpleGate, self).__init__()

    def call(self, inputs, *args, **kwargs):
        x1, x2 = tf.split(inputs,
                          num_or_size_splits=2,
                          axis=-1
                          )
        return x1 * x2

    def get_config(self):
        config = super(SimpleGate, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SimpleChannelAttention(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 kh: int,
                 kw: int
                 ):
        super(SimpleChannelAttention, self).__init__()
        self.n_filters = n_filters
        self.kh = kh
        self.kw = kw

        self.pool = LocalAvgPool2D([kw, kw])
        self.w = tf.keras.layers.Dense(self.n_filters,
                                       activation=None,
                                       kernel_initializer=tf.keras.initializers.VarianceScaling(scale=.02)
                                       )

    def call(self, inputs, *args, **kwargs):
        attention = self.pool(inputs)
        attention = self.w(attention)
        return attention * inputs

    def get_config(self):
        config = super(SimpleChannelAttention, self).get_config()
        config.update({
            'n_filters': self.n_filters,
            'kh': self.kh,
            'kw': self.kw
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NAFBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 dropout_rate: float,
                 kh: int,
                 kw: int,
                 dw_expansion: int = 2,
                 ffn_expansion: int = 2
                 ):
        super(NAFBlock, self).__init__()

        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.kh = kh
        self.kw = kw
        self.dw_filters = n_filters * dw_expansion
        self.ffn_filters = n_filters * ffn_expansion

        self.spatial = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(self.dw_filters,
                                   kernel_size=1,
                                   strides=1,
                                   padding='VALID',
                                   activation=None,
                                   kernel_initializer=tf.keras.initializers.VarianceScaling(scale=.02)
                                   ),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                            strides=1,
                                            padding='SAME',
                                            activation=None,
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=.02)
                                            ),
            SimpleGate(),
            SimpleChannelAttention(self.n_filters,
                                   self.kh,
                                   self.kw
                                   ),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=1,
                                   strides=1,
                                   padding='VALID',
                                   activation=None,
                                   kernel_initializer=tf.keras.initializers.VarianceScaling(scale=.02)
                                   )
        ])
        self.drop1 = tf.keras.layers.Dropout(self.dropout_rate)

        self.channel = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.ffn_filters,
                                  activation=None,
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=.02)
                                  ),
            SimpleGate(),
            tf.keras.layers.Dense(self.n_filters,
                                  activation=None,
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=.02)
                                  )
        ])
        self.drop2 = tf.keras.layers.Dropout(self.dropout_rate)

        self.beta = tf.Variable(
            tf.zeros((1, 1, 1, self.n_filters)),
            trainable=True,
            dtype=tf.float32,
            name='beta'
        )
        self.gamma = tf.Variable(
            tf.zeros((1, 1, 1, self.n_filters)),
            trainable=True,
            dtype=tf.float32,
            name='gamma'
        )

    def call(self, inputs, *args, **kwargs):
        inputs += self.drop1(self.spatial(inputs)) * self.beta
        inputs += self.drop2(self.channel(inputs)) * self.gamma
        return inputs

    def get_config(self):
        config = super(NAFBlock, self).get_config()
        config.update({
            'n_filters': self.n_filters,
            'dropout_rate': self.dropout_rate,
            'kh': self.kh,
            'kw': self.kw,
            'dw_filters': self.dw_filters,
            'ffn_filters': self.ffn_filters
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DropPath(tf.keras.layers.Layer):
    def __init__(self,
                 survival_prob: float,
                 module: tf.keras.layers.Layer
                 ):
        super(DropPath, self).__init__()
        self.survival_prob = survival_prob
        self.forward = module

    def call(self, inputs, *args, **kwargs):

        def _call_train():
            return tf.cond(
                tf.less(tf.random.uniform(shape=[], minval=0., maxval=1.), self.survival_prob),
                lambda: inputs + ((self.forward(inputs) - inputs) / self.survival_prob),
                lambda: inputs
            )

        def _call_test():
            return self.forward(inputs)

        return K.in_train_phase(
            _call_train(),
            _call_test()
        )

    def get_config(self):
        config = super(DropPath, self).get_config()
        config.update({'survival_prob': self.survival_prob})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


### NAFSSR(https://arxiv.org/abs/2204.08714, https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFSSR_arch.py)
class NAFNetSR(tf.keras.models.Model):
    def __init__(
            self,
            upscale_rate: int,
            n_filters: int,
            n_blocks: int,
            stochastic_depth_rate: float,
            train_size: List,
            tlsc_rate: float = 1.5
    ):
        super(NAFNetSR, self).__init__()
        self.upscale_rate = upscale_rate
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.stochastic_depth_rate = stochastic_depth_rate
        self.train_size = train_size
        self.tlsc_rate = tlsc_rate

        self.kh, self.kw = int(train_size[1] * tlsc_rate), int(train_size[2] * tlsc_rate)

        self.intro = tf.keras.layers.Conv2D(
            self.n_filters,
            kernel_size=(3, 3),
            padding='SAME',
            strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=.02)
        )
        self.middles = tf.keras.Sequential([
            DropPath(
                1. - self.stochastic_depth_rate,
                NAFBlock(
                    self.n_filters,
                    0.,
                    self.kh,
                    self.kw
                )
            ) for _ in range(self.n_blocks)
        ])
        self.upscale = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.upscale_rate ** 2,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=.02)
            ),
            PixelShuffle(self.upscale_rate)
        ])

    def forward(self, x, training=False):
        _, h, w, _ = x.get_shape().as_list()
        x_skip = tf.image.resize(
           x, size=[int(h * self.upscale_rate), int(w * self.upscale_rate)],
           method=tf.image.ResizeMethod.BICUBIC
        )
        features = self.intro(x, training=training)
        features = features + self.middles(features, training=training)
        recon = self.upscale(features, training=training)
        recon += x_skip
        return recon

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = False
        return self.forward(inputs, training=training)

    def get_config(self):
        config = super(NAFNetSR, self).get_config()
        config.update(
            {'upscale_rate': self.upscale_rate,
             'n_filters': self.n_filters,
             'n_blocks': self.n_blocks,
             'stochastic_depth_rate': self.stochastic_depth_rate,
             'train_size': self.train_size,
             'tlsc_rate': self.tlsc_rate
             }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
