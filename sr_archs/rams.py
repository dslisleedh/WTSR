import tensorflow as tf
import tensorflow_addons as tfa


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self,
                 scale
                 ):
        super(PixelShuffle, self).__init__()
        self.scale = scale

    def call(self, inputs, *args, **kwargs):
        return tf.nn.depth_to_space(inputs,
                                    block_size=self.scale
                                    )


class ReflectPadding(tf.keras.layers.Layer):
    def __init__(self,
                 n_pads: List[int]
                 ):
        super(ReflectPadding, self).__init__()
        if len(n_pads) == 2:
            self.pad = tf.constant([
                [0, 0],
                [n_pads[0], n_pads[0]],
                [n_pads[1], n_pads[1]],
                [0, 0]
            ])
        elif len(n_pads) == 3:
            self.pad = tf.constant([
                [0, 0],
                [n_pads[0], n_pads[0]],
                [n_pads[1], n_pads[1]],
                [n_pads[2], n_pads[2]],
                [0, 0]
            ])
        else:
            raise NotImplementedError('wrong pads')

    def call(self, inputs, *args, **kwargs):
        return tf.pad(inputs,
                      self.pad,
                      mode='REFLECT'
                      )


class Conv3DWeightedNorm(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 kernel_size: int = 3,
                 padding: str = 'SAME',
                 activation: str = 'linear'
                 ):
        super(Conv3DWeightedNorm, self).__init__()

        self.conv3d = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv3D(n_filters,
                                   kernel_size,
                                   padding=padding,
                                   strides=1,
                                   activation=activation,
                                   use_bias=False
                                   ),
            data_init=False
        )

    def call(self, inputs, *args, **kwargs):
        return self.conv3d(inputs)


class RFAB(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 r: int
                 ):
        super(RFAB, self).__init__()

        self.forward = tf.keras.Sequential([
            Conv3DWeightedNorm(n_filters),
            tf.keras.layers.ReLU(),
            Conv3DWeightedNorm(n_filters)
        ])
        self.attention = tf.keras.Sequential([
            tf.keras.layers.Dense(n_filters // r),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(n_filters)
        ])

    def call(self, inputs, *args, **kwargs):
        features = self.forward(inputs)
        attention = tf.reduce_mean(features,
                                   axis=[1, 2, 3],
                                   keepdims=True
                                   )
        attention = tf.nn.sigmoid(self.attention(attention))
        features = features * attention
        return inputs + features


class TRB(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 r
                 ):
        super(TRB, self).__init__()

        self.pad = ReflectPadding([1, 1, 0])
        self.rfab = RFAB(n_filters, r)
        self.reduction = tf.keras.Sequential([
            Conv3DWeightedNorm(n_filters, padding='VALID'),
            tf.keras.layers.ReLU()
        ])

    def call(self, inputs, *args, **kwargs):
        return self.reduction(self.rfab(self.pad(inputs)))


class Conv2DWeightedNorm(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 kernel_size: int = 3,
                 padding: str = 'SAME',
                 activation: str = 'linear'
                 ):
        super(Conv2DWeightedNorm, self).__init__()

        self.conv2d = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(n_filters,
                                   kernel_size,
                                   padding=padding,
                                   strides=1,
                                   activation=activation,
                                   use_bias=False
                                   ),
            data_init=False
        )

    def call(self, inputs, *args, **kwargs):
        return self.conv2d(inputs)


class RTAB(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 r: int
                 ):
        super(RTAB, self).__init__()

        self.forward = tf.keras.Sequential([
            Conv2DWeightedNorm(n_filters),
            tf.keras.layers.ReLU(),
            Conv2DWeightedNorm(n_filters)
        ])
        self.attention = tf.keras.Sequential([
            tf.keras.layers.Dense(n_filters // r),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(n_filters)
        ])

    def call(self, inputs, *args, **kwargs):
        features = self.forward(inputs)
        attention = tf.reduce_mean(features,
                                   axis=[1, 2],
                                   keepdims=True
                                   )
        attention = tf.nn.sigmoid(self.attention(attention))
        features = features * attention
        return inputs + features


class RAMs(tf.keras.models.Model):
    def __init__(self,
                 n_filters: int,
                 r: int,
                 scale: int,
                 n: int,
                 t: int
                 ):
        super(RAMs, self).__init__()
        self.scale = scale

        self.to_features = tf.keras.Sequential([
            ReflectPadding([1, 1, 0]),
            Conv3DWeightedNorm(n_filters)
        ])
        self.rfabs = tf.keras.Sequential([
            RFAB(n_filters, r) for _ in range(n)
        ] + [
            Conv3DWeightedNorm(n_filters)
        ])
        self.trb = tf.keras.Sequential([
            TRB(n_filters, r) for _ in range(((t - 1) // 2) - 1)
        ] + [
            Conv3DWeightedNorm(n_filters ** 2,
                               kernel_size=3,
                               padding='VALID'
                               ),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3)),
            PixelShuffle(scale)
        ])
        self.rtabs = tf.keras.Sequential([
            ReflectPadding([1, 1]),
            RTAB(t, r),
            Conv2DWeightedNorm(n_filters ** 2,
                               padding='VALID'
                               ),
            PixelShuffle(scale)
        ])
        self.recon = tf.keras.Sequential([
            tf.keras.layers.Conv2D(n_filters,
                                   3,
                                   1,
                                   padding='SAME'
                                   ),
            tf.keras.layers.Conv2D(1,
                                   1,
                                   padding='VALID'
                                   )
        ])

    @tf.function
    def forward(self, x):
        skip = tf.expand_dims(tf.gather(x, -1, axis=-1), axis=-1)
        _, H, W, _ = tf.shape(skip)
        features = self.to_features(tf.expand_dims(x, axis=-1))
        features = features + self.rfabs(features)
        features = self.trb(features)
        global_residual = self.rtabs(x)
        recon = self.recon(features + global_residual)
        recon = recon + tf.image.resize(skip, (H*self.scale, W*self.scale), method='bilinear')
        return recon

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)
