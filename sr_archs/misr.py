import tensorflow as tf
import tensorflow_probability as tfp
from sr_archs.model_utils import PixelShuffle, simple_gate
from sr_archs.sisr import SimpleChannelAttention, LocalAvgPool2D, edge_padding2d
import einops
from einops.layers.keras import Rearrange
from typing import List


# cMSE/cPNSR 및 bad pixel mask는 구현하지 않았음.
# cMSE/cPSNR: 위성사진의 mis-alignment으로 인한 loss를 줄이기 위함인데 수온 데이터는 그런 위험이 적고 mis-alignment가 있더라도 수온의
#             변동량이 더 클 것이라 생각했음. 또한 cMSE를 구하는데 필요한 clearity? 이미지의 품질을 나타내는 항목도 없어서 사용불가함.
# bad pixel mask: 원본 데이터는 구름에 가려진 부분 같은 데를 마스크 처리해서 로스에 포함시키지 않지만 여기는 그럴 걱정 X
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 kernel_size: int
                 ):
        super(ResidualBlock, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.n_filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.PReLU(shared_axes=[1, 2]),
            tf.keras.layers.Conv2D(
                self.n_filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.PReLU(shared_axes=[1, 2])
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs) + inputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 n_layers,
                 kernel_size=3
                 ):
        super(Encoder, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        self.intro = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.n_filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.PReLU(shared_axes=[1, 2])
        ])
        self.blocks = tf.keras.Sequential([
            ResidualBlock(
                self.n_filters,
                self.kernel_size
            ) for _ in range(self.n_layers)
        ])
        self.out = tf.keras.layers.Conv2D(
            self.n_filters,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(1, 1),
            padding='SAME',
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )

    def call(self, inputs, *args, **kwargs):
        features = self.intro(inputs)
        features = self.blocks(features)
        return self.out(features)


class FFN(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 n_mlp_filters: int,
                 dropout_rate: float = 0.
                 ):
        super(FFN, self).__init__()
        self.n_filters = n_filters
        self.n_mlp_filters = n_mlp_filters
        self.dropout_rate = dropout_rate

        self.forward = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.n_mlp_filters,
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                                  activation='gelu'
                                  ),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.n_filters,
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
                                  ),
            tf.keras.layers.Dropout(self.dropout_rate)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs) + inputs


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 n_heads: int,
                 dropout_rate: float = 0.
                 ):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_filters = n_filters
        self.n_heads = n_heads
        self.scale = tf.Variable(
            float(self.n_filters) ** -0.5,
            trainable=False,
            dtype=tf.float32,
            name='scale'
        )
        self.dropout_rate = dropout_rate

        self.to_qkv = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(
                self.n_filters * 3,
                use_bias=False,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
                )
        ])
        self.to_out = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_filters,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.Dropout(self.dropout_rate)
        ])

    def call(self, inputs, attention_mask=None, *args, **kwargs):
        qkv = self.to_qkv(inputs)
        q, k, v = tf.unstack(
            einops.rearrange(
                qkv, 'b n (qkv_expansion h c) -> b qkv_expansion h n c',
                qkv_expansion=3, h=self.n_heads
            ),
            num=3,
            axis=1
        )
        attention_map = tf.matmul(q, k, transpose_b=True) * self.scale
        if tf.is_tensor(attention_mask):
            attention_map += (1. - attention_mask) * tf.DType(1.).min
        attention = tf.nn.softmax(attention_map, axis=-1)
        out = einops.rearrange(
            tf.matmul(attention, v), 'b h n c -> b n (h c)'
        )
        out = self.to_out(out)
        return inputs + out


class Transformer(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 n_blocks: int,
                 n_heads: int,
                 n_mlp_filters: int,
                 dropout_rate: float,
                 ):
        super(Transformer, self).__init__()
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_mlp_filters = n_mlp_filters
        self.dropout_rate = dropout_rate

        self.cls_token = tf.Variable(
            tf.random.truncated_normal(shape=(1, 1, self.n_filters), stddev=0.02),
            trainable=True,
            dtype=tf.float32,
            name='cls_token'
        )
        self.attns = [
            MultiHeadSelfAttention(n_filters, n_heads, dropout_rate) for _ in range(n_blocks)
        ]
        self.ffns = [
            FFN(n_filters, n_mlp_filters, dropout_rate) for _ in range(n_blocks)
        ]

    def call(self, features, attention_mask=None, **kwargs):
        # add cls token not multiply to prevent gradient exploding/vanishing
        cls_token = tf.zeros_like(
            tf.gather(features, [0], axis=1)
        ) + self.cls_token
        features = tf.concat([cls_token, features], axis=1)
        for a, f in zip(self.attns, self.ffns):
            features = a(features, attention_mask)
            features = f(features)
        cls_token = tf.gather(features, [0], axis=1)
        return cls_token


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 scale: int,
                 kernel_size: int = 1
                 ):
        super(Decoder, self).__init__()
        self.kernel_size = kernel_size
        self.scale = scale

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.scale**2,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding='VALID',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.PReLU(shared_axes=[1, 2])
        ])

    def call(self, inputs, *args, **kwargs):
        return tf.nn.depth_to_space(
            self.forward(inputs), self.scale
        )


class TRNet(tf.keras.models.Model):
    def __init__(self,
                 n_filters: int,
                 n_enc_layers: int,
                 n_transform_layers: int,
                 n_heads: int,
                 n_mlp_filters: int,
                 dropout_rate: float,
                 upscale_rate: int
                 ):
        super(TRNet, self).__init__()
        self.n_filters = n_filters
        self.n_enc_layers = n_enc_layers
        self.n_transformer_layers = n_transform_layers
        self.n_heads = n_heads
        self.n_mlp_filters = n_mlp_filters
        self.dropout_rate = dropout_rate
        self.upscale_rate = upscale_rate

        self.encoder = Encoder(
            self.n_filters, self.n_enc_layers
        )
        self.transformer = Transformer(
            self.n_filters, self.n_transformer_layers, self.n_heads, self.n_mlp_filters, self.dropout_rate
        )
        self.decoder = Decoder(
            self.upscale_rate
        )

    def compile(self, ed_optimizer, fu_optimizer, **kwargs):
        super(TRNet, self).compile(**kwargs)
        self.ed_optimizer = ed_optimizer
        self.fu_optimizer = fu_optimizer

    def train_step(self, data):
        inputs, hr = data
        if tf.is_tensor(inputs):
            lr, attention_mask = inputs, None
        else:
            lr, attention_mask = inputs

        with tf.GradientTape(persistent=True) as tape:
            recon = self.forward(lr, attention_mask=attention_mask, training=True)
            loss = self.compiled_loss(hr, recon, regularization_losses=self.losses)
        grads_ed = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        grads_fu = tape.gradient(loss, self.transformer.trainable_variables)
        self.ed_optimizer.apply_gradients(
            zip(grads_ed, self.encoder.trainable_variables + self.decoder.trainable_variables)
        )
        self.fu_optimizer.apply_gradients(
            zip(grads_fu, self.transformer.trainable_variables)
        )

        self.compiled_metrics.update_state(hr, recon)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs, hr = data
        if tf.is_tensor(inputs):
            lr, attention_mask = inputs, None
        else:
            lr, attention_mask = inputs

        recon = self.forward(lr, attention_mask=attention_mask, training=False)
        self.compiled_loss(hr, recon, regularization_losses=self.losses)

        self.compiled_metrics.update_state(hr, recon)
        return {m.name: m.result() for m in self.metrics}

    def forward(self, x, attention_mask=None, training=False):
        b, h, w, t = x.get_shape().as_list()

        skip = tf.image.resize(
            tf.gather(
                tf.reverse(x, axis=[-1]), [0], axis=3
            ), (h * self.upscale_rate, w * self.upscale_rate), method='bicubic'
        )
        if tf.is_tensor(attention_mask):
            attention_mask = tf.pad(attention_mask, ((0, 0), (1, 0)), constant_values=1.)[:, tf.newaxis, tf.newaxis, :]
            attention_mask = einops.rearrange(
                tf.broadcast_to(attention_mask, (b, h, w, t+1)), 'b h w t_pad -> (b h w) t_pad'
            )[:, tf.newaxis, tf.newaxis, :]

        # Encoder
        x = tf.expand_dims(x, axis=-1)
        x_ref = einops.repeat(
            tfp.stats.percentile(x, 50., axis=-2, keepdims=True),
            'b h w () c -> b h w t c', t=t
        )
        x = tf.concat([x, x_ref], axis=-1)
        x = einops.rearrange(x, 'b h w t c -> (b t) h w c')
        features = self.encoder(x, training=training)

        # Transformer
        features = einops.rearrange(
            features, '(b t) h w c -> (b h w) t c', t=t
        )
        features = self.transformer(features, attention_mask=attention_mask, training=training)

        # Decoder(PixelShuffle)
        features = einops.rearrange(
            features, '(b h w) t c -> b h w t c', h=h, w=w
        )[:, :, :, 0, :]
        recon = self.decoder(features, training=training) + skip
        return recon

    def call(self, inputs, training=None, mask=None):
        if tf.is_tensor(inputs):
            lrs, attention_mask = inputs, None
        else:
            lrs, attention_mask = inputs
        if training is None:
            training = False
        return self.forward(lrs, attention_mask=attention_mask, training=training)


class MHSA(tf.keras.layers.Layer):
    def __init__(
            self, n_filters: int, n_heads: int = 8
    ):
        super(MHSA, self).__init__()
        self.n_filters = n_filters
        self.n_heads = n_heads
        self.scale = tf.Variable(tf.sqrt(tf.cast(self.n_filters, tf.float32)), trainable=False)

        self.to_qkv = tf.keras.layers.Dense(
            self.n_filters * 3, use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.to_out = tf.keras.layers.Dense(
            self.n_filters, use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )

    def to_heads(self, x):
        x = einops.rearrange(x, 'b n (h d) -> b h n d', h=self.n_heads)
        return x

    def call(self, inputs, training):
        qkv = self.to_qkv(inputs)
        q, k, v = tf.split(qkv, 3, axis=-1)
        q, k, v = [self.to_heads(x) for x in [q, k, v]]

        attn = tf.matmul(q, k, transpose_b=True) / self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = tf.matmul(attn, v)

        out = einops.rearrange(attn, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TestBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters: int,
            n_heads: int,
            time: int,
            drop_rate: float,
            kh: int,
            kw: int,
            activation: str,
            expansion_rate: int = 2
    ):
        super(TestBlock, self).__init__()
        self.n_filters = n_filters
        self.n_heads = n_heads
        self.time = time
        self.drop_rate = drop_rate
        self.kh = kh
        self.kw = kw
        self.activation = activation
        self.expansion_rate = expansion_rate

        self.spatial_wise = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(
                self.n_filters * self.expansion_rate, 1, use_bias=False,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.DepthwiseConv2D(
                3, padding='same', use_bias=False,
                activation=simple_gate if self.activation == 'simple_gate' else tf.nn.gelu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            # SimpleChannelAttention(
            #     self.n_filters,
            #     self.kh,
            #     self.kw
            # ),
            tf.keras.layers.Conv2D(
                self.n_filters, 1, use_bias=False,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            )
        ])
        self.time_wise = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            MHSA(self.n_filters, self.n_heads)
        ])
        self.gamma = tf.Variable(
            tf.zeros((1, 1, 1, self.n_filters)), trainable=True
        )
        self.beta = tf.Variable(
            tf.zeros((1, 1, self.n_filters)), trainable=True
        )
        self.channel_wise = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(
                self.n_filters * self.expansion_rate, use_bias=False,
                activation=simple_gate if self.activation == 'simple_gate' else tf.nn.gelu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.Dense(
                self.n_filters, use_bias=False,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            )
        ])
        self.phi = tf.Variable(
            tf.zeros((1, 1, 1, self.n_filters)), trainable=True
        )

    def forward(self, inputs):
        _, h, w, c = inputs.get_shape().as_list()
        x = self.spatial_wise(inputs) * self.gamma + inputs
        x = einops.rearrange(
            x, '(b t) h w c -> (b h w) t c', t=self.time
        )
        x = self.time_wise(x) * self.beta + x
        x = einops.rearrange(
            x, '(b h w) t c -> (b t) h w c', h=h, w=w
        )
        x = self.channel_wise(x) * self.phi + x
        return x

    def call(self, inputs, training):
        if training:
            if tf.less(tf.random.uniform([], 0, 1), self.drop_rate):
                return inputs
            else:
                return inputs + ((self.forward(inputs) - inputs) / (1 - self.drop_rate))
        else:
            return self.forward(inputs)


class TestModel(tf.keras.models.Model):
    def __init__(
            self,
            n_filters: int,
            n_blocks: int,
            n_heads: int,
            time: int,
            drop_rate: float,
            scale: int,
            train_size: List,
            activation: str,
            expansion_rate: int = 2,
            tlsc_rate: float = 1.5
    ):
        super(TestModel, self).__init__()
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.time = time
        self.drop_rate = drop_rate
        self.scale = scale
        self.expansion_rate = expansion_rate
        self.train_size = train_size
        self.tlsc_rate = tlsc_rate
        self.activation = activation

        kh, kw = self.train_size[1] * tlsc_rate, self.train_size[2] * tlsc_rate

        self.to_features = tf.keras.layers.Conv2D(
            self.n_filters, 3, padding='same', use_bias=False
        )
        self.blocks = tf.keras.Sequential([
            TestBlock(
                self.n_filters, self.n_heads, self.time, self.drop_rate, kh, kw, self.activation
            ) for _ in range(self.n_blocks)
        ])
        self.to_out = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.n_filters * self.scale ** 2, 3, padding='same', use_bias=False
            ),
            tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, self.scale)),
            tf.keras.layers.Conv2D(
                1, 3, padding='same', use_bias=False
            )
        ])

    def call(self, inputs, training=None, mask=None):
        _, h, w, t = inputs.get_shape().as_list()
        skip = tf.image.resize(
            inputs[:, :, :, slice(self.time-1, self.time)], (h * self.scale, w * self.scale), method='bicubic'
        )
        x = tf.expand_dims(inputs, axis=-1)
        x = einops.rearrange(
            x, 'b h w t c -> (b t) h w c'
        )
        x = self.to_features(x)
        x = self.blocks(x, training) + x
        x = einops.rearrange(
            x, '(b t) h w c -> b t h w c', t=self.time
        )
        x = tf.squeeze(
            x[:, slice(self.time-1, self.time)], axis=1
        )
        x = self.to_out(x)
        return x + skip
