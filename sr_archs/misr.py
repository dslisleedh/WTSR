import tensorflow as tf
import tensorflow_probability as tfp
from sr_archs.model_utils import PixelShuffle
import einops


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
                qkv_expansion=3, h=8
            ),
            num=3,
            axis=1
        ) # b h n c * 3
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
            name='cls_token'
        )
        self.attns = [
            MultiHeadSelfAttention(n_filters, n_heads, dropout_rate) for _ in range(n_blocks)
        ]
        self.ffns = [
            FFN(n_filters, n_mlp_filters, dropout_rate) for _ in range(n_blocks)
        ]

    def call(self, features, attention_mask=None, **kwargs):
        cls_token = tf.ones_like(
            tf.gather(features, [0], axis=1)
        ) * self.cls_token
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
            lr = inputs
            attention_mask = None
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

    def forward(self, x, attention_mask=None, training=False):
        b, h, w, t = x.get_shape().as_list()

        if tf.is_tensor(attention_mask):
            attention_mask = tf.pad(attention_mask, ((0, 0), (1, 0)), constant_values=1.)[:, tf.newaxis, tf.newaxis, :]
            attention_mask = einops.rearrange(tf.broadcast_to(attention_mask, (b, h, w, t+1)), 'b h w t_pad -> (b h w) t_pad')[:, tf.newaxis, tf.newaxis, :]

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
        recon = self.decoder(features, training=training)
        return recon

    def call(self, inputs, training=None, mask=None):
        if tf.is_tensor(inputs):
            lrs = inputs
            attention_mask = None
        else:
            lrs, attention_mask = inputs
        if training is None:
            training = False
        return self.forward(lrs, attention_mask=attention_mask, training=training)
