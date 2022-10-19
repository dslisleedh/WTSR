import tensorflow as tf
import tensorflow.keras.backend as K
from sr_archs.model_utils import *
from typing import List
import tensorflow_addons as tfa
import einops

import numpy as np
import math

from typing import Sequence, Union


# TLSC: Local statistic aggregating
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


# Train때는 Patch로 학습하지만 Inference때는 전체 이미지사용. -> 추출하는 범위가 Train/Inference때 다르기에
# Statistic의 차이가 있는 것을 줄이기 위해 나온 방법.

# Train 시에는 Patch의 Statistic을 반환하고
# Inference 시에는 Patch_size * TLSC_rate 크기의 Patch에서 추출한 Local statistic을 반환함. TLSC_rate는 hyperparameter지만,
# NAFNet에서 사용한 것과 같이 1.5로 고정했음.
class LocalAvgPool2D(tf.keras.layers.Layer):
    def __init__(self,
                 local_size: List[int]
                 ):
        super(LocalAvgPool2D, self).__init__()
        self.local_size = local_size

    def call(self, inputs, training):
        if training:
            return tf.reduce_mean(inputs, axis=[1,2], keepdims=True)
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


class SimpleGate(tf.keras.layers.Layer):
    def __init__(self):
        super(SimpleGate, self).__init__()

    def call(self, inputs, *args, **kwargs):
        x1, x2 = tf.split(inputs,
                          num_or_size_splits=2,
                          axis=-1
                          )
        return x1 * x2


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
        self.beta = tf.Variable(
            tf.zeros((1, 1, 1, self.n_filters)),
            trainable=True,
            dtype=tf.float32,
            name='beta'
        )

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


# 각 Element를 확률적으로 Drop하는 Dropout과 다르게 각 Block을 확률적으로 Drop함.
# Droppath를 구현한 방법엔 여러가지가 있지만, NAFNet에서 구현한 방법을 따라서 구현.

# Train 시 0. ~ 1. 에서 uniform sample된 value가 survival prob보다 낮으면 state = True 아니면 state = False
# True면 Input + (self.forward(inputs) - inputs / self.survival_prob) 을 반환.
# False면 그냥 input만 반환(forward를 거치지 않음)

# Test 시에는 그냥 self.forward(inputs) 반환.
class DropPath(tf.keras.layers.Layer):
    def __init__(self,
                 survival_prob: float,
                 module: tf.keras.layers.Layer
                 ):
        super(DropPath, self).__init__()
        self.survival_prob = survival_prob
        self.forward = module

    def call(self, inputs, training):

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
            _call_test(),
            training=training
        )


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
        features = self.middles(features, training=training)
        recon = self.upscale(features, training=training) + x_skip
        return recon

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = False
        return self.forward(inputs, training=training)
    
    
class NAFNetSRGAN(NAFNetSR):
    def __init__(self, **kwargs):
        self.discriminator_filters = kwargs.pop('discriminator_filters', 32)
        self.gan_loss_weight = kwargs.pop('gan_loss_weight', 1e-3)
        self.discriminator_type = kwargs.pop('discriminator_type', 'sngan')
        super(NAFNetSRGAN, self).__init__(**kwargs)
        self.discriminator = Discriminator(
            self.discriminator_filters,
            kwargs['upscale_rate']
        )
        self.pt_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False, weights='imagenet', input_shape=(96, 96, 3), pooling='avg'
        )
        self.disc_loss = hinge_disc_loss if discriminator_type == 'sngan' else relative_disc_loss
        self.gen_loss = hinge_gen_loss if discriminator_type == 'sngan' else relative_gen_loss

    def compile(self, **kwargs):
        self.d_optimizer = kwargs.pop('d_optimizer', tf.keras.optimizers.Adam(learning_rate=2e-4))
        super(NAFNetSRGAN, self).compile(**kwargs)
        
    def train_step(self, data):
        lr, hr = data

        with tf.GradientTape(persistent=True) as tape:
            sr = self.forward(lr, training=True)
            sr_loss = tf.reduce_mean(
                tf.abs(sr - hr)
            )

            sr = sr[:, slice(0, 48), slice(0, 48), :]
            sr = tf.concat([sr for _ in range(3)], axis=-1)
            hr = hr[:, slice(0, 48), slice(0, 48), :]
            hr = tf.concat([hr for _ in range(3)], axis=-1)
            disc_hr = self.discriminator(hr, training=True)
            disc_sr = self.discriminator(sr, training=True)
            disc_true_loss, disc_fake_loss = self.disc_loss(
                disc_hr, disc_sr, 1.
            )
            gen_loss = self.gen_loss(
                disc_hr, disc_sr, self.gan_loss_weight
            )
            sr_model_loss = sr_loss + gen_loss
            discriminator_loss = (disc_true_loss + disc_fake_loss) * .5

        sr_model_grads = tape.gradient(
            sr_model_loss,
            self.intro.trainable_variables + self.middles.trainable_variables + self.upscale.trainable_variables
        )
        discriminator_grads = tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(
                sr_model_grads,
                self.intro.trainable_variables + self.middles.trainable_variables + self.upscale.trainable_variables
            )
        )
        self.d_optimizer.apply_gradients(
            zip(discriminator_grads, self.discriminator.trainable_variables)
        )

        return {
            'reconstruction_loss': sr_loss,
            'discrimination_true_loss': disc_true_loss,
            'discrimination_fake_loss': disc_fake_loss,
            'generator_loss': gen_loss
        }

    def test_step(self, data):
        lr, hr = data
        sr = self.forward(lr, training=False)
        reconstruction_loss = tf.reduce_mean(tf.abs(sr - hr))
        sr = sr[:, slice(0, 96), slice(0, 96), :]
        sr = tf.concat([sr for _ in range(3)], axis=-1)
        hr = hr[:, slice(0, 96), slice(0, 96), :]
        hr = tf.concat([hr for _ in range(3)], axis=-1)
        sr_feature_map = self.pt_model(sr)
        hr_feature_map = self.pt_model(hr)
        fid = fid_score(hr_feature_map, sr_feature_map)
        psnr = tf.reduce_mean(tf.image.psnr(sr, hr, max_val=1.))
        ssim = tf.reduce_mean(tf.image.ssim(sr, hr, max_val=1.))
        return {
            'reconstruction_loss': reconstruction_loss,
            'psnr': psnr,
            'ssim': ssim,
            'fid': fid
        }


#################################################### SwinIR ####################################################


def window_partition(x, window_size):
    return einops.rearrange(
        x, 'B (H hw) (W ww) C -> (B H W) hw ww C',
        hw=window_size, ww=window_size
    )


def window_reverse(x, h, w):
    return einops.rearrange(
        x, '(B H W) hw ww C -> B (H hw) (W ww) C',
        H=h, W=w
    )


class DropPath_swin(tf.keras.layers.Layer):
    def __init__(
            self,
            survival_prob
    ):
        super(DropPath_swin, self).__init__()
        self.survival_prob = survival_prob

    def call(self, inputs, *args, **kwargs):

        def _call_train():
            state = K.random_bernoulli(
                shape=(), p=self.survival_prob
            )
            return inputs / self.survival_prob * state

        def _call_test():
            return inputs

        return K.in_train_phase(
            _call_train,
            _call_test
        )


class MLP(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters: int,
            expansion_rate: float = 4.,
            drop_rate: float = 0.,
            act=tf.nn.gelu
    ):
        super(MLP, self).__init__()
        self.n_filters = n_filters
        self.expansion_rate = expansion_rate
        self.drop_rate = drop_rate
        self.act = act

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_filters * self.expansion_rate,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            tf.keras.layers.Lambda(lambda x: self.act(x)),
            tf.keras.layers.Dense(
                self.n_filters,
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            )
        ] + [
            DropPath_swin(1. - self.drop_rate) if self.drop_rate > 0. else tf.keras.layers.Layer()
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class WindowAttention(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters: int,
            drop_rate: float,
            window_size: int,
            shift_size: Union[int, None],
            n_heads: int,
            qk_scale: Union[float, None] = None,
            qkv_bias: bool = True
    ):
        super(WindowAttention, self).__init__()
        self.n_filters = n_filters
        self.drop_rate = drop_rate
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.qk_scale = qk_scale or n_filters ** -.5
        self.qkv_bias = qkv_bias

        self.to_qkv = tf.keras.layers.Dense(
            self.n_filters * 3,
            use_bias=self.qkv_bias,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.relative_position_index = self.get_relative_position_index(self.window_size, self.window_size)
        self.proj = tf.keras.layers.Dense(
            self.n_filters,
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.droppath = DropPath_swin(1. - self.drop_rate) if self.drop_rate > 0. else tf.identity

    def get_relative_position_index(self, win_h, win_w):
        xx, yy = tf.meshgrid(range(win_h), range(win_w))
        coords = tf.stack([yy, xx], axis=0)
        coords_flatten = tf.reshape(coords, [2, -1])

        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )
        relative_coords = tf.transpose(
            relative_coords, perm=[1, 2, 0]
        )
        xx = (relative_coords[:, :, 0] + win_h - 1) * (2 * win_w - 1)
        yy = relative_coords[:, :, 1] + win_w - 1
        relative_coords = tf.stack([xx, yy], axis=-1)
        return tf.reduce_sum(relative_coords, axis=-1)

    def get_relative_positional_bias(self):
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            self.relative_position_index,
            axis=0
        )
        return tf.transpose(relative_position_bias, [2, 0, 1])

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.window_size - 1) * (2 * self.window_size - 1), self.n_heads),
            initializer='zeros',
            trainable=True,
            name='relative_position_bias_table'
        )
        super(WindowAttention, self).build(input_shape)

    def call(self, inputs, mask=None, *args, **kwargs):
        # input : B N C
        _, N, _ = inputs.get_shape().as_list()
        q, k, v = tf.unstack(
            einops.rearrange(
                self.to_qkv(
                    inputs
                ), 'B N (QKV H C) -> QKV B H N C',
                QKV=3, H=self.n_heads
            ), num=3, axis=0
        )
        attention_map = tf.matmul(q, k, transpose_b=True) * self.qk_scale
        attention_map = attention_map + self.get_relative_positional_bias()

        if tf.is_tensor(mask):
            num_wins = tf.shape(mask)[0]
            attention_map = tf.reshape(
                attention_map, (-1, num_wins, self.n_heads, N, N)
            )
            attention_map = attention_map + tf.expand_dims(mask, 1)[None, ...]

            attention_map = tf.reshape(attention_map, (-1, self.n_heads, N, N))

        attention = tf.nn.softmax(attention_map, axis=-1)

        out = einops.rearrange(
            tf.matmul(attention, v), 'B H N C -> B N (H C)'
        )
        out = self.droppath(
            self.proj(out)
        )
        return out


class SwinTransformerLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            n_filters: int,
            window_size: int,
            shift_size: Union[int, None],
            n_heads: int,
            qkv_bias: bool,
            drop_rate: float,
            norm=tf.keras.layers.LayerNormalization
    ):
        super(SwinTransformerLayer, self).__init__()
        self.n_filters = n_filters
        self.window_size = window_size
        self.shift_size = shift_size
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate

        self.ln_attn = norm()
        self.window_attention = WindowAttention(
            self.n_filters, self.drop_rate, self.window_size, self.shift_size,
            self.n_heads, qkv_bias=self.qkv_bias
        )
        self.ln_ffn = norm()
        self.ffn = MLP(
            self.n_filters, self.drop_rate
        )

    def get_attention_mask(self, input_shape: Sequence):
        img_mask = np.zeros((1, input_shape[0], input_shape[1], 1))
        cnt = 0
        for h in (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        ):
            for w in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            ):
                img_mask[:, h, w, :] = cnt
                cnt += 1

        img_mask = tf.convert_to_tensor(img_mask, dtype='float32')
        mask_windows = window_partition(
            img_mask, self.window_size
        )
        mask_windows = tf.reshape(
            mask_windows, (-1, self.window_size * self.window_size)
        )
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
        attn_mask = tf.where(attn_mask != 0, -100., attn_mask)
        return tf.where(attn_mask == 0, 0., attn_mask)

    def call(self, inputs, *args, **kwargs):
        _, h, w, _ = inputs.shape  # B H W C

        attn_res = self.ln_attn(inputs)
        if self.shift_size is not None:
            attn_res = tf.roll(
                attn_res, shift=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
            mask = self.get_attention_mask((h, w))
        else:
            mask = None

        attn_res = window_partition(attn_res, self.window_size)  #n_windows*B window_size window_size C
        attn_res = einops.rearrange(
            attn_res, 'B W1 W2 C -> B (W1 W2) C'
        )  # n_windows*B window_size*window_size C

        attn_res = self.window_attention(attn_res, mask=mask)  # n_windows*B window_size*window_size C
        attn_res = einops.rearrange(
            attn_res, 'B (W1 W2) C -> B W1 W2 C',
            W1=self.window_size, W2=self.window_size
        )  # n_windows*B window_size*window_size C
        attn_res = window_reverse(attn_res, h//self.window_size, w//self.window_size)  # B H W C

        if self.shift_size is not None:
            attn_res = tf.roll(
                attn_res, shift=(self.shift_size, self.shift_size), axis=(1, 2)
            )  # B H W C

        inputs += attn_res

        inputs += self.ffn(self.ln_ffn(inputs))

        return inputs


class ResidualSwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            n_blocks: int,
            n_filters: int,
            window_size: int,
            n_heads: int,
            qkv_bias: bool,
            drop_rate: Sequence[float],
            res_connection: str
    ):
        super(ResidualSwinTransformerBlock, self).__init__()
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.window_size = window_size
        self.shift_size = window_size//2
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        assert len(drop_rate) == n_blocks
        self.drop_rate = drop_rate
        self.res_connection = res_connection

        self.forward = tf.keras.Sequential([
            SwinTransformerLayer(
                self.n_filters,
                self.window_size,
                None if (i % 2 == 0) else self.shift_size,
                self.n_heads,
                self.qkv_bias,
                dr
            ) for i, dr in enumerate(self.drop_rate)
        ] + [
            tf.keras.layers.Conv2D(
                self.n_filters,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ) if self.res_connection == '1conv' else tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    self.n_filters//4,
                    (3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                    activation=tf.keras.layers.LeakyReLU(alpha=.2)
                ),
                tf.keras.layers.Conv2D(
                    self.n_filters//4,
                    (1, 1),
                    strides=(1, 1),
                    padding='VALID',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                    activation=tf.keras.layers.LeakyReLU(alpha=.2)
                ),
                tf.keras.layers.Conv2D(
                    self.n_filters,
                    (3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
                )
            ])
        ])

    def call(self, inputs, *args, **kwargs):
        return inputs + self.forward(inputs)


class UpsampleOneStep(tf.keras.layers.Layer):
    def __init__(self, output_channel: int, upsample_rate: int):
        super(UpsampleOneStep, self).__init__()
        self.output_channel = output_channel
        self.upsample_rate = upsample_rate

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.output_channel * (self.upsample_rate ** 2),
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ),
            PixelShuffle(self.upsample_rate)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class Upsample(tf.keras.layers.Layer):
    def __init__(self, n_filters: int, output_channel: int, upsample_rate: int):
        super(Upsample, self).__init__()
        self.n_filters = n_filters
        self.output_channel = output_channel
        self.upsample_rate = upsample_rate

        self.forward = tf.keras.Sequential()
        self.forward.add(
            tf.keras.layers.Conv2D(
                self.n_filters,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                activation=tf.keras.layers.LeakyReLU(alpha=.01)
            )
        )
        if (self.upsample_rate & (self.upsample_rate - 1)) == 0:
            for _ in range(int(math.log(self.upsample_rate, 2))):
                self.forward.add(
                    tf.keras.layers.Conv2D(
                        4 * self.n_filters,
                        (3, 3),
                        strides=(1, 1),
                        padding='SAME',
                        kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
                    )
                )
                self.forward.add(
                    PixelShuffle(2)
                )
        elif self.upsample_rate == 3:
            self.forward.add(
                tf.keras.layers.Conv2D(
                    self.n_filters * 9,
                    (3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
                )
            )
            self.forward.add(
                PixelShuffle(3)
            )
        else:
            raise NotImplementedError('Invalid scale')
        self.forward.add(
            tf.keras.layers.Conv2D(
                self.output_channel,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            )
        )

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class UpsampleNearestConv(tf.keras.layers.Layer):
    def __init__(self, n_filters: int, output_channel: int, upscale_rate: int):
        super(UpsampleNearestConv, self).__init__()
        self.n_filters = n_filters
        self.output_channel = output_channel
        self.upscale_rate = upscale_rate  # 2 or 4

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                self.n_filters,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                activation=tf.keras.layers.LeakyReLU(alpha=.01)
            ),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(
                self.n_filters,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                activation=tf.keras.layers.LeakyReLU(alpha=.2)
            )
        ])
        if upscale_rate == 4:
            self.forward.add(
                tf.keras.layers.UpSampling2D()
            )
            self.forward.add(
                tf.keras.layers.Conv2D(
                    self.n_filters,
                    (3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                    activation=tf.keras.layers.LeakyReLU(alpha=.2)
                )
            )
        self.forward.add(
            tf.keras.layers.Conv2D(
                self.output_channel,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            )
        )

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class SwinIR(tf.keras.models.Model):
    def __init__(
            self,
            n_filters: int,
            n_rstb_blocks: int,
            n_stl_blocks: int,
            window_size: int,
            n_heads: int,
            qkv_bias: bool,
            drop_rate: float,
            res_connection: str,
            upsample_type: str,
            upsample_rate: int,
            img_range: Union[int, float],
            output_channel: int = 3,
            mean: Sequence[float] = [.4488, .4371, .4040]
    ):
        super(SwinIR, self).__init__()
        self.n_filters = n_filters
        self.n_rstb_blocks = n_rstb_blocks
        self.n_stl_blocks = n_stl_blocks
        self.window_size = window_size
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.res_connection = res_connection
        self.upsample_type = upsample_type
        self.upsample_rate = upsample_rate
        self.img_range = img_range
        self.output_channel = output_channel
        self.mean = tf.reshape(
            tf.convert_to_tensor(mean, dtype='float32'),
            (1, 1, 1, -1)
        )

        self.upsample_filters = 64

        stochastic_depth_rate = tf.linspace(
            0., self.drop_rate, self.n_rstb_blocks * self.n_stl_blocks
        )

        self.shallow_feature_extractor = tf.keras.layers.Conv2D(
            self.n_filters,
            (3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
        )
        self.deep_feature_extractor = tf.keras.Sequential([
            ResidualSwinTransformerBlock(
                self.n_stl_blocks,
                self.n_filters,
                self.window_size,
                self.n_heads,
                self.qkv_bias,
                stochastic_depth_rate[n * self.n_stl_blocks:(n + 1) * self.n_stl_blocks],
                self.res_connection
            ) for n in range(self.n_rstb_blocks)
        ] + [
            tf.keras.layers.LayerNormalization()
        ] + [
            tf.keras.layers.Conv2D(
                self.n_filters,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            ) if self.res_connection == '1conv' else tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    self.n_filters//4,
                    (3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                    activation=tf.keras.layers.LeakyReLU(alpha=.2)
                ),
                tf.keras.layers.Conv2D(
                    self.n_filters//4,
                    (1, 1),
                    strides=(1, 1),
                    padding='VALID',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02),
                    activation=tf.keras.layers.LeakyReLU(alpha=.2)
                ),
                tf.keras.layers.Conv2D(
                    self.n_filters,
                    (3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
                )
            ])
        ])

        if self.upsample_type == 'pixelshuffle':
            self.reconstructor = Upsample(
                self.upsample_filters, self.output_channel, self.upsample_rate
            )
        elif self.upsample_type == 'pixelshuffle_onestep':
            self.reconstructor = UpsampleOneStep(
                self.output_channel, self.upsample_rate
            )
        elif self.upsample_type == 'nearest_conv':
            self.reconstructor = UpsampleNearestConv(
                self.upsample_filters, self.output_channel, self.upsample_rate
            )
        else:
            self.reconstructor = tf.keras.layers.Conv2D(
                self.output_channel,
                (3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_initializer=tf.keras.initializers.VarianceScaling(.02)
            )

    def forward(self, x, training=False):
        x = (x - self.mean) * self.img_range

        if self.upsample_type in ['pixelshuffle', 'pixelshuffle_onestep', 'nearest_conv']:
            features = self.shallow_feature_extractor(x, training=training)
            features = self.deep_feature_extractor(features, training=training) + features
            reconstruction = self.reconstructor(features, training=training)
        else:
            features = self.shallow_feature_extractor(x)
            features = self.deep_feature_extractor(features) + features
            reconstruction = self.reconstructor(features) + x

        reconstruction = (reconstruction / self.img_range) + self.mean
        return reconstruction

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = False
        return self.forward(inputs, training=training)
