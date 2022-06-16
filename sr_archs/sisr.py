from einops import rearrange
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, List


class PixelShuffle(nn.Module):
    upsample_rate: int

    @nn.compact
    def __call__(self, x):
        return rearrange(x, ('b h w (hc wc c) -> b (h hc) (w wc) c'),
                         hc=self.upsample_rate, wc=self.upsample_rate
                         )


class NAFBlock(nn.Module):
    n_filters: int
    kh: int
    kw: int
    survival_prob: float
    dw_expansion_rate: int = 2
    ffn_expansion_rate: int = 2

    @nn.compact
    def __call__(self, x, deterministic=False):
        dw_filters = self.n_filters * self.dw_expansion_rate
        ffn_filters = self.n_filters * self.ffn_expansion_rate
        beta = self.param('beta',
                          nn.initializers.zeros,
                          (1, 1, 1, self.n_filters)
                          )
        gamma = self.param('gamma',
                           nn.initializers.zeros,
                           (1, 1, 1, self.n_filters)
                           )

        spatial = nn.LayerNorm()(x)
        spatial = nn.Conv(dw_filters,
                          kernel_size=(1, 1),
                          padding='VALID',
                          kernel_init=nn.initializers.variance_scaling(.02, 'fan_in', 'truncated_normal')
                          )(spatial)
        spatial = nn.Conv(dw_filters,
                          kernel_size=(3, 3),
                          padding='SAME',
                          feature_group_count=dw_filters,
                          kernel_init=nn.initializers.variance_scaling(.02, 'fan_in', 'truncated_normal')
                          )(spatial)
        # simple gate
        spatial_gate1, spatial_gate2 = jnp.split(spatial,
                                                 indices_or_sections=2,
                                                 axis=-1
                                                 )
        spatial = spatial_gate1 * spatial_gate2
        if deterministic:
            # TLSC (https://arxiv.org/pdf/2112.04491v2.pdf)
            b, h, w, c = x.shape
            s = spatial.cumsum(2).cumsum(1)
            s = jnp.pad(s,
                        [[0, 0], [1, 0], [1, 0], [0, 0]]
                        )
            kh, kw = min(h, self.kh), min(w, self.kw)
            s1, s2, s3, s4 = s[:, :-kh, :-kw, :],\
                             s[:, :-kh, kw:, :],\
                             s[:, kh:, :-kw, :],\
                             s[:, kh:, kw:, :]
            spatial_statistic = (s4 + s1 - s2 - s3) / (kh * kw)
            if (kh != h) and (kw != w):
                _, h_s, w_s, _ = spatial_statistic.shape
                h_pad, w_pad = [(h - h_s) // 2, (h - h_s + 1) // 2], [(w - w_s) // 2, (w - w_s + 1) // 2]
                spatial_statistic = jnp.pad(spatial_statistic,
                                            [[0, 0], h_pad, w_pad, [0, 0]],
                                            mode='edge'
                                            )
        else:
            spatial_statistic = jnp.mean(spatial,
                                         axis=(1, 2),
                                         keepdims=True
                                         )
        # simple attention
        spatial_attention = nn.Dense(self.n_filters,
                                     kernel_init=nn.initializers.variance_scaling(.02, 'fan_in', 'truncated_normal')
                                     )(spatial_statistic)
        spatial = spatial * spatial_attention
        spatial = nn.Conv(self.n_filters,
                          kernel_size=(1, 1),
                          padding='VALID',
                          kernel_init=nn.initializers.variance_scaling(.02, 'fan_in', 'truncated_normal')
                          )(spatial)
        spatial = beta * spatial
        x = x + DropPath(self.survival_prob)(spatial, deterministic=deterministic)

        channel = nn.LayerNorm()(x)
        channel = nn.Dense(ffn_filters,
                           kernel_init=nn.initializers.variance_scaling(.02, 'fan_in', 'truncated_normal')
                           )(channel)
        channel_gate1, channel_gate2 = jnp.split(channel,                                  # simple gate
                                                 indices_or_sections=2,
                                                 axis=-1
                                                 )
        channel = channel_gate1 * channel_gate2
        channel = nn.Dense(self.n_filters,
                           kernel_init=nn.initializers.variance_scaling(.02, 'fan_in', 'truncated_normal')
                           )(channel)
        channel = gamma * channel
        x = x + DropPath(self.survival_prob)(channel, deterministic=deterministic)
        return x


class DropPath(nn.Module):
    survival_prob: float
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
        if (self.survival_prob or deterministic) == 1.:
            return inputs
        elif self.survival_prob == 0.:
            return jnp.zeros_like(inputs)

        rng = self.make_rng('droppath')
        broadcast_shape = [inputs[0].shape[0]] + [1 for _ in range(len(inputs[0].shape) - 1)]
        mask = jax.random.bernoulli(key=rng,
                                    p=self.survival_prob,
                                    shape=broadcast_shape
                                    )
        return jnp.where(mask, inputs / self.survival_prob, 0.)


### NAFSSR(https://arxiv.org/abs/2204.08714, https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFSSR_arch.py)
class NAFNetSR(nn.Module):
    upscale_rate: int
    n_filters: int
    n_blocks: int
    stochastic_depth_rate: float
    train_size: List[int]
    tlsc_rate: float = 1.5

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        B, H, W, C = x.shape

        kh, kw = int(self.train_size[1] * self.tlsc_rate), int(self.train_size[2] * self.tlsc_rate)

        # Intro
        features = nn.Conv(self.n_filters,
                           (3, 3),
                           (1, 1),
                           padding='SAME',
                           kernel_init=nn.initializers.variance_scaling(.02, 'fan_in', 'truncated_normal')
                           )(x)

        # Middle
        for _ in range(self.n_blocks):
            features = NAFBlock(self.n_filters,
                                kh, kw, 1 - self.stochastic_depth_rate)(features, deterministic=deterministic)

        # End
        features = nn.Conv(3 * self.upscale_rate ** 2,
                           kernel_size=(3, 3),
                           kernel_init=nn.initializers.variance_scaling(.02, 'fan_in', 'truncated_normal')
                           )(features)
        recon = PixelShuffle(self.upscale_rate)(features)
        recon = nn.Conv(3,
                        kernel_size=(3, 3),
                        kernel_init=nn.initializers.variance_scaling(.02, 'fan_in', 'truncated_normal')
                        )(recon)
        recon = nn.Conv(1,
                        kernel_size=(1, 1),
                        kernel_init=nn.initializers.variance_scaling(.02, 'fan_in', 'truncated_normal')
                        )(recon)
        recon_skip = jax.image.resize(x,
                                      (B, H * self.upscale_rate, W * self.upscale_rate, C),
                                      method='bicubic'
                                      )
        recon = recon + recon_skip
        return recon
