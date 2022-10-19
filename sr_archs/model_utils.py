import tensorflow as tf
import tensorflow_addons as tfa


def hinge_disc_loss(true_logit, fake_logit, multiplier):
    true_loss = tf.reduce_mean(
        tf.nn.relu(1. - true_logit)
    ) * multiplier
    fake_loss = tf.reduce_mean(
        tf.nn.relu(1. + fake_logit)
    ) * multiplier
    return true_loss, fake_loss


def hinge_gen_loss(fake_logit, multiplier, reduce=True):
    gen_loss = -fake_logit * multiplier
    if reduce:
        return tf.reduce_mean(gen_loss)
    else:
        return gen_loss


def vanilla_disc_loss(true_logit, fake_logit, multiplier):
    true_loss = tf.reduce_mean(
        tf.losses.binary_crossentropy(
            tf.ones_like(true_logit), true_logit, from_logits=True
        )
    ) * multiplier
    fake_loss = tf.reduce_mean(
        tf.losses.binary_crossentropy(
            tf.zeros_like(fake_logit), fake_logit, from_logits=True
        )
    ) * multiplier
    return true_loss, fake_loss


def vanilla_gen_loss(fake_logit, multiplier, reduce=True):
    gen_loss = tf.losses.binary_crossentropy(
        tf.ones_like(fake_logit), fake_logit, from_logits=True
    ) * multiplier
    if reduce:
        return tf.reduce_mean(gen_loss)
    else:
        return gen_loss


def relative_disc_loss(true_logit, fake_logit, multiplier):
    true_loss = tf.reduce_mean(
        tf.losses.binary_crossentropy(
            tf.ones_like(true_logit), true_logit - tf.reduce_mean(fake_logit), from_logits=True
        )
    ) * multiplier
    fake_loss = tf.reduce_mean(
        tf.losses.binary_crossentropy(
            tf.zeros_like(true_logit), fake_logit - tf.reduce_mean(true_logit), from_logits=True
        )
    ) * multiplier
    return true_loss, fake_loss


def relative_gen_loss(true_logit, fake_logit, multiplier, reduce=True):
    true_loss = tf.reduce_mean(
        tf.losses.binary_crossentropy(
            tf.zeros_like(true_logit), true_logit - tf.reduce_mean(fake_logit), from_logits=True
        )
    ) * multiplier
    fake_loss = tf.reduce_mean(
        tf.losses.binary_crossentropy(
            tf.ones_like(true_logit), fake_logit - tf.reduce_mean(true_logit), from_logits=True
        )
    ) * multiplier
    gen_loss = (true_loss + fake_loss) * .5
    if reduce:
        return tf.reduce_mean(gen_loss)
    else:
        return gen_loss


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, upsample_rate):
        super(PixelShuffle, self).__init__()
        self.upsample_rate = upsample_rate

    def call(self, inputs, *args, **kwargs):
        return tf.nn.depth_to_space(inputs,
                                    block_size=self.upsample_rate
                                    )


class Discriminator(tf.keras.layers.Layer):
    def __init__(
            self, n_filters: int, scale: int, n_layers: int = 4 # Recommend to fix n_layers as 4
    ):
        super(Discriminator, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.scale = scale
        # 48 -> 24 -> 12 -> 6 -> 3 -> 1
        self.forward = tf.keras.Sequential([
            tfa.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=self.n_filters * (2 ** i),
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding='SAME'
                )
            ) for i in range(self.n_layers)
        ] + [
            tfa.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(
                    filters=1,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='VALID'
                )
            )
        ] + [
            tf.keras.layers.Flatten()
        ])
    
    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


# GAN Evaluation Metric
def fid_score(feature_map_1, feature_map_2):
    mu_1 = tf.reduce_mean(feature_map_1, axis=0)
    mu_2 = tf.reduce_mean(feature_map_2, axis=0)
    sigma_1 = tf.linalg.matmul(
        tf.transpose(feature_map_1 - mu_1), feature_map_1 - mu_1
    ) / tf.cast(tf.shape(feature_map_1)[0], tf.float32)
    sigma_2 = tf.linalg.matmul(
        tf.transpose(feature_map_2 - mu_2), feature_map_2 - mu_2
    ) / tf.cast(tf.shape(feature_map_2)[0], tf.float32)
    sigma_12 = tf.linalg.matmul(
        tf.transpose(feature_map_1 - mu_1), feature_map_2 - mu_2
    ) / tf.cast(tf.shape(feature_map_1)[0], tf.float32)
    trace_1 = tf.linalg.trace(sigma_1)
    trace_2 = tf.linalg.trace(sigma_2)
    trace_12 = tf.linalg.trace(sigma_12)
    fid = tf.reduce_sum(
        (mu_1 - mu_2) ** 2
    ) + trace_1 + trace_2 - 2 * trace_12
    return fid


def simple_gate(x):
    x1, x2 = tf.split(
        x, num_or_size_splits=2, axis=-1
    )
    return x1 * x2


def compute_fr_loss(label, target, eps=.1):
    label_freq = tf.signal.rfft2d(
        tf.transpose(label, [0, 3, 1, 2])
    )
    target_freq = tf.signal.rfft2d(
        tf.transpose(target, [0, 3, 1, 2])
    )
    return eps * tf.reduce_mean(
        tf.abs(
            label_freq - target_freq
        )
    )


class FrequencyReconstructionLoss(tf.keras.losses.Loss):
    def __init__(self, eps=.1):
        super(FrequencyReconstructionLoss, self).__init__()
        self.eps = eps

    def call(self, y_true, y_pred):
        pixel_wise_loss = tf.reduce_mean(
            tf.abs(y_true - y_pred)
        )
        frequency_reconstruction_loss = compute_fr_loss(y_true, y_pred, self.eps)

        return pixel_wise_loss + frequency_reconstruction_loss
