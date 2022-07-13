import tensorflow as tf


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, upsample_rate):
        super(PixelShuffle, self).__init__()
        self.upsample_rate = upsample_rate

    def call(self, inputs, *args, **kwargs):
        return tf.nn.depth_to_space(inputs,
                                    block_size=self.upsample_rate
                                    )