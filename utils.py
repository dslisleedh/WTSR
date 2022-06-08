import tensorflow as tf
import copy
import tensorflow_addons as tfa
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import Optional, List
import optax
import time
import argparse
from sr_archs import (
    ganmodule, sisr, misr
)
from jax import lax
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from model_funcs import *


def load_datasets(model: str):
    if model == 'sisr':
        train = np.load('./data/preprocessed/train.npy')
        valid = np.load('./data/preprocessed/valid.npy')
        test = np.load('./data/preprocessed/test.npy')
    elif model == 'misr':
        train = np.load('./data/preprocessed/ts_train.npy')
        valid = np.load('./data/preprocessed/ts_valid.npy')
        test = np.load('./data/preprocessed/ts_test.npy')
    else:
        raise NotImplementedError("Model selection error. ['sisr', 'misr']")
    return train, valid, test


def get_patch(img, h_start, w_start, crop_size):
    return lax.dynamic_slice(img, (h_start, w_start, 0), (crop_size, crop_size, 1))


def get_patches(rng, img, crop_size):
    b, h, w, c = img.shape
    h_rng, w_rng = jax.random.split(rng, 2)
    h_max, w_max = (
        h - crop_size,
        w - crop_size
    )
    h_starts = jax.random.randint(h_rng, (b,), minval=0, maxval=h_max)
    w_starts = jax.random.randint(w_rng, (b,), minval=0, maxval=w_max)
    return jax.jit(jax.vmap(get_patch, in_axes=(0, 0, 0, None)), static_argnums=3)(img, h_starts, w_starts, crop_size)


def compute_metrics(recon, hr, max_val):
    if isinstance(recon, jnp.ndarray):
        recon = tf.convert_to_tensor(recon)
        hr = tf.convert_to_tensor(hr)
    psnr = tf.image.psnr(recon, hr, max_val=max_val)
    ssim = tf.image.ssim(recon, hr, max_val=max_val)
    return psnr, ssim


@jax.jit
def normalize(
        x,
        x_min=None,
        x_max=None
):
    if not x_min and not x_max:
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)

    x_scaled = (x - x_min) / (x_max - x_min)
    return x_scaled, x_min, x_max


@jax.jit
def inverse_normalize(
        x,
        x_min,
        x_max
):
    x_inversed = x * (x_max-x_min) + x_min
    return x_inversed


@jax.jit
def downsample_bicubic(
        x,
        scale
):
    b, h, w, c = x.shape
    return jax.image.resize(
        x,
        (b, h//scale, w//scale, c),
        method='bicubic'
    )