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


def create_nafnet_train_state(rng, n_steps, args_parsed):
    if args_parsed.usegan:
        rng, critic_rng = jax.random.split(rng, 2)

    model = sisr.NAFNetSR(
        args_parsed.scale,
        args_parsed.n_filters,
        args_parsed.n_blocks,
        args_parsed.stochastic_depth_rate
    )
    assert 48//args_parsed.scale == 1
    params = model.init(rng, jnp.ones([1, 48//args_parsed.scale, 48//args_parsed.scale, 1]))['params']
    scheduler = optax.cosine_onecycle_schedule(
        10 * n_steps,
        args_parsed.learning_rate,
        final_div_factor=3e4
    )
    tx = optax.adamw(
        learning_rate=scheduler,
        weight_decay=args_parsed.weight_decay,
        b1=.9,
        b2=.9
    )
    if args_parsed.usegan:
        critic = ganmodule.FlaxCritic()
        critic_params = model.init(critic_rng, jnp.ones([1, 48, 48, 1]))['params']
        critic_scheduler = optax.cosine_onecycle_schedule(
            50 * n_steps,
            args_parsed.learning_rate,
            final_div_factor=3e4
        )
        critic_tx = optax.adamw(
            learning_rate=critic_scheduler,
            weight_decay=args_parsed.weight_decay,
            b1=.5,
            b2=.9
        )
        return [
            model,
            train_state.TrainState.create(
                apply_fn=model.apply, params=params, tx=tx
            ),
            critic,
            train_state.TrainState.create(
                apply_fn=critic.apply, params=critic_params, tx=critic_tx
            )
        ]

    return [
        model,
        train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx
        )
    ]


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


@jax.jit
def apply_model(model, state, lr, hr, rng):
    model.is_training = True

    def loss_fn(params):
        recon = model.apply({'params': params}, lr, rngs={'dropout': rng, 'droppath': rng})
        loss = jnp.mean(abs(hr - recon))
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    loss, grads = grad_fn(state.params)
    return loss, grads

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

@jax.jit
def apply_critic(model, state, critic, critic_state, lr, hr, rng):
    model.is_training = True
    critic.is_training = True

    def loss_fn(params, critic_params):
        recon = model.apply({'params': params}, lr, rngs={'dropout': rng, 'droppath': rng})
        true_logit = critic.apply({'params': critic_params}, hr)
        fake_logit = critic.apply({'params': critic_params}, recon)
        loss = jnp.mean(
            fake_logit
        ) - jnp.mean(
            true_logit
        )
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    loss, grads = grad_fn(state.params, critic_state.params)
    return loss, grads


@jax.jit
def apply_generator(model, state, critic, critic_state, lr, rng):
    model.is_training = True
    critic.is_training = True

    def loss_fn(params, critic_params):
        recon = model.apply({'params': params}, lr, rngs={'dropout': rng, 'droppath': rng})
        fake_logit = critic.apply({'params': critic_params}, recon)
        loss = -jnp.mean(
            fake_logit
        )
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    loss, grads = grad_fn(state.params, critic_state.params)
    return loss, grads


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


@jax.jit
def evaluate_model(model, state, lr, hr, max_val):
    model.is_training = False
    recon = model.apply({'params': state.params}, lr)
    psnr, ssim = compute_metrics(recon, hr, max_val)
    return psnr, ssim


def compute_metrics(recon, hr, max_val):
    if isinstance(recon, jnp.ndarray):
        recon = tf.convert_to_tensor(recon)
        hr = tf.convert_to_tensor(hr)
    psnr = tf.image.psnr(recon, hr, max_val=max_val)
    ssim = tf.image.ssim(recon, hr, max_val=max_val)
    return psnr, ssim
