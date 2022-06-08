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
from utils import *


def create_nafnet_train_state(rng, n_steps, config):
    if config.usegan:
        rng, critic_rng = jax.random.split(rng, 2)

    model = sisr.NAFNetSR(
        config.scale,
        config.n_filters,
        config.n_blocks,
        config.stochastic_depth_rate
    )
    assert 48//config.scale == 1
    params = model.init(rng, jnp.ones([1, 48//config.scale, 48//config.scale, 1]))['params']
    scheduler = optax.cosine_onecycle_schedule(
        10 * n_steps,
        config.learning_rate,
        final_div_factor=3e4
    )
    tx = optax.adamw(
        learning_rate=scheduler,
        weight_decay=config.weight_decay,
        b1=.9,
        b2=.9
    )
    if config.usegan:
        critic = ganmodule.FlaxCritic()
        critic_params = model.init(critic_rng, jnp.ones([1, 48, 48, 1]))['params']
        critic_scheduler = optax.cosine_onecycle_schedule(
            50 * n_steps,
            config.learning_rate,
            final_div_factor=3e4
        )
        critic_tx = optax.adamw(
            learning_rate=critic_scheduler,
            weight_decay=config.weight_decay,
            b1=.5,
            b2=.9
        )
        return (
            train_state.TrainState.create(
                apply_fn=model.apply, params=params, tx=tx
            ),
            train_state.TrainState.create(
                apply_fn=critic.apply, params=critic_params, tx=critic_tx
            )
        )

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


@jax.jit
def update_model(state, lr, hr, rng):
    def loss_fn(params):
        recon = state.apply_fn({'params': params}, lr, training=True, rngs={'dropout': rng, 'droppath': rng})
        # compute L1 loss
        loss = jnp.mean(
            jnp.abs(hr - recon)
        )
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    loss, grads = grad_fn(state.params)

    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


@jax.jit
def update_critic(state, critic_state, lr, hr, rng, lambda_weights=10.):
    rng1, rng2 = jax.random.split(rng, 2)

    def loss_fn(params, critic_params):
        recon = state.apply_fn({'params': params}, lr, rngs={'dropout': rng1, 'droppath': rng1})

        true_logit = critic_state.apply_fn({'params': critic_params}, hr)
        fake_logit = critic_state.apply_fn({'params': critic_params}, recon)

        # compute gradient penalty
        epsilon = jax.random.uniform(rng2, shape=(hr.shape[0], 1, 1, 1), minval=0., maxval=1.)
        x_hat = epsilon * hr + (1 - epsilon) * recon
        gp_grads = jax.vmap(jax.grad(lambda x: state.apply_fn({'params': critic_params}, x)))(x_hat)
        gp_l2norm = jnp.sqrt(jnp.sum(jnp.square(gp_grads), axis=(1, 2, 3)))
        gp = jnp.mean(jnp.square(gp_l2norm - 1))

        # compute Wasserstein loss and add gradient penalty * lambda
        loss = jnp.mean(
            fake_logit
        ) - jnp.mean(
            true_logit
        ) + (
            gp * lambda_weights
        )
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    loss, grads = grad_fn(state.params, critic_state.params)

    new_critic_state = critic_state.apply_gradients(grads=grads)
    return loss, new_critic_state, rng1


@jax.jit
def apply_generator(state, critic_state, lr, hr, rng, alpha, beta):
    def loss_fn(params, critic_params):
        recon = state.apply_fn({'params': params}, lr, training=True, rngs={'dropout': rng, 'droppath': rng})
        fake_logit = critic_state.apply_fn({'params': critic_params}, recon)
        # L1 Loss
        recon_loss = jnp.mean(
            jnp.abs(recon - hr)
        )
        # Wasserstein Loss
        gen_loss = -jnp.mean(
            fake_logit
        )
        return alpha * recon_loss + beta * gen_loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    loss, grads = grad_fn(state.params, critic_state.params)

    new_gen_state = state.apply_gradients(grads=grads)
    return loss, new_gen_state


@jax.jit
def evaluate_model(state, lr, hr, max_val):
    recon = state.apply_fn({'params': state.params}, lr)
    psnr, ssim = compute_metrics(recon, hr, max_val)
    return psnr, ssim


def nafnet_train_epochs(
        model_state,
        rng,
        train_ds,
        batch_size,
        scale
):
    s, h, w, c = train_ds.shape
    steps_per_epoch = s // batch_size

    rng = jax.random.split(rng, 2)[0]
    perms = jax.random.permutation(rng, s)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_model_loss = []

    for perm in tqdm(perms):
        rng = jax.random.split(rng, 2)[0]
        batch_labels = train_ds[perm, ...]
        batch_hr = get_patches(rng, batch_labels, 48)
        batch_lr = downsample_bicubic(batch_hr, scale)

        loss, model_state = update_model(
            model_state,
            batch_lr,
            batch_hr,
            rng
        )
        epoch_model_loss.append(loss)

    epoch_model_loss = sum(epoch_model_loss) / len(epoch_model_loss)
    return model_state, epoch_model_loss


def nafnet_gan_train_epochs(
        model_state,
        critic_state,
        rng,
        train_ds,
        batch_size,
        scale,
        alpha,
        beta,
):
    s, h, w, c = train_ds.shape
    steps_per_epoch = s // batch_size

    rng = jax.random.split(rng, 2)[0]
    perms = jax.random.permutation(rng, s)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_gan_loss = []
    epoch_gen_loss = []

    for perm in tqdm(perms):
        rng = jax.random.split(rng, 2)[0]
        batch_labels = train_ds[perm, ...]
        batch_hr = get_patches(rng, batch_labels, 48)
        batch_lr = jax.image.resize(
            batch_hr,
            (batch_size, 48 // scale, 48 // scale, c),
            method='bicubic'
        )

        for _ in range(5):
            gan_loss, critic_state, rng = update_critic(
                model_state,
                critic_state,
                batch_lr,
                batch_hr,
                rng
            )
            epoch_gan_loss.append(gan_loss)
        gen_loss, model_state = apply_generator(
            model_state,
            critic_state,
            batch_lr,
            batch_hr,
            rng,
            alpha,
            beta
        )
        epoch_gen_loss.append(gen_loss)

    epoch_gan_loss = sum(epoch_gan_loss) / len(epoch_gan_loss)
    epoch_gen_loss = sum(epoch_gen_loss) / len(epoch_gen_loss)
    return model_state, critic_state, epoch_gan_loss, epoch_gen_loss