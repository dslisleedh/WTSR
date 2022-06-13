import flax.jax_utils
import tensorflow as tf
import copy
import tensorflow_addons as tfa
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, checkpoints
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
from model_funcs import *
from flax.metrics import tensorboard
import time
import ml_collections
from flax import serialization
import pickle


def train_models(config):
    if config.sr_archs == 'sisr':
        train_sisr(config)
    elif config.sr_archs == 'misr':
        train_rams(config)
    else:
        raise NotImplementedError("Model selection error. ['sisr', 'misr']")


def train_sisr(config):
    train_time = time.localtime(time.time())
    work_path = f'./logs/{config.sr_archs}_{train_time[0]}_{train_time[1]}_{train_time[2]}_{train_time[3]}_{train_time[4]}_{train_time[5]}'
    os.mkdir(work_path)
    summary_writer = tensorboard.SummaryWriter(work_path)
    summary_writer.hparams(dict(config))

    train, valid, test = load_datasets(config.sr_archs)
    whole_data = jnp.concatenate([train, valid, test], axis=0)
    data_min, data_max = jnp.min(whole_data), jnp.max(whole_data)
    train = normalize(train, data_min, data_max)[0]
    valid = normalize(valid, data_min, data_max)[0]
    valid_lr = downsample_bicubic(valid, config.scale)
    test = normalize(test, data_min, data_max)[0]
    test_lr = downsample_bicubic(test, config.scale)

    n_steps = train.shape[0] // config.batch_size
    rng = jax.random.PRNGKey(config.random_state)

    early_stopping_loss = jnp.inf
    patient = 1
    bestmodel = None

    if config.usegan:
        model_state, critic_state = create_nafnet_train_state(
            rng, n_steps, config
        )
    else:
        model_state = create_nafnet_train_state(
            rng, n_steps, config
        )

    for epoch in range(1, config.epochs+1):
        if config.usegan:
            model_state, critic_state, epoch_gan_loss, epoch_gen_loss = nafnet_gan_train_epochs(
                model_state, critic_state, rng, train, config.batch_size, config.patch_size,
                config.scale, config.alpha, config.beta
            )
            print(
                f'Epoch {epoch}, training gan_loss: {epoch_gan_loss}, training gen_loss: {epoch_gen_loss}'
            )
            summary_writer.scalar('gan_loss', epoch_gan_loss, epoch)
            summary_writer.scalar('gen_loss', epoch_gen_loss, epoch)
        else:
            model_state, epoch_loss = nafnet_train_epochs(
                model_state, rng, train,
                config.batch_size, config.scale, config.patch_size,
            )
            print(
                f'Epoch {epoch}, training recon loss: {epoch_loss}'
            )
            summary_writer.scalar('recon_loss', epoch_loss, epoch)

        rng = jax.random.split(rng, 2)[0]
        loss, psnr, ssim, _ = evaluate_model(model_state, valid_lr, valid, 1., rng)

        # To prevent early stopped too fast
        if epoch == 1:
            loss = 1.

        print(
            f'Epoch {epoch}, Valid loss : {loss}, Valid psnr: {psnr}, Valid ssim: {ssim}'
        )
        summary_writer.scalar('valid_loss', loss, epoch)
        summary_writer.scalar('valid_psnr', psnr, epoch)
        summary_writer.scalar('valid_ssim', ssim, epoch)

        if early_stopping_loss > loss:
            patient = 1
            early_stopping_loss = copy.deepcopy(loss)
            bestmodel = copy.deepcopy(model_state)
        else:
            patient += 1
            if config.early_stopping < patient:
                print(' ##### Early Stopped training ##### ')
                break

    if bestmodel is None:
        bestmodel = model_state
    checkpoints.save_checkpoint(ckpt_dir=work_path, target=bestmodel, step=-1)
    rng = jax.random.split(rng, 2)[0]
    psnr, ssim, loss, test_recon = evaluate_model(bestmodel, test_lr, test, max_val=1., rng=rng)
    b, h, w, c = test.shape
    test_bicubic = jax.image.resize(
        test_lr,
        (b, h, w, c),
        method='bicubic'
    )

    test = inverse_normalize(test, data_min, data_max)
    test_lr = inverse_normalize(test_lr, data_min, data_max)
    test_recon = inverse_normalize(test_recon, data_min, data_max)
    test_bicubic = inverse_normalize(test_bicubic, data_min, data_max)

    summary_writer.scalar('test_loss', loss, 1)
    summary_writer.scalar('test_loss', psnr, 1)
    summary_writer.scalar('test_loss', ssim, 1)

    fig, ax = plt.subplots(10, 4, figsize=(20, 40))
    plt.tight_layout()
    for i in range(10):
        ax[i, 0].pcolor(test[i * 30, :, :, 0], vmin=data_min, vmax=data_max, cmap=plt.cm.jet)
        ax[i, 1].pcolor(test_lr[i * 30, :, :, 0], vmin=data_min, vmax=data_max, cmap=plt.cm.jet)
        ax[i, 2].pcolor(test_recon[i * 30, :, :, 0], vmin=data_min, vmax=data_max, cmap=plt.cm.jet)
        ax[i, 3].pcolor(test_bicubic[i * 30, :, :, 0], vmin=data_min, vmax=data_max, cmap=plt.cm.jet)
    plt.show()
    plt.savefig(work_path+'/result.png')

###########################################################


def train_rams(args):
    tf.random.set_seed(seed=args.random_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameters to train network')

    parser.add_argument('--random_state', type=int, default=42, help='random_state')
    parser.add_argument('--patch_size', type=int, default=50, help='train patch size')

    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='size of minibatch')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='size of steps to optimize')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='google AdamW')
    parser.add_argument('--early_stopping', type=int, default=10, help='patient')

    parser.add_argument('--n_filters', type=int, default=32, help='number of filters in network')
    parser.add_argument('--scale', type=int, default=10, help='upscale rate')
    parser.add_argument('--n_blocks', type=int, default=12, help='number of blocks(naf, rfab)')
    parser.add_argument('--usegan', type=bool, default=False, help='whether to use gan discriminator')
    parser.add_argument('--sr_archs', type=str, default='sisr', help='select sr_archs to train')
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=1.)
    # NAFNet
    parser.add_argument('--stochastic_depth_rate', type=float, default=.1, help='google Droppath/Stochastic_depth')
    # RAMS
    parser.add_argument('--time', type=int, default=5, help='number of lags')

    args = parser.parse_args()
    cfg_dict = vars(args)
    cfg = ml_collections.config_dict.ConfigDict(cfg_dict)

    train_models(cfg)
