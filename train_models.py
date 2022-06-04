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


parser = argparse.ArgumentParser(description='Hyperparameters to train network')

parser.add_argument('--random_state', type=int, default=42, help='random_state')

parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='size of minibatch')
parser.add_argument('--learning_rate', type=float, default=3e-3, help='size of steps to optimize')
parser.add_argument('--weigh_decay', type=float, default=0., help='google AdamW')
parser.add_argument('--early_stopping', type=int, default=10, help='patient')

parser.add_argument('--n_filters', type=int, default=32, help='number of filters in network')
parser.add_argument('--scale', type=int, default=10, help='upscale rate')
parser.add_argument('--n_blocks', type=int, default=12, help='number of blocks(naf, rfab)')
parser.add_argument('--usegan', type=bool, default=False, help='whether to use gan discriminator')
parser.add_argument('--sr_archs', type=str, default='sisr', help='select sr_archs to train')
# NAFNet
parser.add_argument('--stochstic_depth_rate', type=float, default=.1, help='google Droppath/Stochastic_depth')
# RAMS
parser.add_argument('--time', type=int, default=5, help='number of lags')


def train_models(args_parsed):
    if args_parsed.sr_archs == 'sisr':
        train_nafnet(parser)
    elif args_parsed.sr_archs == 'misr':
        train_rams(parser)
    else:
        raise NotImplementedError("Model selection error. ['sisr', 'misr']")


def train_nafnet(args_parsed):
    train, valid, test = load_datasets(args_parsed.sr_archs)
    n_steps = train//args_parsed.batch_size

    rng = jax.random.PRNGKey(args_parsed.random_state)




def nafnet_epochs(
        model,
        model_state,
        rng,
        train_ds,
        batch_size,
        scale
):
    s, h, w, c = train_ds.shape
    steps_per_epoch = s // batch_size

    perms = jax.random.permutation(rng, s)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        rng = jax.random.split(rng, 2)[0]
        batch_labels = train_ds[perm, ...]
        batch_lr = jax.image.resize(
            get_patches(rng, batch_labels, 48),
            (batch_size, 48 // scale, 48 // scale, c),
            method='bicubic'
        )





###########################################################




def train_rams(args):
    tf.random.set_seed(seed=args.random_state)





if __name__ == "__main__":
    train_models(parser)
