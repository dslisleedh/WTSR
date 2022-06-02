import tensorflow as tf
import copy
import tensorflow_addons as tfa
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, List
import optax
import time
import argparse
from sr_archs import (
    ganmodule, nafnet, rams
)
from jax import lax
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Hyperparameters to train network')

parser.add_argument('--random_state', type=int, default=42, help='random_state')

parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=3e-3, help='size of steps to optimize')
parser.add_argument('--weigh_decay', type=float, default=0., help='google AdamW')
parser.add_argument('--early_stopping', type=int, default=10, help='patient')

parser.add_argument('--n_filters', type=int, default=32, help='number of filters in network')
parser.add_argument('--scale', type=int, default=10, help='upscale rate')
parser.add_argument('--n_blocks', type=int, default=12, help='number of blocks(naf, rfab)')
parser.add_argument('--usegan', type=bool, default=False, help='whether to use gan discriminator')
parser.add_argument('--sr_archs', type=str, default='nafnet', help='select sr_archs to train')
# NAFNet
parser.add_argument('--stochstic_depth_rate', type=float, default=.1, help='Google Droppath/Stochastic_depth')
# RAMS
parser.add_argument('--time', type=int, default=5, help='number of lags')


def train_models(args):
    if args.sr_archs == 'nafnet':
        train_nafnet(args)
    elif args.sr_archs == 'rams':
        train_rams(args)
    else:
        raise NotImplementedError("Model selection error. ['nafnet', 'rams']")


def train_nafnet(args):
    train, valid, test = load_datasets(args.sr_archs)

    model = nafnet.NAFNetSR(
        args.scale,
        args.n_filters,
        args.n_blocks,
        args.stochastic_depth_rate
    )
    seed = jax.random.PRNGKey(args.random_state)
    params = model.init(seed, jnp.zeros((1, 48, 48, 1)))

    scheduler = optax.cosine_onecycle_schedule(
        10,
        args.epochs,
        final_div_factor=3e4
    )
    optimizer = optax.adamw(
        learning_rate=scheduler,
        weight_decay=args.weight_decay,
        b1=.9,
        b2=.9
    )
    optimizer_state = optimizer.init(params)

    for i in range(args.epochs):
        print(f'-----------------------epoch{i+1} start-----------------------')

        train = jax.random.permutation(seed, train, axis=0)
        epoch_loss = []





###########################################################




def train_rams(args):
    tf.random.set_seed(seed=args.random_state)


def load_datasets(model: str):
    if model == 'nafnet':
        train = np.load('./data/preprocessed/train.npy')
        valid = np.load('./data/preprocessed/valid.npy')
        test = np.load('./data/preprocessed/test.npy')
    elif model == 'rams':
        train = np.load('./data/preprocessed/ts_train.npy')
        valid = np.load('./data/preprocessed/ts_valid.npy')
        test = np.load('./data/preprocessed/ts_test.npy')
    else:
        raise NotImplementedError("Model selection error. ['nafnet', 'rams']")
    return train, valid, test


if __name__ == "__main__":
    train_models(parser)
