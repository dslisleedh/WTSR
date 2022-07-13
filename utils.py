import tensorflow as tf
import copy
import tensorflow_addons as tfa
import numpy as np
from typing import Optional, List
import time
import argparse
from sr_archs import (
    ganmodule, sisr, misr
)
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial


def load_datasets(dataset, t=None, run_hpo=False):
    if run_hpo:
        if dataset == 'lowtv':
            train = np.load('../../data/preprocessed/lowtv/train.npy')
            valid_1 = np.load('../../data/preprocessed/lowtv/valid_1.npy')
            valid_2 = np.load('../../data/preprocessed/lowtv/valid_2.npy')
            test_1 = np.load('../../data/preprocessed/lowtv/test_1.npy')
            test_2 = np.load('../../data/preprocessed/lowtv/test_2.npy')
        else:
            train = np.load('../../data/preprocessed/hightv/train.npy')
            valid_1 = np.load('../../data/preprocessed/hightv/valid_1.npy')
            valid_2 = np.load('../../data/preprocessed/hightv/valid_2.npy')
            test_1 = np.load('../../data/preprocessed/hightv/test_1.npy')
            test_2 = np.load('../../data/preprocessed/hightv/test_2.npy')
    else:
        if dataset == 'lowtv':
            train = np.load('./data/preprocessed/lowtv/train.npy')
            valid_1 = np.load('./data/preprocessed/lowtv/valid_1.npy')
            valid_2 = np.load('./data/preprocessed/lowtv/valid_2.npy')
            test_1 = np.load('./data/preprocessed/lowtv/test_1.npy')
            test_2 = np.load('./data/preprocessed/lowtv/test_2.npy')
        else:
            train = np.load('./data/preprocessed/hightv/train.npy')
            valid_1 = np.load('./data/preprocessed/hightv/valid_1.npy')
            valid_2 = np.load('./data/preprocessed/hightv/valid_2.npy')
            test_1 = np.load('./data/preprocessed/hightv/test_1.npy')
            test_2 = np.load('./data/preprocessed/hightv/test_2.npy')
    if t is None:
        valid = np.concatenate([
            valid_1, valid_2
        ], axis=0)
        test = np.concatenate([
            test_1, test_2
        ], axis=0)
    else:
        train = make_timeseries(train, t)
        valid = np.concatenate([
            make_timeseries(valid_1, t), make_timeseries(valid_2, t)
        ], axis=0)
        test = np.concatenate([
            make_timeseries(test_1, t), make_timeseries(test_2, t)
        ], axis=0)
    return train, valid, test


def preprocessing(hr, crop_size, scale):
    hr = tf.image.random_crop(
        tf.expand_dims(hr, 0), (1, crop_size, crop_size, 1)
    )
    lr = tf.image.resize(
        hr, size=(crop_size//scale, crop_size//scale),
        method=tf.image.ResizeMethod.BICUBIC
    )[0, :, :, :]
    return lr, hr[0, :, :, :]


def eval_preprocessing(hr, scale):
    h, w, c = hr.get_shape().as_list()
    lr = tf.image.resize(
        tf.expand_dims(hr, axis=0),
        size=(h//scale, w//scale),
        method=tf.image.ResizeMethod.BICUBIC
    )[0, :, :, :]
    return lr, hr


def ts_preprocessing(hr, crop_size, scale):
    h, w, t = hr.get_shape().as_list()

    hr = tf.image.random_crop(hr, (crop_size, crop_size, t))

    hr = tf.expand_dims(hr, 0)
    lr = tf.image.resize(
        hr, size=(crop_size//scale, crop_size//scale),
        method=tf.image.ResizeMethod.BICUBIC
    )[0, :, :, :]
    label = hr[0, :, :, -1:]
    return lr, label


def ts_eval_preprocessing(hr, scale):
    hr = tf.expand_dims(hr, 0)
    lr = tf.image.resize(
        hr, size=(100//scale, 100//scale),
        method=tf.image.ResizeMethod.BICUBIC
    )[0, :, :, :]
    label = hr[0, :, :, -1:]
    return lr, label


def compute_metrics(recon, hr, max_val):
    psnr = tf.reduce_mean(tf.image.psnr(recon, hr, max_val=max_val))
    ssim = tf.reduce_mean(tf.image.ssim(recon, hr, max_val=max_val))
    return psnr, ssim


def normalize(
        x,
        x_min=None,
        x_max=None
):
    if x_min is None:
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)

    x_scaled = (x - x_min) / (x_max - x_min)
    return x_scaled, x_min, x_max


def inverse_normalize(
        x,
        x_min,
        x_max
):
    x_inversed = x * (x_max-x_min) + x_min
    return x_inversed


def make_timeseries(
        data, t
):
    time_list = [data[i: -(t - i), :, :, :] for i in range(t)]
    return np.concatenate(time_list, axis=-1)