import tensorflow as tf
import copy
import tensorflow_addons as tfa
import numpy as np
from typing import Optional, List
import time
import argparse
from sr_archs import (
    sisr, misr
)
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
import pywt


def load_datasets(dataset, size, t=None, run_hpo=False):
    # 하이퍼 파라미터 튜닝할 때는 current directory가 달라지기에 불러오는 방법을 다르게 함.
    # 복잡한 이미지(High Total Variance)를 데이터셋으로 사용하는 경우와
    # 덜 복잡한 이미지(Low Total Variance)를 데이터셋으로 사용하는 경우.
    if run_hpo:
        if dataset == 'lowtv':
            train = np.load('../../../../data/preprocessed/lowtv/train.npy')
            valid_1 = np.load('../../../../data/preprocessed/lowtv/valid_1.npy')
            valid_2 = np.load('../../../../data/preprocessed/lowtv/valid_2.npy')
            test_1 = np.load('../../../../data/preprocessed/lowtv/test_1.npy')
            test_2 = np.load('../../../../data/preprocessed/lowtv/test_2.npy')
        else:
            train = np.load('../../../../data/preprocessed/hightv/train.npy')
            valid_1 = np.load('../../../../data/preprocessed/hightv/valid_1.npy')
            valid_2 = np.load('../../../../data/preprocessed/hightv/valid_2.npy')
            test_1 = np.load('../../../../data/preprocessed/hightv/test_1.npy')
            test_2 = np.load('../../../../data/preprocessed/hightv/test_2.npy')
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

    # 시점 개수를 주면(Multi image) (B, H, W, T) 이미지 반환. 아니면 (B, H, W, 1) 반환.
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

    return train[:, :size, :size, :], valid[:, :size, :size, :], test[:, :size, :size, :]


# TF Dataset에서 Mapping을 위해 쓰이는 함수.
# SISR Train 시 Augmentation 및 주어진 데이터에서 X와 y를 생성함.
def preprocessing(hr, crop_size, scale):
    hr = tf.image.random_crop(
        tf.expand_dims(hr, 0), (1, crop_size, crop_size, 1)
    )
    lr = tf.image.resize(
        hr, size=(crop_size//scale, crop_size//scale),
        method=tf.image.ResizeMethod.BICUBIC
    )[0, :, :, :]
    return lr, hr[0, :, :, :]


# TF Dataset에서 Mapping을 위해 쓰이는 함수.
# SISR Val/Test 시 주어진 데이터에서 X와 y를 생성함.
def eval_preprocessing(hr, scale):
    h, w, c = hr.get_shape().as_list()
    lr = tf.image.resize(
        tf.expand_dims(hr, axis=0),
        size=(h//scale, w//scale),
        method=tf.image.ResizeMethod.BICUBIC
    )[0, :, :, :]
    return lr, hr


# TF Dataset에서 Mapping을 위해 쓰이는 함수.
# MISR Train 시 Augmentation 및 주어진 데이터에서 X와 y를 생성함.
def ts_preprocessing(hr, crop_size, scale):
    h, w, t = hr.get_shape().as_list()

    hr = tf.image.random_crop(hr, (crop_size, crop_size, t))

    hr = tf.expand_dims(hr, 0)
    lr = tf.image.resize(
        hr, size=(crop_size//scale, crop_size//scale),
        method=tf.image.ResizeMethod.BICUBIC
    )[0, :, :, :]
    # 시점의 가장 마지막(최신) 이미지가 y로 들어감.
    label = hr[0, :, :, -1:]
    return lr, label


# TF Dataset에서 Mapping을 위해 쓰이는 함수.
# MISR Val/Test 시 주어진 데이터에서 X와 y를 생성함.
def ts_eval_preprocessing(hr, scale):
    h, w, c = hr.get_shape().as_list()
    hr = tf.expand_dims(hr, 0)
    lr = tf.image.resize(
        hr, size=(h//scale, w//scale),
        method=tf.image.ResizeMethod.BICUBIC
    )[0, :, :, :]
    # 시점의 가장 마지막(최신) 이미지가 y로 들어감.
    label = hr[0, :, :, -1:]
    return lr, label


# Reconstruction(y_hat)과 HR(y)가 주어지면 Metric(PSNR, SSIM)을 계산하여 반환함.
def compute_metrics(recon, hr, max_val):
    psnr = tf.reduce_mean(tf.image.psnr(recon, hr, max_val=max_val))
    ssim = tf.reduce_mean(tf.image.ssim(recon, hr, max_val=max_val))
    return psnr, ssim


# 전체 데이터셋에 대해 Train/Validation/Test Dataset을 normalize하기 위함.
def normalize(
        x,
        x_min=None,
        x_max=None
):
    if x_min is None:
        x_min = x.min()
        x_max = x.max()

    x_scaled = (x - x_min) / (x_max - x_min)
    return x_scaled, x_min, x_max


# 전체 데이터셋에 대해 Train/Validation/Test Dataset을 normalize하기 위함.
def inverse_normalize(
        x,
        x_min,
        x_max
):
    x_inversed = x * (x_max-x_min) + x_min
    return x_inversed


# Single Image로 이뤄진 Dataset을 주어진 시점만큼 Multi Image로 바꿔줌.
def make_timeseries(
        data, t
):
    time_list = [data[i: -(t - i), :, :, :] for i in range(t)]
    return np.concatenate(time_list, axis=-1)


# TR-MISR 논문에서 Validation metric이 3회동안 증가하지 않으면 LR을 .05% drop함.
class TRMISR_callback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.patience = 0
        self.best = tf.DType(1.).min

    def on_train_begin(self, logs=None):
        self.patience = 0
        self.best = tf.DType(1.).min

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_psnr')

        def _good_cond():
            self.patience = 0
            self.best = current

        def _bad_cond():
            self.patience += 1
            if self.patience >= 3:
                self.patience = 0
                ed_lr = float(tf.keras.backend.get_value(self.model.ed_optimizer.lr))
                new_ed_lr = ed_lr * .95
                tf.keras.backend.set_value(self.model.ed_optimizer.lr, new_ed_lr)
                fu_lr = float(tf.keras.backend.get_value(self.model.fu_optimizer.lr))
                new_fu_lr = fu_lr * .95
                tf.keras.backend.set_value(self.model.fu_optimizer.lr, new_fu_lr)

                print('\nLR dropped by .05')

        tf.cond(
            tf.math.less(self.best, current),
            _good_cond,
            _bad_cond
        )


def return_interpolate_weights(
        psnr_weights, gan_weights, alpha: float = .2
):
    weights = []
    for psnr_weight, gan_weight in zip(psnr_weights, gan_weights):
        weights.append((1 - alpha) * psnr_weight + alpha * gan_weight)

    return weights


def calcuate_wt_rmse(hr, sr):
    # Get high frequency component
    _, (_, _, hr_hh) = pywt.dwt2(hr, 'haar', axes=(1, 2))
    _, (_, _, sr_hh) = pywt.dwt2(sr, 'haar', axes=(1, 2))

    # Calculate ergas on high frequency component
    rmse = np.sqrt(
        np.mean(
            np.square(hr_hh - sr_hh)
        )
    )
    return rmse
