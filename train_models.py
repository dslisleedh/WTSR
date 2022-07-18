import tensorflow as tf
import tensorflow_addons as tfa

from typing import Optional, List
import time
import copy
import os
import time
from tqdm import tqdm
import joblib

import matplotlib.pyplot as plt
import argparse
import ml_collections

import model_funcs
import utils
from sr_archs import (
    ganmodule, sisr, misr
)
from utils import *
from model_funcs import *


# 주어진 Config에 해당하는 모델을 하나 학습함. -> 단순 테스트용 혹은 직접 하나하나 넣어보며 실험하고 싶을 때 사용
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameters to train network')

    parser.add_argument('--random_state', type=int, default=42, help='random_state')
    parser.add_argument('--dataset', type=str, default='lowtv', choices=['lowtv', 'hightv'])
    parser.add_argument('--image_size', type=int, default=100, help='inference image size')
    parser.add_argument('--patch_size', type=int, default=50, help='train patch size')

    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='size of minibatch')
    parser.add_argument('--learning_rate', type=float, default=3e-3, help='size of steps to optimize')
    parser.add_argument('--early_stopping', type=int, default=30, help='patient')

    parser.add_argument('--n_filters', type=int, default=16, help='number of filters in network')
    parser.add_argument('--scale', type=int, default=10, help='upscale rate')
    parser.add_argument('--n_blocks', type=int, default=8, help='number of blocks(naf, rfab)')
    parser.add_argument('--sr_archs', type=str, default='sisr', choices=['sisr', 'misr', 'sisr_swin'],
                        help='select sr_archs to train')
    parser.add_argument('--drop_rate', type=float, default=.2, help='dropout / droppath')

    # NAFNet
    parser.add_argument('--weight_decay', type=float, default=0., help='google AdamW')

    # TR-MISR
    parser.add_argument('--time', type=int, default=5, help='number of lags')
    parser.add_argument('--n_heads', type=int , default=8, help='n_heads')
    parser.add_argument('--n_enc_blocks', type=int, default=2)

    # parser로 주어진 args를 config로 변환.
    args = parser.parse_args()
    cfg_dict = vars(args)
    cfg = ml_collections.config_dict.ConfigDict(cfg_dict)

    tf.random.set_seed(cfg.random_state)

    loss, psnr, ssim = train_models(cfg)
    print(f'Train end!')
    print(f'result loss:{loss}, psnr:{psnr}, ssim:{ssim}')
