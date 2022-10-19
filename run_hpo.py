import optuna
import argparse
import ml_collections
import time
import os
import tensorflow as tf
import model_funcs
import numpy as np
from omegaconf import OmegaConf


# 하이퍼파라미터 튜닝을 실행하는 함수.
# 내부에서 Hyperparameter의 범위를 세팅한 후 실제로 학습을 하는 함수(train_models)를 돌림.
def objective(trials):
    if cfg.sr_archs == 'sisr':
        cfg_params = {
            'batch_size': trials.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'learning_rate': trials.suggest_loguniform('learning_rate', 1e-6, 1e-2),
            'n_filters': trials.suggest_categorical('n_filters', [8, 16, 32, 64, 128]),
            'n_blocks': trials.suggest_int('n_blocks', 4, 36),
            'drop_rate': trials.suggest_float('drop_rate', 0., .4, step=.05),
            # 'use_frloss': trials.suggest_categorical('use_frloss', [True, False]),
            'use_frloss': False
        }
    elif cfg.sr_archs == 'misr_test':
        cfg_params = {
            'batch_size': trials.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'learning_rate': trials.suggest_loguniform('learning_rate', 1e-6, 1e-2),
            'n_filters': trials.suggest_categorical('n_filters', [8, 16, 32, 64, 128]),
            'n_blocks': trials.suggest_int('n_blocks', 4, 24),
            'drop_rate': trials.suggest_float('drop_rate', 0., .4, step=.05),
            'n_heads': trials.suggest_categorical('n_heads', [2, 4, 8]),
            'activation': trials.suggest_categorical('activation', ['simple_gate', 'gelu']),
            'time': trials.suggest_int('time', 3, 7),
        }
    else:
        cfg_params = {
            'batch_size': trials.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'learning_rate': trials.suggest_loguniform('learning_rate', 1e-6, 1e-2),
            'n_filters': trials.suggest_categorical('n_filters', [8, 16, 32, 64, 128]),
            'n_blocks': trials.suggest_int('n_blocks', 4, 24),
            'drop_rate': trials.suggest_float('drop_rate', 0., .4, step=.05),
            'time': trials.suggest_int('time', 3, 9),
            'n_heads': trials.suggest_categorical('n_heads', [2, 4, 8]),
            'n_enc_blocks': trials.suggest_int('n_enc_blocks', 1, 10)
        }

    if cfg.use_gan:
        cfg.update(
            {
                'discriminator_filters': trials.suggest_categorical('discriminator_filters', [4, 8, 16, 32]),
                'discriminator_type': trials.suggest_categorical('discriminator_type', ['sngan', 'ragan']),
                'gan_loss_weight': trials.suggest_categorical('discriminator_type', [1., 1e-1, 1e-2, 1e-3])
            }
        )

    # Config파일에 Hyperparameter를 추가.
    cfg.update(cfg_params)
    output = model_funcs.train_models(cfg, run_hpo=True)
    # Metric을 PSNR로 사용하기에 PSNR 반환.
    return output


if __name__ == '__main__':
    cfg = OmegaConf.load('./config.yaml')

    # # 하이퍼파라미터 튜닝을 위한 세팅
    # parser = argparse.ArgumentParser(description='HPO Setting')
    #
    # parser.add_argument('--random_state', type=int, default=42, help='random_state')
    # parser.add_argument('--n_trials', type=int, default=100)
    #
    # parser.add_argument('--dataset', type=str, default='lowtv', choices=['lowtv', 'hightv'])
    # parser.add_argument('--sr_archs', type=str, default='sisr', help='select sr_archs to train')
    # parser.add_argument('--scale', type=int, default=10, choices=[2, 3, 4, 6, 8, 10], help='upscale rate')
    # parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    # parser.add_argument('--early_stopping', type=int, default=30, help='patient')
    # parser.add_argument('--use_gan', type=bool, default=False, help='weather to use gan')
    #
    # # Scale에 따라 Image/Patch 사이즈가 달라지기에 Config 업데이트
    # args = parser.parse_args()
    # cfg_dict = vars(args)

    if cfg['scale'] in [10, 2]:
        cfg.update({
            'image_size': 100,
            'patch_size': 50
        })
    else:
        cfg.update({
            'image_size': 96,
            'patch_size': 48
        })

    # 시드 설정
    tf.random.set_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    # 학습 폴더 생성 및 이동
    train_time = time.localtime(time.time())
    study_name = f'{train_time[0]}{train_time[1]:02}{train_time[2]:02}_{train_time[3]}_{train_time[4]:02}_{train_time[5]:02}'
    study_work_path = f'./hpo_logs/{cfg.sr_archs}/{cfg.scale}/' + study_name
    os.makedirs(study_work_path)
    os.chdir(study_work_path)

    # PSNR/FID를 Maximize/Minimize 하는 방향으로 TPE sampler를 사용해 Hyperparameter Update.
    study = optuna.create_study(
        study_name=study_name,
        storage=f'sqlite:///{study_name}',
        direction='minimize' if cfg.use_gan else 'maximize',
        sampler=optuna.samplers.TPESampler(
            multivariate=True, seed=cfg.random_state
        )
    )
    study.optimize(objective, n_trials=cfg.n_trials)

    # 학습 완료 후 결과 반환.
    with open('./hpo_results.txt', 'w') as f:
        f.write(f'best trial number: {study.best_trial.number} \n')
        if cfg.use_gan:
            f.write(f'best fid: {study.best_value} \n')
        else:
            f.write(f'best psnr: {study.best_value} \n')
        f.close()

    print('HPO end!')
    print(f'best trial number: {study.best_trial.number}')
    if cfg.use_gan:
        print(f'best fid: {study.best_value}')
    else:
        print(f'best psnr: {study.best_value}')
