import os
import tensorflow as tf
import ml_collections
from utils import *
from sr_archs import (
    ganmodule, sisr, misr
)
import matplotlib.pyplot as plt
import time


# runhpo.py 혹은 train_model.py를 이용해 학습을 돌릴 때 실제로 모델이 돌리는 부분이 정의된 코드.
# If문 중복이 많이 되긴 하나 추후 수정/추가/삭제를 쉽게 하기 위해 블록 형태로 만들었음.

def train_models(config, run_hpo=False):
    tf.keras.backend.clear_session()

    ############################# Block 1 #############################
    # 각 모델이 돌아가는 작업공간 생성 및 이동.
    train_time = time.localtime(time.time())
    if run_hpo:
        work_path = f'./{config.sr_archs}_{train_time[0]}{train_time[1]:02}{train_time[2]:02}_{train_time[3]}_{train_time[4]:02}_{train_time[5]:02}'
    else:
        work_path = f'./logs/{config.sr_archs}/{train_time[0]}{train_time[1]:02}{train_time[2]:02}_{train_time[3]}_{train_time[4]:02}_{train_time[5]:02}'
    os.makedirs(work_path)
    with open(work_path+'/hparams.txt', 'w') as file:
        file.write(str(dict(config)))
    config.update({'work_path': work_path})

    ############################# Block 2 #############################
    # 주어진 Config로 학습을 위한 전처리를 실행.
    if run_hpo:
        if config.sr_archs == 'sisr':
            print('current params: \n')
            print(f'batch_size: {config.batch_size}')
            print(f'learning_rate: {config.learning_rate}')
            print(f'weight_decay: {config.weight_decay}')
            print(f'n_filters: {config.n_filters}')
            print(f'n_blocks: {config.n_blocks}')
            print(f'drop_rate: {config.drop_rate}')
        elif config.sr_archs == 'misr':
            print('current params: \n')
            print(f'batch_size: {config.batch_size}')
            print(f'learning_rate: {config.learning_rate}')
            print(f'n_filters: {config.n_filters}')
            print(f'n_blocks: {config.n_blocks}')
            print(f'drop_rate: {config.drop_rate}')
            print(f'time: {config.time}')
            print(f'n_heads: {config.n_heads}')
            print(f'n_enc_blocks: {config.n_enc_blocks}')
        elif config.sr_archs == 'sisr_swin':
            print('current params: \n')
            print(f'batch_size: {config.batch_size}')
            print(f'learning_rate: {config.learning_rate}')
            print(f'weight_decay: {config.weight_decay}')
            print(f'n_filters: {config.n_filters}')
            print(f'n_blocks: {config.n_blocks}')
            print(f'drop_rate: {config.drop_rate}')
        else:
            raise ValueError("Model selection error. ['sisr', 'misr', 'sisr_swin']")

    # MISR이면 (B, H, W, T)데이터셋 반환. SISR이면 (B, H, W, 1) 데이터셋 반환.
    if config.sr_archs == 'misr':
        train, valid, test = load_datasets(config.dataset, config.image_size, config.time, run_hpo=run_hpo)
    else:
        train, valid, test = load_datasets(config.dataset, config.image_size, run_hpo=run_hpo)
    whole_data = tf.concat([train, valid, test], axis=0)
    # 데이터셋에서 통계값 추출 후 데이터 normalization
    data_min, data_max = tf.reduce_min(whole_data), tf.reduce_max(whole_data)
    del whole_data
    train = normalize(train, data_min, data_max)[0]

    # 사용할 모델에 따라 dataset 생성.
    if config.sr_archs == 'misr':
        train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(buffer_size=len(train)) \
            .map(lambda x: ts_preprocessing(x, config.patch_size, config.scale)) \
            .batch(batch_size=config.batch_size, drop_remainder=True) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        valid = normalize(valid, data_min, data_max)[0]
        valid_ds = tf.data.Dataset.from_tensor_slices(valid).map(lambda x: ts_eval_preprocessing(x, config.scale)) \
            .batch(batch_size=16, drop_remainder=False) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        test = normalize(test, data_min, data_max)[0]
        test_ds = tf.data.Dataset.from_tensor_slices(test).map(lambda x: ts_eval_preprocessing(x, config.scale)) \
            .batch(batch_size=16, drop_remainder=False) \
            .prefetch(tf.data.experimental.AUTOTUNE)
    else:
        train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(buffer_size=len(train)) \
            .map(lambda x: preprocessing(x, config.patch_size, config.scale)) \
            .batch(batch_size=config.batch_size, drop_remainder=True) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        valid = normalize(valid, data_min, data_max)[0]
        valid_ds = tf.data.Dataset.from_tensor_slices(valid).map(lambda x: eval_preprocessing(x, config.scale)) \
            .batch(batch_size=128, drop_remainder=False) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        test = normalize(test, data_min, data_max)[0]
        test_ds = tf.data.Dataset.from_tensor_slices(test).map(lambda x: eval_preprocessing(x, config.scale)) \
            .batch(batch_size=128, drop_remainder=False) \
            .prefetch(tf.data.experimental.AUTOTUNE)

    ############################# Block 3 #############################
    # Metric 정의
    def psnr(y_true, y_pred):
        return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1))

    def ssim(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))

    # Config에 맞는 모델 반환.
    # SISR은 NAFSSR의 세팅을 최대한 따랐으며(SwinIR에서 Optimizer setting을 못찾았음)
    # MISR은 Decay부분 구현이 힘들어서 이부분 빼고 따라함.
    if config.sr_archs == 'sisr':
        model = sisr.NAFNetSR(
            config.scale, config.n_filters, config.n_blocks, config.drop_rate,
            [None, config.patch_size, config.patch_size, 1]
        )
        lr_scheduler = tf.keras.experimental.CosineDecay(
            config.learning_rate, 400000, alpha=3e-4
        )
        model.compile(
            optimizer=tfa.optimizers.AdamW(
                learning_rate=lr_scheduler, weight_decay=config.weight_decay,
                beta_1=.9, beta_2=.9, epsilon=1e-7,
                clipvalue=1.0
            ),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=[psnr, ssim]
        )
    elif config.sr_archs == 'sisr_swin':
        model = sisr.SwinIR(
            n_filters=config.n_filters,
            n_rstb_blocks=config.n_blocks,
            n_stl_blocks=6,
            window_size=4,
            n_heads=6,
            qkv_bias=False,
            drop_rate=config.drop_rate,
            res_connection='3conv',
            upsample_type='pixelshuffle',
            upsample_rate=config.scale,
            img_range=1.,
            output_channel=1,
            mean=tf.reduce_mean(
                train, axis=[0, 1, 2]
            )
        )
        lr_scheduler = tf.keras.experimental.CosineDecay(
            config.learning_rate, 400000, alpha=3e-4
        )
        model.compile(
            optimizer=tfa.optimizers.AdamW(
                learning_rate=lr_scheduler, weight_decay=config.weight_decay,
                beta_1=.9, beta_2=.9, epsilon=1e-7,
                clipvalue=1.0
            ),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=[psnr, ssim]
        )
    elif config.sr_archs == 'misr':
        model = misr.TRNet(
            config.n_filters,
            config.n_enc_blocks,
            config.n_blocks,
            config.n_heads,
            config.n_filters * 2,
            config.drop_rate,
            config.scale
        )
        model.compile(
            ed_optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.learning_rate,
                beta_1=.9, beta_2=.9, epsilon=1e-7,
                clipvalue=1.0
            ),
            fu_optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.learning_rate/2,
                beta_1=.9, beta_2=.9, epsilon=1e-7,
                clipvalue=1.0
            ),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[psnr, ssim]
        )
    else:
        raise ValueError("Model selection error. ['sisr', 'misr', 'sisr_swin']")

    # Callback정의. PSNR이 더이상 감소하지 않으면 학습 중단 및 텐서보드에 학습 기록.
    if config.sr_archs == 'misr':
        model_callbacks = [
            TRMISR_callback(),
            tf.keras.callbacks.EarlyStopping(monitor='val_psnr',
                                             patience=config.early_stopping, restore_best_weights=True,
                                             mode='max'
                                             ),
            tf.keras.callbacks.TensorBoard(log_dir=config.work_path)
        ]
    else:
        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_psnr',
                                             patience=config.early_stopping, restore_best_weights=True,
                                             mode='max'
                                             ),
            tf.keras.callbacks.TensorBoard(log_dir=config.work_path)
        ]
    # 모델 학습.
    model.fit(
        train_ds, validation_data=valid_ds, epochs=config.epochs, callbacks=model_callbacks
    )
    # Fine tuning
    if config.sr_archs == 'misr':
        model_callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=config.work_path)
        ]
        model.fu_optimizer.lr = config.learning_rate / 40
        model.ed_optimizer.lr = config.learning_rate / 20
        model.fit(
            train_ds, validation_data=valid_ds, epochs=20, callbacks=model_callbacks
        )

    # 학습이 끝난 모델의 weights 저장.
    model.save_weights(config.work_path+'/bestmodel_ckpt')

    ############################# Block 4 #############################
    # Test Set에 대해 성능평가.
    test_recon = model.predict(test_ds)
    test_label = test[:, :, :, -1:]
    _, test_h, test_w, _ = test_label.get_shape().as_list()
    test_lr = tf.image.resize(
        test_label,
        size=(test_h//config.scale, test_w//config.scale),
        method=tf.image.ResizeMethod.BICUBIC
    )
    if config.sr_archs == 'sisr':
        test_loss = tf.reduce_mean(
            tf.abs(test_recon - test_label)
        )
    else:
        test_loss = tf.reduce_mean(
            tf.square(test_recon - test_label)
        )
    test_psnr, test_ssim = compute_metrics(test_recon, test_label, max_val=1.)
    test_rmse = tf.reduce_mean(
        tf.sqrt(
            tf.square(
                test_recon - test_label
            ) + 1e-7
        )
    )
    # Bicubic 성능평가.
    bicubic_recon = tf.image.resize(
        test_lr,
        size=(test_h, test_w),
        method=tf.image.ResizeMethod.BICUBIC
    )
    if config.sr_archs == 'sisr':
        bicubic_loss = tf.reduce_mean(
            tf.abs(bicubic_recon - test_label)
        )
    else:
        bicubic_loss = tf.reduce_mean(
            tf.abs(bicubic_recon - test_label)
        )
    bicubic_rmse = tf.reduce_mean(
        tf.sqrt(
            tf.square(
                bicubic_recon - test_label
            ) + 1e-7
        )
    )
    bicubic_psnr, bicubic_ssim = compute_metrics(bicubic_recon, test_label, max_val=1.)

    # 결과 문서로 저장.
    with open(config.work_path+'/results.txt', 'w') as f:
        f.write(f'test_loss: {test_loss} \n')
        f.write(f'test_psnr: {test_psnr} \n')
        f.write(f'test_ssim: {test_ssim} \n')
        f.write(f'test_rmse: {test_rmse} \n')
        f.write(f'bicubic_loss: {bicubic_loss} \n')
        f.write(f'bicubic_psnr: {bicubic_psnr} \n')
        f.write(f'bicubic_ssim: {bicubic_ssim} \n')
        f.write(f'bicubic_rmse: {bicubic_rmse}')
        f.close()

    # 학습 결과 플로팅.
    fig, ax = plt.subplots(10, 4, figsize=(20, 40))
    plt.tight_layout()
    for i in range(10):
        ax[i, 0].pcolor(test[i * 30 if config.dataset == 'lowtv' else i * 17, :, :, -1], vmin=0., vmax=1.,
                        cmap=plt.cm.jet)
        ax[i, 1].pcolor(test_lr[i * 30 if config.dataset == 'lowtv' else i * 17, :, :, 0], vmin=0., vmax=1.,
                        cmap=plt.cm.jet)
        ax[i, 2].pcolor(test_recon[i * 30 if config.dataset == 'lowtv' else i * 17, :, :, 0], vmin=0., vmax=1.,
                        cmap=plt.cm.jet)
        ax[i, 3].pcolor(bicubic_recon[i * 30 if config.dataset == 'lowtv' else i * 17, :, :, 0], vmin=0., vmax=1.,
                        cmap=plt.cm.jet)
    plt.show()
    plt.savefig(config.work_path+'/result.png')
    return test_loss, test_psnr, test_ssim

