import os
import tensorflow as tf
import ml_collections
from utils import *
import matplotlib.pyplot as plt
import time


def train_models(config, run_hpo=False):
    train_time = time.localtime(time.time())
    if run_hpo:
        work_path = f'./{config.sr_archs}_{train_time[0]}{train_time[1]:02}{train_time[2]:02}_{train_time[3]}_{train_time[4]:02}_{train_time[5]:02}'
    else:
        work_path = f'./logs/{config.sr_archs}/{train_time[0]}{train_time[1]:02}{train_time[2]:02}_{train_time[3]}_{train_time[4]:02}_{train_time[5]:02}'
    os.makedirs(work_path)
    with open(work_path+'/hparams.txt', 'w') as file:
        file.write(str(dict(config)))
    config.update({'work_path': work_path})

    if config.sr_archs == 'sisr':
        loss, psnr, ssim = train_sisr(config, run_hpo=run_hpo)
    elif config.sr_archs == 'misr':
        loss, psnr, ssim = train_misr(config, run_hpo=run_hpo)
    else:
        raise NotImplementedError("Model selection error. ['sisr', 'misr']")
    return loss, psnr, ssim


def train_sisr(config, run_hpo=False):
    if run_hpo:
        print('current params: \n')
        print(f'batch_size: {config.batch_size}')
        print(f'learning_rate: {config.learning_rate}')
        print(f'weight_decay: {config.weight_decay}')
        print(f'n_filters: {config.n_filters}')
        print(f'n_blocks: {config.n_blocks}')
        print(f'drop_rate: {config.drop_rate}')
    train, valid, test = load_datasets(config.dataset, run_hpo=run_hpo)
    whole_data = tf.concat([train, valid, test], axis=0)
    data_min, data_max = tf.reduce_min(whole_data), tf.reduce_max(whole_data)
    del whole_data
    train = normalize(train, data_min, data_max)[0]
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

    model = sisr.NAFNetSR(
        config.scale, config.n_filters, config.n_blocks, config.drop_rate,
        [None, config.patch_size, config.patch_size, 1]
    )

    def psnr(y_true, y_pred):
        return tf.reduce_mean(tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1)))

    def ssim(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))

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

    model_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=config.early_stopping, restore_best_weights=True
                                         ),
        tf.keras.callbacks.TensorBoard(log_dir=config.work_path)
    ]
    model.fit(
        train_ds, validation_data=valid_ds, epochs=config.epochs, callbacks=model_callbacks
    )

    model.save_weights(config.work_path+'/bestmodel_ckpt')
    test_recon = model.predict(test_ds)
    test_label = test[:, :, :, -1:]
    test_lr = tf.image.resize(
        test_label,
        size=(100//config.scale, 100//config.scale),
        method=tf.image.ResizeMethod.BICUBIC
    )
    loss = tf.reduce_mean(
        tf.abs(test_recon - test_label)
    )
    psnr, ssim = compute_metrics(test_recon, test_label, max_val=1.)
    test_rmse = tf.reduce_mean(
        tf.sqrt(
            tf.square(
                test_recon - test_label
            ) + 1e-7
        )
    )
    bicubic_recon = tf.image.resize(
        test_lr,
        size=(100, 100),
        method=tf.image.ResizeMethod.BICUBIC
    )
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

    with open(config.work_path+'/results.txt', 'w') as f:
        f.write(f'test_loss: {loss} \n')
        f.write(f'test_psnr: {psnr} \n')
        f.write(f'test_ssim: {ssim} \n')
        f.write(f'test_rmse: {test_rmse} \n')
        f.write(f'bicubic_loss: {bicubic_loss} \n')
        f.write(f'bicubic_psnr: {bicubic_psnr} \n')
        f.write(f'bicubic_ssim: {bicubic_ssim} \n')
        f.write(f'bicubic_rmse: {bicubic_rmse}')
        f.close()

    fig, ax = plt.subplots(10, 4, figsize=(20, 40))
    plt.tight_layout()

    test = inverse_normalize(test, data_min, data_max)
    test_lr = inverse_normalize(test_lr, data_min, data_max)
    test_recon = inverse_normalize(test_recon, data_min, data_max)
    bicubic_recon = inverse_normalize(bicubic_recon, data_min, data_max)

    for i in range(10):
        ax[i, 0].pcolor(test[i * 30 if config.dataset == 'lowtv' else i * 17, :, :, 0], vmin=data_min, vmax=data_max,
                        cmap=plt.cm.jet)
        ax[i, 1].pcolor(test_lr[i * 30 if config.dataset == 'lowtv' else i * 17, :, :, 0], vmin=data_min, vmax=data_max,
                        cmap=plt.cm.jet)
        ax[i, 2].pcolor(test_recon[i * 30 if config.dataset == 'lowtv' else i * 17, :, :, 0], vmin=data_min, vmax=data_max,
                        cmap=plt.cm.jet)
        ax[i, 3].pcolor(bicubic_recon[i * 30 if config.dataset == 'lowtv' else i * 17, :, :, 0], vmin=data_min,
                        vmax=data_max, cmap=plt.cm.jet)
    plt.show()
    plt.savefig(config.work_path+'/result.png')
    return loss, psnr, ssim


def train_misr(config, run_hpo=False):
    if run_hpo:
        print('current params: \n')
        print(f'batch_size: {config.batch_size}')
        print(f'learning_rate: {config.learning_rate}')
        print(f'n_filters: {config.n_filters}')
        print(f'n_blocks: {config.n_blocks}')
        print(f'drop_rate: {config.drop_rate}')
        print(f'time: {config.time}')
        print(f'n_heads: {config.n_heads}')
        print(f'n_enc_blocks: {config.n_enc_blocks}')

    train, valid, test = load_datasets(config.dataset, config.time, run_hpo=run_hpo)
    whole_data = tf.concat([train, valid, test], axis=0)
    data_min, data_max = tf.reduce_min(whole_data), tf.reduce_max(whole_data)
    del whole_data
    train = normalize(train, data_min, data_max)[0]
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

    model = misr.TRNet(
        config.n_filters,
        config.n_enc_blocks,
        config.n_blocks,
        config.n_heads,
        config.n_filters*2,
        config.drop_rate,
        config.scale
    )

    def psnr(y_true, y_pred):
        return tf.reduce_mean(tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1)))

    def ssim(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))

    ed_lr_scheduler = tf.keras.experimental.CosineDecay(
        config.learning_rate, 400000, alpha=3e-4
    )
    fu_lr_scheduler = tf.keras.experimental.CosineDecay(
        config.learning_rate/2, 400000, alpha=3e-4/2
    )
    model.compile(
        ed_optimizer=tf.keras.optimizers.Adam(
            learning_rate=ed_lr_scheduler,
            beta_1=.9, beta_2=.9, epsilon=1e-7,
            clipvalue=1.0
        ),
        fu_optimizer=tf.keras.optimizers.Adam(
            learning_rate=fu_lr_scheduler,
            beta_1=.9, beta_2=.9, epsilon=1e-7,
            clipvalue=1.0
        ),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[psnr, ssim]
    )

    model_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=config.early_stopping, restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=config.work_path)
    ]
    model.fit(
        train_ds, validation_data=valid_ds, epochs=config.epochs, callbacks=model_callbacks
    )

    model.save_weights(config.work_path+'/bestmodel_ckpt')
    test_recon = model.predict(test_ds)
    test_label = test[:, :, :, -1:]
    test_lr = tf.image.resize(
        test_label,
        size=(100//config.scale, 100//config.scale),
        method=tf.image.ResizeMethod.BICUBIC
    )
    loss = tf.reduce_mean(
        tf.abs(test_recon - test_label)
    )
    psnr, ssim = compute_metrics(test_recon, test_label, max_val=1.)
    test_rmse = tf.reduce_mean(
        tf.sqrt(
            tf.square(
                test_recon - test_label
            ) + 1e-7
        )
    )
    bicubic_recon = tf.image.resize(
        test_lr,
        size=(100, 100),
        method=tf.image.ResizeMethod.BICUBIC
    )
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

    with open(config.work_path+'/results.txt', 'w') as f:
        f.write(f'test_loss: {loss} \n')
        f.write(f'test_psnr: {psnr} \n')
        f.write(f'test_ssim: {ssim} \n')
        f.write(f'test_rmse: {test_rmse} \n')
        f.write(f'bicubic_loss: {bicubic_loss} \n')
        f.write(f'bicubic_psnr: {bicubic_psnr} \n')
        f.write(f'bicubic_ssim: {bicubic_ssim} \n')
        f.write(f'bicubic_rmse: {bicubic_rmse}')
        f.close()

    fig, ax = plt.subplots(10, 4, figsize=(20, 40))
    plt.tight_layout()

    test = inverse_normalize(test, data_min, data_max)
    test_lr = inverse_normalize(test_lr, data_min, data_max)
    test_recon = inverse_normalize(test_recon, data_min, data_max)
    bicubic_recon = inverse_normalize(bicubic_recon, data_min, data_max)

    for i in range(10):
        ax[i, 0].pcolor(test[i * 30 if config.dataset == 'lowtv' else i * 17, :, :, 0], vmin=data_min, vmax=data_max,
                        cmap=plt.cm.jet)
        ax[i, 1].pcolor(test_lr[i * 30 if config.dataset == 'lowtv' else i * 17, :, :, 0], vmin=data_min, vmax=data_max,
                        cmap=plt.cm.jet)
        ax[i, 2].pcolor(test_recon[i * 30 if config.dataset == 'lowtv' else i * 17, :, :, 0], vmin=data_min, vmax=data_max,
                        cmap=plt.cm.jet)
        ax[i, 3].pcolor(bicubic_recon[i * 30 if config.dataset == 'lowtv' else i * 17, :, :, 0], vmin=data_min,
                        vmax=data_max, cmap=plt.cm.jet)

    plt.show()
    plt.savefig(config.work_path+'/result.png')
    return loss, psnr, ssim