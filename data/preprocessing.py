from netCDF4 import Dataset as NetCDFFile
import numpy as np
from tqdm import tqdm
import os
import natsort


def _get_data_from_cdf(
        is_make_timeseries: bool = True,
        t: int = 5,
        val_rate: float = .15,
        test_rate: float = .15,
        data_path: str = './data/raw/',
        save_path: str = './data/preprocessed/'
):
    filenames = os.listdir(data_path)
    filenames = natsort.natsorted(filenames, reverse=True)
    print('loading files ...')
    print(f'files found: {len(filenames)}')
    data = []
    for fname in tqdm(filenames):
        mnc = NetCDFFile(data_path+fname)
        # cut data into 100x100 grid and subtract K
        data.append(np.array(mnc.variables['analysed_sst'][:])[:, :100, :100] - 273.15)
        mnc.close()
    data = np.concatenate(data, axis=0)

    num_instances = len(data)
    test = np.expand_dims(data[-int(num_instances * test_rate):, :, :], axis=-1)
    valid = np.expand_dims(data[-int(num_instances * (test_rate + val_rate)):-int(num_instances * test_rate), :, :],
                           axis=-1)
    train = np.expand_dims(data[:-int(num_instances * (test_rate + val_rate)), :, :], axis=-1)

    np.save(save_path + 'train.npy', train)
    np.save(save_path + 'valid.npy', valid)
    np.save(save_path + 'test.npy', test)

    if is_make_timeseries:
        train = _make_timeseries(train, t)
        valid = _make_timeseries(valid, t)
        test = _make_timeseries(valid, t)

        np.save(save_path + 'ts_train.npy', train)
        np.save(save_path + 'ts_valid.npy', valid)
        np.save(save_path + 'ts_test.npy', test)


def _make_timeseries(inputs, t):
    time_list = [inputs[i: -(t - i), :, :, :] for i in range(t)]
    return np.concatenate(time_list, axis=-1)


if __name__ == "__main__":
    _get_data_from_cdf()
