from netCDF4 import Dataset as NetCDFFile
import numpy as np
from tqdm import tqdm
import os


def get_data_from_cdf(
        is_make_timeseries: bool = True,
        t: int = 5,
        val_rate: float = .2,
        test_rate: float = .15,
        data_path: str = './data/raw/',
        save_path: str = './data/raw/'
):
    filenames = os.listdir(data_path)
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
    test = data[-int(num_instances * test_rate):, :, :]
    valid = data[-int(num_instances * (test_rate + val_rate)):-int(num_instances * test_rate), :, :]
    train = data[:-int(num_instances * (test_rate + val_rate)), :, :]

    np.save(save_path + 'train.npy', train)
    np.save(save_path + 'valid.npy', valid)
    np.save(save_path + 'test.npy', test)

    if is_make_timeseries:
        train = make_timeseries(train, t)
        valid = make_timeseries(valid, t)
        test = make_timeseries(valid, t)

        np.save(save_path + 'ts_train.npy', train)
        np.save(save_path + 'ts_valid.npy', valid)
        np.save(save_path + 'ts_test.npy', test)


def make_timeseries(data, t):
    data = np.expand_dims(data, axis=-1)
    time_list = [data[i:, -(t - i)] for i in range(t)]
    return np.concatenate(time_list, axis=-1)


if __name__ == "__main__":
    get_data_from_cdf()
