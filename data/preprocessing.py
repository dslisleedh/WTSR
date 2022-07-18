from netCDF4 import Dataset as NetCDFFile
import numpy as np
from tqdm import tqdm
import os
import natsort


# 주어진 netCDF File에서 Numpy array 추출
def _get_data_from_cdf(
        val_rate: float = .15,
        test_rate: float = .15,
        data_path: str = './data/raw/',
        save_path: str = './data/preprocessed/'
):
    # 원래는 파일이 여러개일줄알고 이렇게 만들었는데 하나에 다 들어와서 굳이 이렇게 짤 필요 없음.
    # 추후 데이터 추가시 수정 필요.
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

    # 복잡도가 증가하는 부분의 경계를 인덱스로 잘라서 데이터분리함.
    # 수치적으로 잘라낼 방법이 안보였음(Total Variation의 차이로 안걸러짐)
    lowtv, hightv = data[:4453], data[4453:]

    # 각 데이터에서 일정 부분을 잘라 Train/Validation/Test로 분리
    # 이미지화 하기 위해 3차원(B, H, W) 데이터를 4차원(B, H, W, 1)로 늘려줌.
    num_instances = len(lowtv)
    test = np.expand_dims(lowtv[-int(num_instances * test_rate):, :, :], axis=-1)
    valid = np.expand_dims(lowtv[-int(num_instances * (test_rate + val_rate)):-int(num_instances * test_rate), :, :],
                           axis=-1)
    train = np.expand_dims(lowtv[:-int(num_instances * (test_rate + val_rate)), :, :], axis=-1)
    valid_1, valid_2 = valid[: int(.5 * len(valid))], valid[int(.5 * len(valid)):]
    test_1, test_2 = test[: int(.5 * len(test))], valid[int(.5 * len(test)):]
    np.save(save_path + '/lowtv/train.npy', train)
    np.save(save_path + '/lowtv/valid_1.npy', valid_1)
    np.save(save_path + '/lowtv/valid_2.npy', test_1)
    np.save(save_path + '/lowtv/test_1.npy', valid_2)
    np.save(save_path + '/lowtv/test_2.npy', test_2)

    num_instances = len(hightv)
    test = np.expand_dims(hightv[-int(num_instances * test_rate):, :, :], axis=-1)
    valid = np.expand_dims(hightv[-int(num_instances * (test_rate + val_rate)):-int(num_instances * test_rate), :, :],
                           axis=-1)
    train = np.expand_dims(hightv[:-int(num_instances * (test_rate + val_rate)), :, :], axis=-1)
    valid_1, valid_2 = valid[: int(.5 * len(valid))], valid[int(.5 * len(valid)):]
    test_1, test_2 = test[: int(.5 * len(test))], valid[int(.5 * len(test)):]
    np.save(save_path + '/hightv/train.npy', train)
    np.save(save_path + '/hightv/valid_1.npy', valid_1)
    np.save(save_path + '/hightv/valid_2.npy', test_1)
    np.save(save_path + '/hightv/test_1.npy', valid_2)
    np.save(save_path + '/hightv/test_2.npy', test_2)

if __name__ == "__main__":
    _get_data_from_cdf()
