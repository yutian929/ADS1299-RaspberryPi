# 2023/07/28 zjn: 输入格式改为16*63
# 2023/07/28 zyt: STFT only keep 0~24Hz signal 16*63

import os.path
import mne
from mne.preprocessing import (ICA)
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('TkAgg')


def stft_8chs(data_frame):
    # store the final 8 CHs stft data
    list_stft = []

    # Set parameters
    sampling_rate = 250  # Sampling rate: 250 Hz
    signal_length = 5  # Signal length: 5 seconds
    total_samples = sampling_rate * signal_length

    for idx, ch in enumerate(data_frame):
        # Extract EEG signal
        eeg_signal = data_frame[ch].values

        # Convert EEG signal to a PyTorch tensor
        eeg_tensor = torch.tensor(eeg_signal[:total_samples], dtype=torch.float32)

        # Perform STFT
        window_length = 160
        hop_length = 20
        stft_result = torch.stft(eeg_tensor, window_length, hop_length, return_complex=True)

        # Extract magnitude spectrum
        magnitude = torch.abs(stft_result)
        magnitude = magnitude.numpy()

        # Filter out frequencies above 20 Hz
        max_freq_index = int(24 * (window_length - 1) / sampling_rate) + 1
        magnitude = magnitude[:max_freq_index]
        list_stft.append(magnitude)

    stft_8chs_ndarray = np.array(list_stft)

    return stft_8chs_ndarray


def normalize_stft_8chs(stft_8chs_ndarray):
    #  turn list_stft_8chs into PyTorch Tensor
    data_tensor = torch.tensor(stft_8chs_ndarray, dtype=torch.float32)

    # calc the min and max
    min_value = data_tensor.min()
    max_value = data_tensor.max()

    # normalize
    normalized_data_tensor = (data_tensor - min_value) / (max_value - min_value)

    # return ndarray
    normalized_stft_8chs_ndarray = normalized_data_tensor.numpy()

    return normalized_stft_8chs_ndarray


def draw_normalize_stft_8chs(normalized_stft_8chs_ndarray, columns):
    # Define subplot layout
    plt.figure(figsize=(14, 10))
    rows, cols = 2, 4
    signal_length = 5
    for idx, ch in enumerate(normalized_stft_8chs_ndarray):
        # Plot the magnitude spectrum of the STFT result on the corresponding subplot
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(ch, aspect='auto', origin='lower',
                   extent=[0, signal_length, 0, 24])
        plt.colorbar(label='Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('STFT of EEG Signal - ' + str(columns[idx]))

    # Adjust spacing between subplots for better layout
    plt.tight_layout()
    plt.show()


def stft(data_frame, enable_draw=False):
    """
    data_frame :param 8chs EEG signal after filtering
    enable_draw :param draw the normalized stft pic
    NORMALIZED_STFT_8CHS_NDARRAY, COLUMNS :return easy to know
    @ZakiZhang 2023/07/25
    eg:
    file_path = 'eg2.xls'
    sheet_name = 'Sheet1'
    DATA_FRAME = pd.read_excel(file_path, sheet_name=sheet_name)
    stft(DATA_FRAME, enable_draw=True)
    """
    # read columns
    COLUMNS = data_frame.columns

    # stft
    STFT_8CHS_NDARRAY = stft_8chs(data_frame)
    # print(STFT_8CHS_NDARRAY.shape)

    # normalize
    NORMALIZED_STFT_8CHS_NDARRAY = normalize_stft_8chs(STFT_8CHS_NDARRAY)
    # print(NORMALIZED_STFT_8CHS_NDARRAY.shape)

    if enable_draw:
        # draw
        draw_normalize_stft_8chs(NORMALIZED_STFT_8CHS_NDARRAY, COLUMNS)

    return NORMALIZED_STFT_8CHS_NDARRAY, COLUMNS


def run_ica(data_array, sfreq=250, low_freq=5., high_freq=25., need_ica=True):
    if data_array.shape[0] > 2000:
       data_array = data_array[500:, :]
    # 将NumPy数组重新调整为形状为(8, )的数组
    data = data_array.T
    info = mne.create_info(
        ch_names=['P3', 'Pz', 'P4', 'T5', 'O1', 'OZ', 'O2', 'T6'],
        ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg'],
        sfreq=sfreq
    )
    custom_raw = mne.io.RawArray(data, info)
    #print(custom_raw)
    #滤波操作
    custom_raw.filter(low_freq, high_freq, fir_design='firwin')
   # custom_raw.plot_psd()
    if need_ica == True:
        ica = ICA(n_components=8, random_state=97)
        # 对滤波过的raw进行ICA
        ica.fit(custom_raw)
        ica.exclude = [0, 1]
        # 去除伪迹
        ica.apply(custom_raw)
    filt_data, times = custom_raw[:]
   # custom_raw.plot(n_channels=8,title='after')
   # plt.show()
    return filt_data


def preprocessing_all(data_array):
    data = run_ica(data_array=data_array)
    time_window_size = 1250
    original_shape = data.shape
    # 计算新数组中第一个维度 n 的大小
    n = original_shape[1] // time_window_size
    # 重新组织数组形状为 (n, 8, 1250)
    data = np.reshape(data[:, :n * time_window_size], (n, 8, time_window_size))
    final_data = np.zeros([n, 8, 63, 63])
    # print(final_data.shape)
    # 遍历第0维后对第一维和第二维的数据进行 DataFrame 操作
    for i in range(n):
        # 获取每个数据块的第一维和第二维数据
        chunk_data = data[i, :, :]
        # 转置数据，使其形状为 (1250, 8)
        chunk_data = chunk_data.T
        # 创建 DataFrame
        columns = ['P3', 'Pz', 'P4', 'T5', 'O1', 'OZ', 'O2', 'T6']
        filt_df = pd.DataFrame(chunk_data, columns=columns)
        # 在这里你可以对 filt_df 进行你想要的 DataFrame 操作
        normalize_stft_8chs_data, _ = stft(filt_df)
        final_data[i] = normalize_stft_8chs_data
        # print(normalize_stft_8chs_data.shape)
    # print(final_data.shape)
    return final_data

def preprocessing_single(data_array, ica=True):
    data = run_ica(data_array=data_array, need_ica=ica)
    # 创建 DataFrame
    data = data.T
    columns = ['P3', 'Pz', 'P4', 'T5', 'O1', 'OZ', 'O2', 'T6']
    filt_df = pd.DataFrame(data, columns=columns)
    # 在这里你可以对 filt_df 进行你想要的 DataFrame 操作
    normalize_stft_8chs_data, _ = stft(filt_df)
    #print(normalize_stft_8chs_data.shape)
    return normalize_stft_8chs_data


def get_dataset(input_folder, labels):
    root_folder = input_folder
    data = []
    target = []
    for second_folder in os.listdir(root_folder):
        second_folder_path = os.path.join(root_folder, second_folder)
        if os.path.isdir(second_folder_path):
            for label in labels:
                third_folder_path = os.path.join(second_folder_path, label)
                if os.path.isdir(third_folder_path):
                    for file in os.listdir(third_folder_path):
                        if file.endswith('.xls'):
                            file_path = os.path.join(third_folder_path, file)
                            df = pd.read_excel(file_path, header=None)
                            # 确保数据具有相同的形状
                            data_array = df.values
                            single_data = preprocessing_single(data_array)
                            data.append(single_data)
                            target.append(labels.index(label))  # convert labels to integers

    # Convert data to numpy arrays and scale it
    data =np.array(data)
    target = np.array(target)
    return data, target


# file_path = 'eg2.xls'
# sheet_name = 'Sheet1'
# DATA_FRAME = pd.read_excel(file_path, sheet_name=sheet_name)
# normalized_stft_8chs_ndarray, _ = stft(DATA_FRAME, enable_draw=True)
# print(normalized_stft_8chs_ndarray.shape)


# if __name__ == '__main__':
#     df = pd.read_excel('saved_data5.xls',header=None)
#     # 将DataFrame的值存储为NumPy数组
#     data_array = df.values
#     data = preprocessing_single(data_array)
#     print(data.shape)



