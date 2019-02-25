import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, lfilter
import soundfile as sf
import pandas as pd
np.random.seed(435)
from matplotlib import pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Creates a butter filter and returns its bounds.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a butter filter and returns the filtered data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def extract_audio_features(filename, peaks, freq_range, thresh, downsample):
    """
    Loads some audio data, for each peak, filter
    within the freq_range, and threshold 0/1 at thresh.
    :param filename:
    :param peaks:
    :param freq_range:
    :param thresh:
    :return:
    """
    audio_samples, sr = sf.read(filename)
    peak_series = {}
    for freq in peaks:
        filtered_audio = butter_bandpass_filter(audio_samples, freq - freq_range, freq + freq_range, sr)
        high_values = filtered_audio > thresh
        low_values = filtered_audio <= thresh
        filtered_audio[high_values] = 1
        filtered_audio[low_values] = 0
        filtered_audio = filtered_audio[::int(downsample / 10)]
        filtered_audio = maximum_filter1d(filtered_audio, size=5)
        filtered_audio = filtered_audio[::10]
        if freq == 12000:
            neural_start = np.argwhere(filtered_audio == 1)[0][0]
            neural_end = np.argwhere(filtered_audio == 1)[-1][0]
            filtered_audio[neural_start:neural_end] = 1
        peak_series[freq] = np.array(filtered_audio)
    peak_series['audio_length'] = len(audio_samples) / float(sr)
    peak_series['sr'] = sr
    return peak_series


def quantile_normalize(df_input):
    df = df_input.copy()
    dic = {}
    for col in df:
        dic.update({col : sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis=1).tolist()
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df


def normalize_cell_traces(cell_traces, percentile=50):
    """
    Perform median normalization.
    :param cell_traces: [# cells, # samples]
    :return: cell_traces_normalized
    """
    cell_medians = np.median(cell_traces, axis=1)
    cell_percentile = np.percentile(cell_traces, q=percentile, axis=1)
    cell_traces = ((cell_traces.transpose() - cell_medians) / cell_percentile).transpose()
    return np.array(cell_traces)


def split_audio_events(audio_data, splits=30):
    audio_range = np.arange(start=0, stop=len(audio_data), step=1)
    non_audio_event_times = np.where(audio_data == 0)[0]
    non_audio_event_times_split = np.array_split(non_audio_event_times, splits)
    non_audio_event_times_split = [(data_bin[0], data_bin[-1]) for data_bin in non_audio_event_times_split]
    non_audio_event_times_split = np.array([audio_range[data_bin[0]:data_bin[1] + 1]
                                            for data_bin in non_audio_event_times_split])
    training_times_indices = np.random.choice(np.arange(start=0, stop=splits), int(2*splits/3), replace=False)
    testing_times_indices = np.setdiff1d(np.arange(start=0, stop=splits), training_times_indices)
    training_times = non_audio_event_times_split[training_times_indices]
    training_times = np.concatenate(training_times).ravel()
    testing_times = non_audio_event_times_split[testing_times_indices]
    testing_times = np.concatenate(testing_times).ravel()
    return training_times, testing_times


def split_data(data, training_splits, testing_splits):
    if len(data) == 0:
        return data, data
    training_data = data[training_splits]
    testing_data = data[testing_splits]
    return training_data, testing_data
