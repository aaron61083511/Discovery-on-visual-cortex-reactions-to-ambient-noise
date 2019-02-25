import h5py
import numpy as np
import data_cleaning_tools
from matplotlib import pyplot as plt

filename = "../data/19120_2_00004_spikes_and_times_v2_preprocessed.h5"

with h5py.File(filename, 'r') as f:
    print('Available datasets:', list(f.keys()))
    cell_traces = np.array(f['cell_traces'])
    cell_xyz = np.array(f['cell_xyz'])
    frame_times = np.array(f['frame_times'])
    pupil_radius = np.array(f['pupil_radius'])
    pupil_xy = np.array(f['pupil_xy'])
    treadmill_velocity = np.array(f['treadmill_velocity'])
    audio_peak_8750 = np.array(f['audio_peak_8750'])
    audio_peak_12000 = np.array(f['audio_peak_12000'])
    audio_peak_13700 = np.array(f['audio_peak_13700'])

h5f_training = h5py.File(filename[:-3] + "_training" + filename[-3:], "w")
h5f_testing = h5py.File(filename[:-3] + "_testing" + filename[-3:], "w")

audio_length = frame_times[-1] - frame_times[0]
sr = 1 / (audio_length / len(frame_times))

training_times, testing_times = data_cleaning_tools.split_audio_events(audio_peak_8750 + audio_peak_13700)
pupil_radius_training, pupil_radius_testing = data_cleaning_tools.split_data(pupil_radius, training_times, testing_times)
treadmill_velocity_training, treadmill_velocity_testing = data_cleaning_tools.split_data(treadmill_velocity, training_times, testing_times)
audio_peak_8750_training, audio_peak_8750_testing = data_cleaning_tools.split_data(audio_peak_8750, training_times, testing_times)
audio_peak_12000_training, audio_peak_12000_testing = data_cleaning_tools.split_data(audio_peak_12000, training_times, testing_times)
audio_peak_13700_training, audio_peak_13700_testing = data_cleaning_tools.split_data(audio_peak_13700, training_times, testing_times)
cell_traces_training, cell_traces_testing = data_cleaning_tools.split_data(cell_traces.transpose(), training_times, testing_times)
cell_traces_training, cell_traces_testing = cell_traces_training.transpose(), cell_traces_testing.transpose()

h5f_training.create_dataset('frame_times', data=training_times / sr)
h5f_training.create_dataset('pupil_radius', data=pupil_radius_training)
h5f_training.create_dataset('treadmill_velocity', data=treadmill_velocity_training)
h5f_training.create_dataset('audio_peak_8750', data=audio_peak_8750_training)
h5f_training.create_dataset('audio_peak_12000', data=audio_peak_12000_training)
h5f_training.create_dataset('audio_peak_13700', data=audio_peak_13700_training)
h5f_training.create_dataset('cell_traces', data=cell_traces_training)
h5f_testing.create_dataset('frame_times', data=testing_times / sr)
h5f_testing.create_dataset('pupil_radius', data=pupil_radius_testing)
h5f_testing.create_dataset('treadmill_velocity', data=treadmill_velocity_testing)
h5f_testing.create_dataset('audio_peak_8750', data=audio_peak_8750_testing)
h5f_testing.create_dataset('audio_peak_12000', data=audio_peak_12000_testing)
h5f_testing.create_dataset('audio_peak_13700', data=audio_peak_13700_testing)
h5f_testing.create_dataset('cell_traces', data=cell_traces_testing)

h5f_training.close()
h5f_testing.close()
