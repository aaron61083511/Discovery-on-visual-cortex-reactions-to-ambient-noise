import data_cleaning_tools
import h5py
import numpy as np
from matplotlib import pyplot as plt

filename = "../data/19120_2_00004_spikes_and_times_v2.h5"

with h5py.File(filename, 'r') as f:
    print('Available datasets:', list(f.keys()))
    cell_delays = np.array(f['cell_delays'])
    cell_traces = np.array(f['cell_traces'])
    cell_xyz = np.array(f['cell_xyz'])
    frame_times = np.array(f['frame_times'])
    pupil_radius = np.array(f['pupil_radius'])
    pupil_times = np.array(f['pupil_times'])
    pupil_xy = np.array(f['pupil_xy'])
    treadmill_times = np.array(f['treadmill_times'])
    treadmill_velocity = np.array(f['treadmill_velocity'])

h5f = h5py.File(filename[:-3] + "_preprocessed" + filename[-3:], "w")

# # Remove NaNs and shift time arrays to start at the beginning of the cell trace recording
cell_delays = np.nan_to_num(cell_delays)
cell_traces = np.nan_to_num(cell_traces)
cell_xyz = np.nan_to_num(cell_xyz)
frame_times = np.nan_to_num(frame_times)
pupil_radius = np.nan_to_num(pupil_radius)
pupil_times = np.nan_to_num(pupil_times)
pupil_xy = np.nan_to_num(pupil_xy)
treadmill_times = np.nan_to_num(treadmill_times)
treadmill_velocity = np.nan_to_num(treadmill_velocity)

h5f.create_dataset('pupil_xy', data=pupil_xy)
h5f.create_dataset('cell_xyz', data=cell_xyz)
h5f.create_dataset('frame_times', data=frame_times - frame_times[0])

# Align recording start for each time series and interpolate to frame_times (cell_data)
recording_start = frame_times[0]
frame_times -= recording_start
treadmill_times -= recording_start
treadmill_times = treadmill_times[treadmill_times >= 0]
treadmill_velocity = treadmill_velocity[-len(treadmill_times):]
treadmill_velocity = np.interp(frame_times, treadmill_times, treadmill_velocity)
pupil_times -= recording_start
pupil_times = pupil_times[pupil_times >= 0]
pupil_radius = pupil_radius[-len(pupil_times):]
if len(pupil_radius != 0):
    pupil_radius = np.interp(frame_times, pupil_times, pupil_radius)

h5f.create_dataset('treadmill_velocity', data=treadmill_velocity)
h5f.create_dataset('pupil_radius', data=pupil_radius)

# Normalize Cell Traces
cell_traces_normalized = data_cleaning_tools.normalize_cell_traces(cell_traces)

# # Create a (time, cell #) matrix of cell times to account for delays
cell_times_stack = [frame_times for cell in range(cell_traces_normalized.shape[0])]
cell_times = np.vstack(cell_times_stack)
cell_times = (cell_times.transpose() + cell_delays).transpose()

# Interpolate all cell traces to frame times
cell_traces_interp = np.array([np.interp(frame_times, cell_times[i], cell_traces_normalized[i])
                               for i in range(cell_traces_normalized.shape[0])])
h5f.create_dataset('cell_traces', data=cell_traces_interp)

filename = "../data/audio_timestamps_19120-2-4.h5"
downsample = 1000

with h5py.File(filename, 'r') as f:
    print('Available datasets:', list(f.keys()))
    audio_times = np.array(f['audio_times'])[::2 * downsample]

audio_file = "../data/19120_2_00004_audio.flac"
audio_event_frequencies = [8750, 12000, 13700]
neural_recording_frequency = 12000
peaks = data_cleaning_tools.extract_audio_features(audio_file, audio_event_frequencies, 100, 0.005, downsample=downsample)
sr = peaks['sr']

for frequency in audio_event_frequencies:
    audio_peak = peaks[frequency]
    audio_peak = np.interp(frame_times, audio_times, audio_peak)
    audio_peak = np.where(audio_peak > 0, 1, 0)
    h5f.create_dataset('audio_peak_' + str(frequency), data=audio_peak)
h5f.close()
