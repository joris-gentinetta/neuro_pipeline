######################################
animal = '444'
alert_when_done = True
######################################

import os
import shlex
import subprocess
import time
import numpy as np
import pickle5 as pkl
from natsort import natsorted
from scipy.signal import butter, sosfiltfilt
from scipy.signal import hilbert
from utils import alert

butter_order = 3
theta_band = [4, 12]
spike_band = [300, 3000]
sampling_rate = 20000
frame_rate = 50  # frame rate of the video

start_time = time.time()
# folder preparation
animal_folder = r'E:/anxiety_ephys/' + animal + '/'
sub_folder = 'circus'
target_folder = animal_folder + sub_folder + '/'
experiment_names = natsorted(os.listdir(animal_folder))
if 'circus' in experiment_names:
    experiment_names.remove('circus')
circus_entrypoint = target_folder + 'dat_files/' + experiment_names[
    0] + '_0.dat'  # file to run spyking circus on (first of the data files)
##todo undo commenting:
# if os.listdir(target_folder + 'dat_files_mod'): #can only crop once
#     raise Exception('Croppig already done, to redo run config.py again first.')
mouse_is_late = {'2021-02-19_mBWfus010_EZM_ephys': 70,
                 # seconds delay from trigger signal to mouse in the maze #same as in 1_config.py
                 '2021-02-19_mBWfus009_EZM_ephys': 42,
                 '2021-02-26_mBWfus012_EZM_ephys': 35}
accomodation = 20  # time to let the late mice settle #same as in 1_config.py

logbook = np.zeros(len(experiment_names))  # will contain the length of the recordings for all sessions

for index, experiment_name in enumerate(experiment_names):
    eventfile = animal_folder + experiment_name + '/ephys_processed/' + experiment_name + '_events.pkl'
    ##get events/movement
    with open(eventfile, 'rb') as f:
        events = pkl.load(f)
    movement = events['movement']
    xy = np.array([movement['calib_traj_x'], movement['calib_traj_y']],
                  dtype=np.float32)  # first row: x coord, second row: y coord
    movement_trigger_file = animal_folder + '/' + experiment_name + '/log.txt'
    with open(movement_trigger_file, 'r') as file:
        data = file.read().replace('\n', ' ')
    point = data.index('.')
    movement_trigger = int(float(data[point - 2:point + 3]) * frame_rate)  # get the 50Hz movement trigger
    # introduce offset for the movement #is done in 1_config.py for ephys
    if experiment_name in mouse_is_late:
        off = (mouse_is_late[experiment_name] + accomodation) * frame_rate
    else:
        off = 0
    movement_trigger += off
    xy = xy[:, movement_trigger:]  # crop from movement trigger
    mPFC_concatenated = np.load(target_folder + 'mPFC_raw/' + experiment_name + '.npy')
    vHIP_concatenated = np.load(target_folder + 'vHIP_raw/' + experiment_name + '.npy')
    ##cut out the time segments specified in the cutter file
    if os.path.exists(target_folder + 'cutter/' + experiment_name + '.npy'):
        mPFC_spike_range = np.load(target_folder + 'mPFC_spike_range/' + experiment_name)
        cutter = np.load(
            target_folder + 'cutter/' + experiment_name + '.npy')  # first row start time of cuts, second row stop time, dtype uint32
        boolean_sampling_rate = np.ones(mPFC_concatenated.shape[1], dtype=np.bool)
        boolean_frame_rate = np.ones(xy.shape[1], dtype=np.bool)
        for cut in range(cutter.shape[1]):
            boolean_sampling_rate[cutter[0, cut] * sampling_rate // frame_rate: cutter[
                                                                                    1, cut] * sampling_rate // frame_rate] = 0  # mask the values to cut out 20000Hz
            boolean_frame_rate[cutter[0, cut]: cutter[1, cut]] = 0  # mask the values to cut out 20000Hz
        ##discard the masked values:
        mPFC_concatenated = mPFC_concatenated[:, boolean_sampling_rate]
        vHIP_concatenated = vHIP_concatenated[:, boolean_sampling_rate]
        mPFC_spike_range = mPFC_spike_range[:, boolean_sampling_rate]
        xy = xy[:, boolean_frame_rate]
        np.save(target_folder + 'mPFC_spike_range/' + experiment_name,
                mPFC_spike_range.astype(np.int16))  # used to calculate mean waveform in 4_sanity_check.py
        ####todo:
        transitions = {}
        for mode in events['transitions']:
            event_indices = events['transitions'][mode]
            event_boolean = np.zeros(mPFC_concatenated.shape[1], dtype=bool)
            event_boolean[event_indices] = 1
            event_boolean = event_boolean[boolean_frame_rate]
            transitions[mode] = list(np.nonzero(event_boolean))
    with open(target_folder + 'transition_files/' + experiment_name + '.pkl', 'wb') as f:
        pkl.dump(transitions, f)
    np.save(target_folder + 'movement_files/' + experiment_name, xy.astype(np.float32))
    ##theta bandpass filtering:
    sos_theta = butter(N=butter_order, Wn=theta_band, btype='bandpass', analog=False, output='sos', fs=sampling_rate)
    theta_filtered = sosfiltfilt(sos_theta, vHIP_concatenated, axis=1)
    hilbert_phase = np.angle(hilbert(theta_filtered, axis=1), deg=True)  # get theta phase
    data_for_spikesorting = np.transpose(mPFC_concatenated)
    # save files:
    data_for_spikesorting.tofile(
        target_folder + 'dat_files/' + experiment_names[index] + '_' + str(index) + '.dat')  # used by spykingcircus
    np.save(target_folder + 'vHIP_phase/' + experiment_name, hilbert_phase.astype(np.int16))  # save theta phase
    logbook[index] = data_for_spikesorting.shape[0]  # logbook[i] contains the length of session i, 20000Hz
np.save(target_folder + 'utils/logbook', logbook.astype(np.uint32))

before_clustering_time = time.time()
##start clustering process:
cluster_command = 'spyking-circus ' + circus_entrypoint + ' -c 10'
os.system(cluster_command)

##convert output of spykingcircus for the phy viewer:
converter_command = 'spyking-circus ' + circus_entrypoint + ' -m converting -c 10'
args = shlex.split(converter_command)
converter = subprocess.run(args, stdout=subprocess.PIPE,
                           input='a\n', encoding='ascii')
print('Converter return_code: {}'.format(converter.returncode))

end_time = time.time()
print('Cropping for animal {} done! \nTime needed: {} minutes. Time for SpykingCircus: {}'.format(animal, (
            end_time - start_time) / 60, (end_time - before_clustering_time) / 60))
if alert_when_done:
    alert()

##create a batchfile to start the phy viewer by clicking on the file
viewer_command = '@echo off \ncall conda activate circus \ncircus-gui-python ' + circus_entrypoint
with open(target_folder + 'utils/start_viewer.bat', 'w') as f:
    f.write(viewer_command)

##start the phy viewer:
viewer_command = 'circus-gui-python ' + circus_entrypoint
args = shlex.split(viewer_command)
viewer = subprocess.run(args, stdout=subprocess.PIPE,
                        encoding='ascii')
print('Viewer return_code: {}'.format(viewer.returncode))
