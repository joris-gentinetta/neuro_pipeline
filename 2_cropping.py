######################################
animal = '211'
alert_when_done = False
######################################

import numpy as np
from load_intan_rhd_format.load_intan_rhd_format import read_data
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import subprocess
import shlex
from natsort import natsorted
import pickle5 as pkl
from utils import alert
from scipy.signal import hilbert
from scipy.signal import butter, sosfiltfilt

butter_order = 3
theta_band = [4, 12]
spike_band = [300, 3000]
sampling_rate = 20000
frame_rate = 50

# folder preparation
animal_folder = r'E:/anxiety_ephys/' + animal + '/'
sub_folder = 'circus'
target_folder = animal_folder + sub_folder + '/'
experiment_names = natsorted(os.listdir(animal_folder))
if 'circus' in experiment_names:
    experiment_names.remove('circus')
data_folders = [animal_folder + experiment_name + '/' for experiment_name in experiment_names]
templates = r'E:/pipe/spykingcircus_sorter_2/templates/'
circus_entrypoint = target_folder + 'dat_files/' + experiment_names[0] + '_0.dat'
if os.listdir(target_folder + 'dat_files_mod'):
    raise Exception('Croppig already done, to redo run config again first.')
#os.mkdir(target_folder + 'movement_files')#todo remove

mouse_is_late = {'2021-02-19_mBWfus010_EZM_ephys': 70,
                 '2021-02-19_mBWfus009_EZM_ephys': 42,
                 '2021-02-26_mBWfus012_EZM_ephys': 35}
accomodation = 20

logbook = np.zeros(len(experiment_names))
for index, experiment_name in enumerate(experiment_names):

    eventfile = animal_folder + experiment_name + '/ephys_processed/' + experiment_name + '_events.pkl'
    #get events/movement
    with open(eventfile, 'rb') as f:
        events = pkl.load(f)
    movement = events['movement']
    xy = np.array([movement['calib_traj_x'], movement['calib_traj_y']], dtype=np.float32)

    movement_trigger_file = animal_folder + '/' + experiment_name + '/log.txt'
    with open(movement_trigger_file, 'r') as file:
        data = file.read().replace('\n', ' ')
    point = data.index('.')
    movement_trigger = int(float(data[point - 2:point + 3]) * frame_rate)

    if experiment_name in mouse_is_late:
        off = (mouse_is_late[experiment_name] + accomodation) * sampling_rate
    else:
        off = 0
    movement_trigger += off
    xy = xy[:,movement_trigger:]
    mPFC_concatenated = np.load(target_folder + 'mPFC_raw/' + experiment_name + '.npy')
    vHIP_concatenated = np.load(target_folder + 'vHIP_raw/' + experiment_name + '.npy')
    if os.path.exists(target_folder + 'cutter/' + experiment_name + '.npy'):
        mPFC_spike_range = np.load(target_folder + 'mPFC_spike_range/' + experiment_name)
        cutter = np.load(target_folder + 'cutter/' + experiment_name + '.npy')
        # first row start time of cuts, second row stop time, dtype uint32
        boolean_sampling_rate = np.ones(mPFC_concatenated.shape[1], dtype=np.bool)
        boolean_frame_rate = np.ones(xy.shape[1], dtype=np.bool)
#crop spike range as well
        for cut in range(cutter.shape[1]):
            boolean_sampling_rate[cutter[0, cut]*sampling_rate//frame_rate: cutter[1, cut]*sampling_rate//frame_rate] = 0
            boolean_frame_rate[cutter[0, cut]: cutter[1, cut]] = 0
        mPFC_concatenated = mPFC_concatenated[:, boolean_sampling_rate]
        vHIP_concatenated = vHIP_concatenated[:, boolean_sampling_rate]
        mPFC_spike_range = mPFC_spike_range[:, boolean_sampling_rate]
        xy = xy[:, boolean_frame_rate]
        np.save(target_folder + 'mPFC_spike_range/' + experiment_name, mPFC_spike_range)

    np.save(target_folder + 'movement_files/' + experiment_name, xy)

    sos_theta = butter(N=butter_order, Wn=theta_band, btype='bandpass', analog=False, output='sos', fs=sampling_rate)
    theta_filtered = sosfiltfilt(sos_theta, vHIP_concatenated, axis=1)
    hilbert_phase = np.angle(hilbert(theta_filtered, axis=1), deg=True)  # use np.unwrap()?
    data_for_spikesorting = np.transpose(mPFC_concatenated)
    # save files:
    data_for_spikesorting.tofile(target_folder + 'dat_files/' + experiment_names[index] + '_' + str(index) + '.dat')
    np.save(target_folder + 'vHIP_phase/' + experiment_name, hilbert_phase)
    logbook[index] = data_for_spikesorting.shape[0]
np.save(target_folder + 'utils/logbook', logbook)



# start clustering process #15min


cluster_command = 'spyking-circus ' + circus_entrypoint + ' -c 10'
os.system(cluster_command)

# args = shlex.split(cluster_command)
# cluster = subprocess.run(args, stdout=subprocess.PIPE,
#                          encoding='ascii')
# print(cluster.stdout)
#
# print('clustering return code: ', cluster.returncode)


converter_command = 'spyking-circus ' + circus_entrypoint + ' -m converting -c 10'
args = shlex.split(converter_command)

converter = subprocess.run(args, stdout=subprocess.PIPE,
                           input='a\n', encoding='ascii')

print(converter.returncode)
# print(converter.stdout)


# start viewer
viewer_command = '@echo off \ncall conda activate circus \ncircus-gui-python ' + circus_entrypoint
with open(target_folder + 'utils/start_viewer.bat', 'w') as f:
    f.write(viewer_command)
args = shlex.split(viewer_command)
if alert_when_done:
    alert()
viewer = subprocess.run(args, stdout=subprocess.PIPE,
                        encoding='ascii')

print(viewer.returncode)
# print(viewer.stdout)

print('Cropping for animal {} done!'.format(animal))
