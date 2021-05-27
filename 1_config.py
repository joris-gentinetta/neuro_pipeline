## only cell to adapt
animal = '012'
max_impedance = 2000000
max_channel = 33

import numpy as np
from load_intan_rhd_format.load_intan_rhd_format import read_data
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import subprocess
import shlex

from scipy.signal import hilbert
from scipy.signal import butter, sosfilt

butter_order = 3
theta_band = [4, 12]
spike_band = [300, 3000]
sampling_rate = 20000

# folder preparation
animal_folder = r'E:/anxiety_ephys/' + animal + '/'
sub_folder = 'circus'
target_folder = animal_folder + sub_folder + '/'

if (os.path.exists(target_folder)):
    print('overwrite ', target_folder, '? [y/n]')
    if input() == 'y':
        shutil.rmtree(target_folder)
        time.sleep(5)
        os.mkdir(target_folder)
        os.mkdir(target_folder + 'dat_files')
        os.mkdir(target_folder + 'phase_files')
        os.mkdir(target_folder + 'numpy_files')
        os.mkdir(target_folder + 'dat_files_mod')
else:
    os.mkdir(target_folder)
    os.mkdir(target_folder + 'dat_files')
    os.mkdir(target_folder + 'phase_files')
    os.mkdir(target_folder + 'numpy_files')
    os.mkdir(target_folder + 'dat_files_mod')
    # else:
    #     raise KeyboardInterrupt('Config step was already done.')

experiment_names = os.listdir(animal_folder)
if 'circus' in experiment_names:
    experiment_names.remove('circus')
data_folders = [animal_folder + experiment_name + '/' for experiment_name in experiment_names]
# rhd_folder = data_folder + 'ephys/'
templates = r'E:/pipe/spykingcircus_sorter_2/templates/'
# rhd_files = os.listdir(rhd_folder)
param_file = target_folder + 'dat_files/' + experiment_names[0] + '_0.params'
circus_entrypoint = target_folder + 'dat_files/' + experiment_names[0] + '_0.dat'



# get common channels, create trigger files
for index, data_folder in enumerate(data_folders):
    rhd_folder = data_folder + 'ephys/'
    rhd_files = os.listdir(rhd_folder)
    amp_channels = read_data(rhd_folder + rhd_files[0])['amplifier_channels']
    channel_set = {int(channel['native_channel_name'][2:]) for channel in amp_channels}
    if index == 0:
        valid_channels = channel_set
    else:
        valid_channels = valid_channels.intersection(channel_set)

    # create trigger file
    r = read_data(rhd_folder + rhd_files[0])
    trigger = np.array(r['board_dig_in_data'][0])
    i, = np.where(trigger == 1)
    np.save(target_folder + experiment_names[index] + '_trigger', i[-1])
# pta[x] denotes the channel of pad x+1
pta = [28, 29, 27, 12, 20, 21, 11, 7, 1, 8, 10, 15, 18, 23, 26, 31, 2, 6, 9, 14, 17, 22, 25, 30, 19, 24, 0, 13, 16, 5,
       4, 3, 32, 45, 43, 56, 35, 40, 60, 52, 33, 38, 41, 46, 49, 54, 57, 62, 34, 39, 42, 47, 50, 55, 58, 63, 44, 36, 59,
       61, 37, 48, 53, 51]
ordered_channels = []
for i in range(64):
    for channel in valid_channels:
        if pta.index(channel) + 1 == i:
            ordered_channels.append(channel)
mPFC_channels = [channel for channel in ordered_channels if pta.index(channel) + 1 < max_channel]
vHIP_channels = [channel for channel in ordered_channels if pta.index(channel) + 1 >= max_channel]

# pads[x] denotes the pad corresponding to data['amplifier_data'][x]
mPFC_pads = [pta[channel] + 1 for channel in mPFC_channels]
vHIP_pads = [pta[channel] + 1 for channel in vHIP_channels]
np.save(target_folder + 'phase_files/' + 'vHIP_pads', vHIP_pads)
np.save(target_folder + 'numpy_files/' + 'mPFC_pads', mPFC_pads)
# probe file 'graph'
graph = []

# probe file 'geometry'
geometry = {}
for i in range(len(mPFC_pads)):
    if mPFC_pads[i] < 33:
        geometry[i] = (0, (mPFC_pads[i] - 32) * 70)
    else:
        geometry[i] = (3000, (mPFC_pads[i] - 64) * 70)
# param file 'nchannels'
total_nb_channels = len(mPFC_channels)
channels = list(np.arange(0, total_nb_channels))

#create .dat files
#15min
start = time.time()
logbook = np.zeros(len(experiment_names))
for index, data_folder in tqdm(enumerate(data_folders)):
    rhd_folder = data_folder + 'ephys/'
    rhd_files = os.listdir(rhd_folder)
    amp_channels = read_data(rhd_folder + rhd_files[0])['amplifier_channels']  # time: 02:28min for 25 files
    channel_list = [int(channel['native_channel_name'][2:]) for channel in amp_channels]
    mPFC_indices = [channel_list.index(valid_channel) for valid_channel in mPFC_channels]
    vHIP_indices = [channel_list.index(valid_channel) for valid_channel in vHIP_channels]
    total_size = 0
    for i, rhdfile in enumerate(rhd_files):
        rhd_data = np.array(read_data(rhd_folder + rhdfile)['amplifier_data'], dtype=np.int16)
        mPFC_data = rhd_data[mPFC_indices]
        vHIP_data = rhd_data[vHIP_indices]
        if i == 0:
            mPFC_concatenated = np.zeros((mPFC_data.shape[0], mPFC_data.shape[1] * len(rhd_files)), dtype=np.int16)
            vHIP_concatenated = np.zeros((vHIP_data.shape[0], vHIP_data.shape[1] * len(rhd_files)), dtype=np.int16)
        mPFC_concatenated[:, total_size:total_size + mPFC_data.shape[1]] = mPFC_data
        vHIP_concatenated[:, total_size:total_size + vHIP_data.shape[1]] = vHIP_data
        total_size += vHIP_data.shape[1]
    mPFC_concatenated = mPFC_concatenated[:,:total_size]
    vHIP_concatenated = vHIP_concatenated[:,:total_size]
    # raw = tosave[:, :total_size] - np.median(tosave[:, :total_size], axis = 0)[None, :]
    sos_spike = butter(N=butter_order, Wn=spike_band, btype='bandpass', analog=False, output='sos', fs=sampling_rate)
    sos_theta = butter(N=butter_order, Wn=theta_band, btype='bandpass', analog=False, output='sos', fs=sampling_rate)
    theta_filtered = sosfilt(sos_theta, vHIP_concatenated, axis=1)
    spike_filtered = sosfilt(sos_spike, mPFC_concatenated, axis=1)
    hilbert_phase = np.angle(hilbert(theta_filtered, axis=1), deg=True)  # use np.unwrap()?
    data_for_spikesorting = np.transpose(mPFC_concatenated)
    data_for_spikesorting.tofile(target_folder + 'dat_files/' + experiment_names[index] + '_' + str(index) + '.dat')
    np.save(target_folder + 'phase_files/' + experiment_names[index], hilbert_phase)
    np.save(target_folder + 'numpy_files/' + experiment_names[index], spike_filtered)

    logbook[index] = total_size
np.save(target_folder + 'logbook', logbook)
end = time.time()

# # prepare data for plotting
# rhd_folder = data_folders[0] + 'ephys/'
# rhd_files = os.listdir(rhd_folder)
# x = np.array(read_data(rhd_folder + rhd_files[0])['amplifier_data'], dtype=np.int16)[mPFC_indices]
# mPFC = np.zeros(x.shape[1], dtype=np.int16)
# n_mPFC = 0
# vH = np.zeros(x.shape[1], dtype=np.int16)
# n_vH = 0
# for i in range(x.shape[0]):
#     if mPFC_pads[i] < 33:
#         mPFC += x[i]
#         n_mPFC += 1
#     else:
#         vH += x[i]
#         n_vH += 1
# for i in range(x.shape[0]):
#     if mPFC_pads[i] < 33:
#         x[i] = (x[i] - mPFC / n_mPFC).astype(np.int16)
#
#     else:
#         x[i] = (x[i] - vH / n_vH).astype(np.int16)
#
# # plot channels
# lw = 0.4
# d = x[:, x.shape[1] // 10 * 8:x.shape[1] // 10 * 9]
# zero = np.zeros(d.shape[1])
# frontier = [1 if i % 2 == 0 else 0 for i in range(d.shape[1])]
# fig = plt.figure(figsize=(20, 30))
# gs = fig.add_gridspec(64, hspace=0)
# axs = gs.subplots(sharex=True, sharey=True)
# for i, ax in tqdm(enumerate(axs)):
#     for ind in range(64):
#         if ind >= len(mPFC_pads):
#             if i == 31:
#                 ax.plot(zero, label='0, frontier', linewidth=lw)
#             else:
#                 ax.plot(zero, label='0', linewidth=lw)
#         elif mPFC_pads[ind] == i + 1:
#             if i == 31:
#                 if amp_channels[ind]['electrode_impedance_magnitude'] > max_impedance:
#                     ax.plot(d[ind], label=str(ind) + ', impedance' + ', frontier', linewidth=lw)
#                 else:
#                     ax.plot(d[ind], label=str(ind) + ', frontier', linewidth=lw)
#             else:
#                 if amp_channels[ind]['electrode_impedance_magnitude'] > max_impedance:
#                     ax.plot(d[ind], label=str(ind) + ', impedance', linewidth=lw)
#                 else:
#                     ax.plot(d[ind], label=str(ind), linewidth=lw)
#     ax.legend(loc='upper right')
# plt.ylim(-150, 150)
# plt.show()

# prepare probe.prb file
radius = 100
cap1 = ('total_nb_channels = ' + str(total_nb_channels)
        + '\nradius = ' + str(radius)
        + '\ngraph = ' + str(graph)
        + '\nchannels = ' + str(channels)
        + '\ngeometry = ' + str(geometry))

# create probe.prb file
with open(target_folder + 'probe.prb', 'w') as f:
    f.write(cap1)
with open(templates + 'probespecs.prb', 'r') as f:
    t = f.read()
with open(target_folder + 'probe.prb', 'a') as f:
    f.write(t)

# prepare parameters.params file
cap2 = ('[data]'
        + '\nnb_channels = ' + str(total_nb_channels)
        + '\nmapping = ' + target_folder + 'probe.prb'
        + '\noutput_dir = ' + target_folder + 'dat_files_mod')

# create parameters.params file

with open(param_file, 'w') as f:
    f.write(cap2)
with open(templates + 'parameters.params', 'r') as f:
    t = f.read()
with open(param_file, 'a') as f:
    f.write(t)

# start clustering process #15min
os.system('spyking-circus ' + circus_entrypoint + ' -c 10')

cluster_command = 'spyking-circus ' + circus_entrypoint + ' -c 10'
args = shlex.split(cluster_command)

cluster = subprocess.run(args, stdout=subprocess.PIPE,
                         encoding='ascii')

print(cluster.returncode)
# print(cluster.stdout)


converter_command = 'spyking-circus ' + circus_entrypoint + ' -m converting -c 10'
args = shlex.split(converter_command)

converter = subprocess.run(args, stdout=subprocess.PIPE,
                           input='a\n', encoding='ascii')

print(converter.returncode)
# print(converter.stdout)


# start viewer
viewer_command = 'circus-gui-python ' + circus_entrypoint
with open(target_folder + 'start_viewer.txt', 'w') as f:
    f.write(viewer_command)
args = shlex.split(viewer_command)

viewer = subprocess.run(args, stdout=subprocess.PIPE,
                        encoding='ascii')

print(viewer.returncode)
# print(viewer.stdout)


print('config  --done')
