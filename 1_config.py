######################################
animal = '111'
max_impedance = 2500000
select_channels = True
delete_circus_folder = False
alert_when_done = True
######################################

import numpy as np
from load_intan_rhd_format.load_intan_rhd_format import read_data
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from natsort import natsorted
import utils

from scipy.signal import butter, sosfiltfilt

first_pad_vHIP = 33
butter_order = 3
theta_band = [4, 12]
spike_band = [300, 3000]
sampling_rate = 20000

start_time = time.time()
# folder preparation
animal_folder = r'E:/anxiety_ephys/' + animal + '/'
sub_folder = 'circus'
target_folder = animal_folder + sub_folder + '/'

if (os.path.exists(target_folder)):
    # print('overwrite ', target_folder, '? [y/n]')
    # if input() == 'y':
    if delete_circus_folder:
        shutil.rmtree(target_folder)
        time.sleep(5)
        utils.create_directories(target_folder)
else:
    utils.create_directories(target_folder)


experiment_names = natsorted(os.listdir(animal_folder))
if 'circus' in experiment_names:
    experiment_names.remove('circus')
data_folders = [animal_folder + experiment_name + '/' for experiment_name in experiment_names]
templates = r'E:/pipe/spykingcircus_sorter_2/templates/'
param_file = target_folder + 'dat_files/' + experiment_names[0] + '_0.params'
circus_entrypoint = target_folder + 'dat_files/' + experiment_names[0] + '_0.dat'

# get common channels, build physio_triggers
physio_triggers = {}
for index, data_folder in enumerate(data_folders):
    rhd_folder = data_folder + 'ephys/'
    rhd_files = natsorted(os.listdir(rhd_folder))
    amp_channels = read_data(rhd_folder + rhd_files[0])['amplifier_channels']
    channel_set = {int(channel['native_channel_name'][2:]) for channel in amp_channels}

    if index == 0:
        valid_channels = channel_set #valid_channels contains all channels that are present in all sessions
    else:
        valid_channels = valid_channels.intersection(channel_set)

    # get physio_trigger
    r = read_data(rhd_folder + rhd_files[0])
    trigger = np.array(r['board_dig_in_data'][0])
    i, = np.where(trigger == 1)
    physio_triggers[experiment_names[index]] = i[-1]

# pta[x] denotes the channel of pad x+1, pta.index(channel) denotes pad-1 of channel
pta = [28, 29, 27, 12, 20, 21, 11, 7, 1, 8, 10, 15, 18, 23, 26, 31, 2, 6, 9, 14, 17, 22, 25, 30, 19, 24, 0, 13, 16, 5,
       4, 3, 32, 45, 43, 56, 35, 40, 60, 52, 33, 38, 41, 46, 49, 54, 57, 62, 34, 39, 42, 47, 50, 55, 58, 63, 44, 36, 59,
       61, 37, 48, 53, 51]
ordered_channels = [] #ordered_channels contains all valid channels ordered by their pad number
for i in range(64):
    for channel in valid_channels:
        if pta.index(channel) + 1 == i:
            ordered_channels.append(channel)
mPFC_channels = [channel for channel in ordered_channels if pta.index(channel) + 1 < first_pad_vHIP]
vHIP_channels = [channel for channel in ordered_channels if pta.index(channel) + 1 >= first_pad_vHIP]
#get mPFC_impedance and vHIP_impedance from last session:
data_folder = data_folders[-1] #folder of the last session
rhd_folder = data_folder + 'ephys/' #folder with the ephys rhd files
rhd_files = natsorted(os.listdir(rhd_folder)) #all rhd files from the last session
amp_channels = read_data(rhd_folder + rhd_files[0])['amplifier_channels']  # all channels
channel_list = [int(channel['native_channel_name'][2:]) for channel in amp_channels] #all channel numbers
mPFC_indices = [channel_list.index(valid_channel) for valid_channel in mPFC_channels] #the indices of the valid channels of mPFC in this session
vHIP_indices = [channel_list.index(valid_channel) for valid_channel in vHIP_channels]#the indices of the valid channels of vHIP in this session
amp_channels = np.array(amp_channels)#todo why does this work
mPFC_impedance = [channel['electrode_impedance_magnitude'] for channel in amp_channels[mPFC_indices]] #mPFC impedance of the valid channels in the right order
vHIP_impedance = [channel['electrode_impedance_magnitude'] for channel in amp_channels[vHIP_indices]] #vHIP impedance of the valid channels in the right order
if select_channels:
    rhd_file = rhd_files[-2]
    rhd_data = np.array(read_data(rhd_folder + rhd_file)['amplifier_data'], dtype=np.int16) #raw ephys data of the second to last minute in the recording
    mPFC_data = rhd_data[mPFC_indices] #ephys data in the right order, containing only valid channels
    vHIP_data = rhd_data[vHIP_indices] #ephys data in the right order, containing only valid channels
    sos_spike = butter(N=butter_order, Wn=spike_band, btype='bandpass', analog=False, output='sos', fs=sampling_rate) #prepare the params for the filtering
    spike_filtered = sosfiltfilt(sos_spike, mPFC_data - np.median(mPFC_data, axis=0)[None, :], axis=1) # remove median an filter to spike_band
    toplot = spike_filtered[:, spike_filtered.shape[1] // 10 * 8:spike_filtered.shape[1] // 10 * 9] # take subset of the time to display
    fig = plt.figure(figsize=(20, 30))
    gs = fig.add_gridspec(toplot.shape[0], hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for row, ax in tqdm(enumerate(axs)): #plot the spike_band filtered ephys data on all valid channel in the right order
        label = 'channel: ' + str(mPFC_channels[row]) + ', ' + str(
            mPFC_impedance[row] * 100 // max_impedance) + '% max impedance' #label with the channel number and % of max impedance
        ax.plot(toplot[toplot.shape[0]-row-1], label=label, linewidth=1)
        ax.legend(loc='upper right')
    plt.ylim(-50, 50)
    fig.suptitle('mPFC channels', size=100)
    plt.show()
    plt.close(fig)

    #do the same for the channels in vHIP:
    spike_filtered = sosfiltfilt(sos_spike, vHIP_data - np.median(vHIP_data, axis=0)[None, :], axis=1)
    spike_filtered = spike_filtered - np.median(spike_filtered, axis=0)[None, :]

    toplot = spike_filtered[:, spike_filtered.shape[1] // 10 * 8:spike_filtered.shape[1] // 10 * 9]
    fig = plt.figure(figsize=(20, 30))
    gs = fig.add_gridspec(toplot.shape[0], hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for row, ax in tqdm(enumerate(axs)):
        label = 'channel: ' + str(vHIP_channels[row]) + ', ' + str(
            vHIP_impedance[row] * 100 // max_impedance) + '% max impedance'
        ax.plot(toplot[toplot.shape[0]-row-1], label=label, linewidth=1)
        ax.legend(loc='upper right')
    plt.ylim(-50, 50)
    fig.suptitle('vHIP channels', size=100)
    plt.show()
    plt.close(fig)
    discarded_channels = np.array(input('discarded channels seperated by space: ').split(), dtype=np.int8)
    for discarded_channel in discarded_channels:
        if discarded_channel in mPFC_channels:
            mPFC_channels.remove(discarded_channel)
        elif discarded_channel in vHIP_channels:
            vHIP_channels.remove(discarded_channel)
  #  np.save(target_folder + 'discarded_channels', discarded_channels.astype(np.uint16))

mPFC_channels = list(np.array(mPFC_channels)[np.array(mPFC_impedance) < max_impedance]) #remove channels above max impedance
vHIP_channels = list(np.array(vHIP_channels)[np.array(vHIP_impedance) < max_impedance])
# pads[x] denotes the pad corresponding to data['amplifier_data'][x]
mPFC_pads = [pta.index(channel) + 1 for channel in mPFC_channels]
vHIP_pads = [pta.index(channel) + 1 for channel in vHIP_channels]
np.save(target_folder + 'utils/vHIP_pads', np.array(vHIP_pads, dtype=np.uint16))#save the pads corresponding to the channels/data_rows
np.save(target_folder + 'utils/mPFC_pads', np.array(mPFC_pads, dtype=np.uint16))
# probe file 'graph'
graph = []  #needs to be empty for legacy reasons

# probe file 'geometry'
geometry = {} #the geometry of the probe, mostly used for visualisation
for i in range(len(mPFC_pads)):
    if mPFC_pads[i] < 33:
        geometry[i] = (0, (mPFC_pads[i] - 32) * 70)
    else:
        geometry[i] = (3000, (mPFC_pads[i] - 64) * 70)
# param file 'nchannels'
total_nb_channels = len(mPFC_channels)
channels = list(np.arange(0, total_nb_channels))

mouse_is_late = {'2021-02-19_mBWfus010_EZM_ephys': 70, #keys: all the mice that are not present in the maze at the video trigger
                 '2021-02-19_mBWfus009_EZM_ephys': 42, #values: seconds after the trigger until they are present
                 '2021-02-26_mBWfus012_EZM_ephys': 35}

accomodation = 20 #time after mouse is introduced into the maze to cut off in seconds (only used if mouse_is_late)
#prepare the data files for cropping.py:
for index, experiment_name in tqdm(enumerate(experiment_names)):
    physio_trigger = physio_triggers[experiment_name] #20000Hz trigger
    #calculate offset for late mice:
    if experiment_name in mouse_is_late:
        off = (mouse_is_late[experiment_name] + accomodation)*sampling_rate
    else:
        off = 0
    physio_trigger += off

    rhd_folder = animal_folder + experiment_name + '/ephys/'
    rhd_files = natsorted(os.listdir(rhd_folder))
    amp_channels = read_data(rhd_folder + rhd_files[0])['amplifier_channels']  # all amplifier channels
    channel_list = [int(channel['native_channel_name'][2:]) for channel in amp_channels] # all amplifier channel names
    mPFC_indices = [channel_list.index(valid_channel) for valid_channel in mPFC_channels] #the indices of the valid channels
    vHIP_indices = [channel_list.index(valid_channel) for valid_channel in vHIP_channels]
    total_size = 0
    for i, rhdfile in enumerate(rhd_files):
        rhd_data = np.array(read_data(rhd_folder + rhdfile)['amplifier_data'], dtype=np.int16)#rows are unordered channls
        mPFC_data = rhd_data[mPFC_indices] #data ordered by pad, only valid channels
        vHIP_data = rhd_data[vHIP_indices]
        if i == 0:
            #make empty np array to store the data of all rhd files of the session
            mPFC_concatenated = np.zeros((mPFC_data.shape[0], mPFC_data.shape[1] * len(rhd_files)), dtype=np.int16)
            vHIP_concatenated = np.zeros((vHIP_data.shape[0], vHIP_data.shape[1] * len(rhd_files)), dtype=np.int16)
        mPFC_concatenated[:, total_size:total_size + mPFC_data.shape[1]] = mPFC_data #assign data from the current rhd
        #file to its timeslot
        vHIP_concatenated[:, total_size:total_size + vHIP_data.shape[1]] = vHIP_data
        total_size += vHIP_data.shape[1]
    mPFC_concatenated = mPFC_concatenated[:, physio_trigger:total_size] #crop the data to the relevant time-window
    vHIP_concatenated = vHIP_concatenated[:, physio_trigger:total_size]
    np.save(target_folder + 'mPFC_raw/' + experiment_names[index], mPFC_concatenated.astype(np.int16))
    np.save(target_folder + 'vHIP_raw/' + experiment_names[index], vHIP_concatenated.astype(np.int16))
    #filter mPFC data to spike-range
    sos_spike = butter(N=butter_order, Wn=spike_band, btype='bandpass', analog=False, output='sos', fs=sampling_rate)
    spike_filtered = sosfiltfilt(sos_spike, mPFC_concatenated - np.median(mPFC_concatenated, axis=0)[None, :], axis=1)
    np.save(target_folder + 'mPFC_spike_range/' + experiment_names[index], spike_filtered.astype(np.int16))
    #use this data to make cutout timestamps

radius = 100 #radius in muekrometers to consider for clustering
#prepare data that will be on top of the templat in the probe file
cap1 = ('total_nb_channels = ' + str(total_nb_channels)
        + '\nradius = ' + str(radius)
        + '\ngraph = ' + str(graph)
        + '\nchannels = ' + str(channels)
        + '\ngeometry = ' + str(geometry))

# create probe.prb file
with open(target_folder + 'utils/probe.prb', 'w') as f:
    f.write(cap1)
with open(templates + 'probespecs.prb', 'r') as f:
    t = f.read()
with open(target_folder + 'utils/probe.prb', 'a') as f:
    f.write(t)

#prepare data that will be on top of the templat in the parameters file
cap2 = ('[data]'
        + '\nnb_channels = ' + str(total_nb_channels)
        + '\nmapping = ' + target_folder + 'utils/probe.prb'
        + '\noutput_dir = ' + target_folder + 'dat_files_mod')

# create parameters.params file
with open(param_file, 'w') as f:
    f.write(cap2)
with open(templates + 'parameters.params', 'r') as f:
    t = f.read()
with open(param_file, 'a') as f:
    f.write(t)

end_time = time.time()
print('Config for animal {} done! \nTime needed: {} minutes.'.format(animal, (end_time-start_time)/60))

if alert_when_done:
    utils.alert()
