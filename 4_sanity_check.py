#%%

######################################
animal = '211'
alert_when_done = True
######################################

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from natsort import natsorted
from utils import alert
from time import sleep
from shutil import rmtree
from scipy.stats import pearsonr
from scipy.ndimage.filters import gaussian_filter1d



animal_folder = r'E:/anxiety_ephys/' + animal + '/'
experiment_names = natsorted(os.listdir(animal_folder))
if 'circus' in experiment_names:
    experiment_names.remove('circus')
sub_folder = 'circus'
target_folder = animal_folder + sub_folder + '/'
archive = pd.read_pickle(target_folder + 'archive.pkl')
channels = archive.loc[:, ('characteristics', 'data_row')].values
waveform_dict = {}
isi_dict = {}
cluster_names = np.load(target_folder + 'utils/cluster_names' + '.npy')
mPFC_pads = np.load(target_folder + 'utils/mPFC_pads.npy')
plot_folder = target_folder + 'plots/'
idx = pd.IndexSlice



if os.path.exists(plot_folder):
    rmtree(plot_folder, ignore_errors=True)
    sleep(5)
    os.mkdir(plot_folder)
else:
    os.mkdir(plot_folder)
sanity_plots = plot_folder + 'sanity_plots/'
os.mkdir(sanity_plots)
number_of_bins_isi = 200

isi_bin_size = 0.5  # ms
max_isi_time = number_of_bins_isi*isi_bin_size  # miliseconds
sampling_rate = 20000
for experiment_name in experiment_names:
    experiment_plots = plot_folder+ experiment_name
    os.mkdir(experiment_plots)
    spikes_20000 = np.load(target_folder + 'spikes_20000/' + experiment_name + '.npy')
    mPFC_spike_range = np.load(target_folder + 'mPFC_spike_range/' + experiment_name + '.npy').astype(np.int16)#todo
    number_of_units = spikes_20000.shape[0]
    number_of_channels = mPFC_spike_range.shape[0]

    mean_wave_forms = np.empty((number_of_units, number_of_channels, 60))
    for shift in range(-20, 40):
        shifted = np.zeros(spikes_20000.shape, dtype=bool)
        if shift == 0:
            shifted = spikes_20000
        elif shift < 0:
            shifted[:, :shift] = spikes_20000[:, -shift:]
        else:
            shifted[:, shift:] = spikes_20000[:, :-shift]
        for unit in range(number_of_units):
            mean_wave_forms[unit, :, shift+20] = np.mean(mPFC_spike_range[:, shifted[unit]], axis=1)
    waveform_dict[experiment_name] = mean_wave_forms


    isi = np.zeros((number_of_units, number_of_bins_isi))
    for unit in range(number_of_units):
        frames, = np.nonzero(spikes_20000[unit])
        for index in range(frames.size - 1):
            frames_diff = int((frames[index + 1] - frames[index]) * 1000 / isi_bin_size // sampling_rate)
            if frames_diff < number_of_bins_isi:
                isi[unit, frames_diff] += 1
    isi_dict[experiment_name]= isi
#%%
sorted_units = zip(channels,np.arange(cluster_names.size), cluster_names)
sorted_units = sorted(sorted_units)
#%%

for unit in range(number_of_units):
    unit_id = sorted_units[unit][2]
    unit_index = sorted_units[unit][1]
    channel = sorted_units[unit][0]
    fig = plt.figure(figsize=(2, 40))
    for index, key in enumerate(natsorted(waveform_dict.keys())):

        toplot = waveform_dict[key][unit_index,:,:] + np.linspace(0, 100*number_of_channels, number_of_channels)[:, None]

        plt.plot((toplot+index*15).T, 'b', linewidth=0.5)
    axes = plt.axes()
    axes.set_yticks(np.linspace(0, 100 * number_of_channels, number_of_channels))
    axes.set_yticklabels(mPFC_pads)
    axes.set_xticks(np.linspace(0, 60, 2))
    axes.set_xticklabels([-1, 2])
    axes.set_title(sorted_units[unit][2], loc='right')
    plt.savefig(sanity_plots + 'waveform_unit_' + str(unit_id))
    #plt.show()
    plt.close(fig)

    fig, axs = plt.subplots(1, len(isi_dict.keys()), sharex=True, sharey=True)
    fig.set_figheight(5)
    fig.set_figwidth(25)
    for index, key in enumerate(natsorted(isi_dict.keys())):
        toplot = isi_dict[key][unit_index, :]/np.sum(isi_dict[key][unit_index, :])
        axs[index].bar(np.arange(number_of_bins_isi), toplot)
      #  axs[0, index].set_yticks(np.linspace(0, 100 * number_of_channels, number_of_channels))
        axs[ index].set_xticks(np.linspace(0, number_of_bins_isi, 10))
        axs[index].set_xticklabels((np.linspace(0, number_of_bins_isi, 10)//(1/isi_bin_size)).astype(int))
        axs[ index].set_title(key, loc='right')


    plt.savefig(sanity_plots + 'isi_unit_' + str(unit_id))
    #plt.show()
    plt.close(fig)
#%%
sigma=3
pearson_coefficients = pd.DataFrame(index= sorted_units[:][2], columns=natsorted(waveform_dict.keys()))
p_values = pd.DataFrame(index= sorted_units[:][2], columns=natsorted(waveform_dict.keys()))

fig, axs = plt.subplots(number_of_units, len(isi_dict.keys()), sharex=True)
fig.set_figheight(25)
fig.set_figwidth(25)
for unit in range(number_of_units):
    unit_id = sorted_units[unit][2]
    unit_index = sorted_units[unit][1]
    channel = sorted_units[unit][0]
    sum = np.zeros(number_of_bins_isi)
    for index, key in enumerate(natsorted(isi_dict.keys())):
        toplot = np.where(np.sum(isi_dict[key][unit_index, :]) != 0, isi_dict[key][unit_index, :]/np.sum(isi_dict[key][unit_index, :]),0)
        sum+=toplot
        axs[unit, index].bar(np.arange(number_of_bins_isi), toplot)
      #  axs[0, index].set_yticks(np.linspace(0, 100 * number_of_channels, number_of_channels))
        axs[unit, index].set_xticks(np.linspace(0, number_of_bins_isi, 10))
        axs[unit, index].set_xticklabels((np.linspace(0, number_of_bins_isi, 10)//(1/isi_bin_size)).astype(int))
        axs[unit, index].set_title(unit_id, loc='right')
    mean = sum/len(isi_dict.keys())
    archive.loc[unit_id, idx['isi',:]] = mean

    for index, key in enumerate(natsorted(waveform_dict.keys())):
        session_isi = isi_dict[key][unit_index, :]
        pearson_coefficients.loc[unit_id, key], p_values.loc[unit_id, key] =\
            pearsonr(gaussian_filter1d(mean, sigma=sigma), gaussian_filter1d(session_isi,sigma=sigma))
pearson_coefficients.to_pickle(target_folder + 'pearson_coefficients_isi.pkl')
p_values.to_pickle(target_folder + 'p_values_isi.pkl')

plt.savefig(sanity_plots + 'isi_unit_' + str(unit_id))
plt.show()
plt.close(fig)

#%%
pearson_coefficients = pd.DataFrame(index= sorted_units[:][2], columns=natsorted(waveform_dict.keys()))
p_values = pd.DataFrame(index= sorted_units[:][2], columns=natsorted(waveform_dict.keys()))
fig, axs = plt.subplots(1, number_of_units, sharex=True, sharey=True)
fig.set_figheight(25)
fig.set_figwidth(25)
for unit in range(number_of_units):
    unit_id = sorted_units[unit][2]
    unit_index = sorted_units[unit][1]
    channel = sorted_units[unit][0]
    sum = np.zeros(60)
    for index, key in enumerate(natsorted(waveform_dict.keys())):
        if not np.isnan(channel):
            sum += waveform_dict[key][unit_index,int(channel),:]
        else:
            sum = 1
        toplot = waveform_dict[key][unit_index,:,:] + np.linspace(0, 100*number_of_channels, number_of_channels)[:, None]
        axs[unit].plot((toplot + index * 15).T, 'b', linewidth=0.5)
    mean = sum / len(waveform_dict.keys())
    axs[unit].set_yticks(np.linspace(0, 100*number_of_channels, number_of_channels))
    axs[unit].set_yticklabels(mPFC_pads)
    axs[unit].set_xticks(np.linspace(0,60, 2))
    axs[unit].set_xticklabels([-1,2])
    axs[unit].set_title(unit_id, loc='right')
    archive.loc[unit_id, idx['mean_waveform',:]] = mean
    for index, key in enumerate(natsorted(waveform_dict.keys())):
        main_waveform = waveform_dict[key][unit_index, int[channel], :]
        pearson_coefficients.loc[unit_id, key], p_values.loc[unit_id, key] = pearsonr(mean, main_waveform)
pearson_coefficients.to_pickle(target_folder + 'pearson_coefficients_waveform.pkl')
p_values.to_pickle(target_folder + 'p_values_waveform.pkl')

plt.savefig(sanity_plots + 'waveform_all_units')
plt.show()
plt.close(fig)
archive.to_pickle(target_folder + 'archive.pkl')

if alert_when_done:
    alert()

