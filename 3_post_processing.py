######################################
animal = '999'
alert_when_done = True
######################################

import os
import time

import numpy as np
import pandas as pd
from natsort import natsorted

import utils

start_time = time.time()
##prepare folder/file names:
animal_folder = r'E:/anxiety_ephys/' + animal + '/'
experiment_names = natsorted(os.listdir(animal_folder))
if 'circus' in experiment_names:
    experiment_names.remove('circus')
experiment_name = experiment_names[0]
sub_folder = 'circus'
target_folder = animal_folder + sub_folder + '/'
folder = target_folder + 'dat_files_mod/' + experiment_name + '_0.GUI/'
stimes = 'spike_times.npy'
sclusters = 'spike_clusters.npy'
infofile = 'cluster_info.tsv'

##params:
frame_rate = 50
sampling_rate = 20000
number_of_bins_transitions = 20  # in 5 second window around transitions #has to be the same as in 5_day_analysis.py #even number
number_of_bins_phase = 20  # in 360 degrees #has to be the same as in 5_day_analysis.py #even number
number_of_bins_isi = 200  # ISI binsize is 0.5 ms #has to be the same as in 4_sanity_check.py #even number

info = pd.read_csv(folder + infofile, sep='\t', index_col=0)  # file with all units produced by spyking circus
idx = np.array(info.index)  # the IDs of the units
group = np.array(info.group)  # the group of the units ('good', 'mua', 'noise') as labeled in phy
##indexer[i] contains the number of the unit when counting the good units with unit_ID i:
indexer = np.zeros(idx.max() + 1, dtype=np.int8)
n = 1
for i in range(idx.size):
    if group[i] == 'good':
        indexer[idx[i]] = n
        n += 1


## assign the firing events to individual recording sessions (arena, EZM, OFT):
logbook_1 = np.load(target_folder + 'utils/logbook.npy')  # contains the lengths of the sessions, 20000Hz
logbook_2 = np.zeros(len(experiment_names) + 1)
logbook_2[1:] = logbook_1[
                :len(experiment_names)]  # shift to the right to make space for the 0 index of the first session

logbook_3 = np.zeros(len(experiment_names) + 1, dtype=np.int64)
for i in range(logbook_3.size):
    logbook_3[i] = np.sum(logbook_2[0:i + 1])
original_logbook_3 = logbook_3  # contains the idices of the start times of the sessions, 20000Hz
logbook_3 = logbook_3 * frame_rate // sampling_rate  # contains the idices of the start times of the sessions, 50 Hz

original_spiketimes = np.load(folder + stimes)  # array with the spiketimes, 20000Hz
spiketimes = original_spiketimes * frame_rate // sampling_rate  # array with the spiketimes, 50Hz
clusters = np.load(folder + sclusters)  # array with the unit_IDs associated with the spiketimes
data = np.zeros((int(indexer.max() + 1), logbook_3[-1]), dtype=np.uint8)  # rows: units, columns: 50Hz frames
original_data = np.zeros((int(indexer.max() + 1), original_logbook_3[-1]),
                         dtype=bool)  # rows: units, columns: 20000Hz frames

##for every unit (row) insert the nummber of spikes that happend in that frame (column) #data[0] contains the firing of all units that were not labeled 'good'
for index, timepoint in enumerate(spiketimes):
    data[indexer[clusters[index]], timepoint] += 1

for index, timepoint in enumerate(original_spiketimes):
    original_data[indexer[clusters[index]], timepoint] += 1

##cluster_names[i] contains the unit_id associated with row i in data and original_data
cluster_names = np.zeros(data.shape[0], dtype=np.uint8)
for i in range(1, cluster_names.size):
    cluster_names[i] = np.where(indexer == i)[0]
cluster_names[0] = 255  # the combined noise/mua gets unit_ID 255

np.save(target_folder + 'utils/cluster_names', cluster_names.astype(np.uint16))

##split the data into the individual sessions:
for i in range(logbook_3.size - 1):
    np.save(target_folder + 'spikes_50/' + experiment_names[i], data[:, logbook_3[i]:logbook_3[i + 1]].astype(np.uint8))
    np.save(target_folder + 'spikes_20000/' + experiment_names[i],
            original_data[:, original_logbook_3[i]:original_logbook_3[i + 1]].astype(bool))

vHIP_pads = np.load(target_folder + 'utils/vHIP_pads.npy')
archive = utils.create_archive(vHIP_pads, cluster_names, number_of_bins_transitions, number_of_bins_phase,
                               number_of_bins_isi)  # create empty archive
##fill the archive with information provided in the info file created by phy:
archive.loc[:, ('characteristics', 'pad')] = info.loc[:, 'depth'] / 70 + 32
archive.loc[:, ('characteristics', 'amplitude')] = info.loc[:, 'amp']
archive.loc[:, ('characteristics', 'overall_firing_rate')] = info.loc[:, 'fr']
archive.loc[:, ('characteristics', 'purity')] = info.loc[:, 'purity']
archive.loc[:, ('characteristics', 'data_row')] = info.loc[:, 'ch']

archive.to_pickle(target_folder + 'archive.pkl')

end_time = time.time()
print('Post_processing for animal {} done! \nTime needed: {} minutes'.format(animal, (end_time - start_time) / 60))
if alert_when_done:
    utils.alert()
