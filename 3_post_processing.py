######################################
animal = '111'
alert_when_done = True
######################################

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from natsort import natsorted
import utils
import time


start_time = time.time()
frame_rate = 50
sampling_rate = 20000
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


number_of_bins_transitions = 20  # in 5 second window around transitions
number_of_bins_phase = 8  # phaseplot
number_of_bins_isi = 200

info = pd.read_csv(folder+infofile, sep = '\t', index_col=0)
idx=np.array(info.index)
group=np.array(info.group)
indexer = np.zeros(idx.max()+1, dtype = np.int8)
n=1
for i in range(idx.size):
    if group[i] == 'good':
        indexer[idx[i]] = n
        n+=1

logbook_1 = np.load(target_folder + 'utils/logbook.npy')
logbook_2 = np.zeros(len(experiment_names)+1)
logbook_2[1:] = logbook_1[:len(experiment_names)]

logbook_3 = np.zeros(len(experiment_names)+1, dtype=np.int64)
for i in range(logbook_3.size):
    logbook_3[i] = np.sum(logbook_2[0:i+1])
original_logbook_3 = logbook_3
logbook_3 = logbook_3*frame_rate//sampling_rate



original_spiketimes = np.load(folder + stimes)
spiketimes = original_spiketimes*frame_rate//sampling_rate  #sampled at 50Hz
clusters = np.load(folder + sclusters)
data = np.zeros((int(indexer.max()+1), logbook_3[-1]), dtype=np.uint8)
original_data = np.zeros((int(indexer.max()+1), original_logbook_3[-1]), dtype=bool)



for index, timepoint in enumerate(spiketimes):
    data[indexer[clusters[index]],timepoint] += 1

for index, timepoint in enumerate(original_spiketimes):
    original_data[indexer[clusters[index]], timepoint] += 1



cluster_names = np.zeros(data.shape[0], dtype = np.uint8)
for i in range(1,cluster_names.size):
    cluster_names[i] =  np.where(indexer == i)[0]
cluster_names[0] = 255
np.save(target_folder + 'utils/cluster_names', cluster_names.astype(np.uint16))


# lw =0.4
# fig = plt.figure(figsize = (30,1.5 * data.shape[0]))
# gs = fig.add_gridspec(data.shape[0]-1, hspace=0)
# axs = gs.subplots(sharex=True, sharey=True)
# for i, ax in tqdm(enumerate(axs)):
#     ax.plot(data[i+1, :data.shape[1]//16], label = str(i), linewidth = lw)
#     ax.legend(loc = 'upper right')
# #plt.ylim(-250, 250)
#


for i in range(logbook_3.size-1):
    np.save(target_folder + 'spikes_50/' + experiment_names[i], data[:,logbook_3[i]:logbook_3[i+1]].astype(np.uint8))
    np.save(target_folder + 'spikes_20000/' + experiment_names[i], original_data[:,original_logbook_3[i]:original_logbook_3[i+1]].astype(bool))

vHIP_pads = np.load(target_folder + 'utils/vHIP_pads.npy')
archive = utils.create_archive(vHIP_pads, cluster_names, number_of_bins_transitions, number_of_bins_phase, number_of_bins_isi)
archive.loc[:, ('characteristics', 'pad')] = info.loc[:, 'depth'] / 70 + 32
archive.loc[:, ('characteristics', 'amplitude')] = info.loc[:, 'amp']
archive.loc[:, ('characteristics', 'overall_firing_rate')] = info.loc[:, 'fr']
archive.loc[:, ('characteristics', 'purity')] = info.loc[:, 'purity']
archive.loc[:, ('characteristics', 'data_row')] = info.loc[:, 'ch']

archive.to_pickle(target_folder + 'archive.pkl')

end_time = time.time()
print('Post_processing for animal {} done! \nTime needed: {} minutes'.format(animal, (end_time-start_time)/60))
if alert_when_done:
    utils.alert()