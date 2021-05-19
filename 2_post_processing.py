animal = '111'


import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os



animal_folder = r'E:/anxiety_ephys/' + animal + '/'
experiment_names = os.listdir(animal_folder)[:-1]
experiment_name = experiment_names[0]
sub_folder = 'circus'
circus = animal_folder + sub_folder + '/'
folder = circus + 'dat_files_mod/' + experiment_name + '_0.GUI/'

stimes = 'spike_times.npy'
sclusters = 'spike_clusters.npy'
infofile = 'cluster_info.tsv'




info = pd.read_csv(folder+infofile, sep = '\t', index_col=0)
idx=np.array(info.index)
group=np.array(info.group)
indexer = np.zeros(idx.max()+1, dtype = np.int8)
n=1
for i in range(idx.size):
    if group[i] == 'good':
        indexer[idx[i]] = n
        n+=1



spiketimes = np.load(folder + stimes)
spiketimes = spiketimes*50//20000  #sampled at 50Hz
clusters = np.load(folder + sclusters)
data = np.zeros((int(indexer.max()+1), int(max(spiketimes)+1)), dtype=np.uint8)



for index, time in enumerate(spiketimes):
    data[indexer[clusters[index]],time] += 1



cluster_names = np.zeros(data.shape[0], dtype = np.uint8)
for i in range(1,cluster_names.size):
    cluster_names[i] =  np.where(indexer == i)[0]
cluster_names[0] = -1
np.save(circus + 'cluster_names', cluster_names)


lw =0.4
fig = plt.figure(figsize = (30,1.5 * data.shape[0]))
gs = fig.add_gridspec(data.shape[0]-1, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
for i, ax in tqdm(enumerate(axs)):
    ax.plot(data[i+1, :data.shape[1]//16], label = str(i), linewidth = lw)
    ax.legend(loc = 'upper right')
#plt.ylim(-250, 250)


logbook_1 = np.load(circus + 'logbook.npy')
logbook_2 = np.zeros(len(experiment_names)+1)
logbook_2[1:] = logbook_1[:len(experiment_names)]

logbook_3 = np.zeros(len(experiment_names)+1, dtype=np.int64)
for i in range(logbook_3.size):
    logbook_3[i] = np.sum(logbook_2[0:i+1])
logbook_3 = logbook_3*50//20000

for i in range(logbook_3.size-1):
    np.save(circus + experiment_names[i], data[:,logbook_3[i]:logbook_3[i+1]])
