animal = '012'
toplot = ['statistics']  # subselection of: ['raw', 'classic', 'environment', 'transitions', 'statistics']

delete_plot_folder = False
show = True
save = False

import os
import shutil
import time

import numpy as np
import pickle5 as pkl

import plots

mouse_is_late = {'2021-02-19_mBWfus010_EZM_ephys': 70,
                 '2021-02-19_mBWfus009_EZM_ephys': 42,
                 '2021-02-26_mBWfus012_EZM_ephys': 35}

sorter = 'circus'
data_folder = r'E:/anxiety_ephys/'
target_folder = data_folder + animal + '/' + sorter + '/'
all_plots = target_folder + 'plots/'
framerate = 50

animal_folder = data_folder + animal + '/'
experiment_names = os.listdir(animal_folder)
if 'circus' in experiment_names:
    experiment_names.remove('circus')

# delete old plots
if os.path.exists(all_plots):
    if delete_plot_folder:
        shutil.rmtree(all_plots, ignore_errors=True)
        os.mkdir(all_plots)
else:
    os.mkdir(all_plots)
time.sleep(5)

for experiment_name in experiment_names:

    if experiment_name[-7] == 'M':
        environment = 'EZM'
    elif experiment_name[-7] == 'F':
        environment = 'OFT'
    else:
        continue
    eventfile = data_folder + animal + '/' + experiment_name + '/ephys_processed/' + experiment_name + '_events.pkl'
    datafile = target_folder + experiment_name + '.npy'
    ptriggerfile = target_folder + experiment_name + '_trigger.npy'
    mtriggerfile = data_folder + animal + '/' + experiment_name + '/log.txt'
    plot_folder = all_plots + experiment_name + '/'

    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)

    with open(mtriggerfile, 'r') as file:
        data = file.read().replace('\n', ' ')
    point = data.index('.')
    video_trigger = int(float(data[point - 2:point + 3]) * framerate)

    physio_trigger = int(np.load(ptriggerfile) * framerate // 20000)

    raw_data = np.load(datafile) * framerate
    raw_data = raw_data.astype(np.float32, copy=False)

    with open(eventfile, 'rb') as f:
        events = pkl.load(f)

    # offset for nan in 2021-02-26_mBWfus012_EZM_ephys_movement
    # nans = np.where(np.isnan(events['movement']['calib_traj_y'][video_trigger:]))[0]
    # if nans.shape[0] == 0:
    #     off = 0
    # else:
    #     off = nans[-1] + 1
    if experiment_name in mouse_is_late:
        off = (mouse_is_late[experiment_name] + 20)*framerate
    cluster_names = np.load(target_folder + 'cluster_names.npy')
    #################################
    if 'raw' in toplot:
        plots.plot_raw(environment, plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger
                       , cluster_names, minp=0, maxp=90, n=150, show=show, save=save)
    #################################
    if 'classic' in toplot:
        plots.plot_classic(environment, plot_folder, experiment_name, raw_data, events, video_trigger, off,
                           physio_trigger, cluster_names, sigma=10, minp=0, maxp=95, n=150, show=show, save=save)
    #################################
    if environment == 'EZM':
        if 'environment' in toplot:
            plots.plot_circle(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
                              cluster_names, n=360, sigma=-1, show=show, save=save)
        if 'transitions' in toplot:
            # plotmode is one of ['std', 'percent']
            plots.plot_transitions(plot_folder, experiment_name, raw_data, events, cluster_names, video_trigger,
                                   mode='lingering_exittime', plotmode='percent', n=200, m=5, show=show, save=save)
        if 'statistics' in toplot:
            plots.plot_arms(plot_folder, experiment_name, raw_data, events, video_trigger, off,
                            physio_trigger,
                            cluster_names, transition_size=5, minp=0, maxp=90, n=150, show=show, save=save)
    else:
        if 'environment' in toplot:
            plots.plot_grid(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
                            cluster_names, minp=0, maxp=100, n=5, show=show, save=save)
        if 'statistics' in toplot:
            plots.plot_corners(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
                               cluster_names, n=5, show=show, save=save)
#################################
