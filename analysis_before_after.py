import copy

import pandas as pd

animal = '111' #one of the sets with one day
toplot = ['transitions', 'statistics']  # subselection of: ['raw', 'classic', 'environment', 'transitions', 'statistics']
             # for archive need at least ['transitions', 'statistics']

delete_plot_folder = False
show = False
save = False
do_archive=True


import os
import shutil
import time

import numpy as np
import pickle5 as pkl

import plots
transition_keys = ['open_closed_entrytime', 'open_closed_exittime', 'closed_open_entrytime', 'closed_open_exittime',
                  'lingering_entrytime', 'lingering_exittime', 'prolonged_open_closed_entrytime', 'prolonged_open_closed_exittime',
                  'prolonged_closed_open_entrytime', 'prolonged_closed_open_exittime', 'withdraw_entrytime', 'withdraw_exittime',
                  'nosedip_starttime', 'nosedip_stoptime']

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
cluster_names = np.load(target_folder + 'cluster_names.npy')

level_1 = ['characteristics' for _ in range(12)] \
          + ['ROI_EZM' for _ in range(8)] \
          + ['ROI_OF' for _ in range(9)] \
          + ['open_closed_entrytime' for _ in range(1001)] \
          + ['open_closed_exittime' for _ in range(1001)] \
          + ['closed_open_entrytime' for _ in range(1001)] \
          + ['closed_open_exittime' for _ in range(1001)]\
          + ['lingering_entrytime' for _ in range(1001)] \
          + ['lingering_exittime' for _ in range(1001)] \
          + ['prolonged_open_closed_entrytime' for _ in range(1001)] \
          + ['prolonged_open_closed_exittime' for _ in range(1001)] \
          + ['prolonged_closed_open_entrytime' for _ in range(1001)]\
          + ['prolonged_closed_open_exittime' for _ in range(1001)]\
          + ['withdraw_entrytime' for _ in range(1001)]\
          + ['withdraw_exittime' for _ in range(1001)] \
          + ['nosedip_starttime' for _ in range(1001)] \
          + ['nosedip_stoptime' for _ in range(1001)] \
          + ['theta_phase' for _ in range(360)]

five_sec_range = [i for i in range(-500, 501)]
ranges = copy.copy(five_sec_range)
for i in range(13):
    ranges.extend(five_sec_range)
level_2 = ['ezm_open_close_score', 'ezm_transition_score', 'ezm_closed', 'ezm_transition', 'of_corners_score',
           'of_middle_score', 'of_corners', 'of_middle', 'mean_before', 'mean_after', 'mean_EZM', 'mean_OFT'] \
          + [i for i in range(8)] \
          + [i for i in range(9)] \
          + ranges \
          + [i for i in range(-180,180)]
tuples = list(zip(level_1, level_2))
columns = pd.MultiIndex.from_tuples(tuples)
if do_archive:
    archive = pd.DataFrame(index=cluster_names, columns=columns)

for experiment_name in experiment_names:

    if experiment_name[-7] == 'M':
        environment = 'EZM'
    elif experiment_name[-7] == 'F':
        environment = 'OFT'
    elif experiment_name[21:23] == 'be':
        environment = 'before'
    elif experiment_name[21:23] == 'af':
        environment = 'after'

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
    accomodation = 20
    if experiment_name in mouse_is_late:
        off = (mouse_is_late[experiment_name] + accomodation)*framerate
    else:
        off = 0

    if do_archive:
        archive['characteristics']['mean_'+environment] = np.mean(raw_data[:, physio_trigger+off:], axis=1)
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
            for mode in transition_keys:
                # plotmode is one of ['std', 'percent']
                archive = plots.plot_transitions(plot_folder, experiment_name, raw_data, events, cluster_names, video_trigger, archive,
                                       mode=mode, plotmode='percent', n=200, m=5, show=show, save=save, do_archive=do_archive)
        if 'statistics' in toplot:
            archive = plots.plot_arms(plot_folder, experiment_name, raw_data, events, video_trigger, off,
                            physio_trigger,
                            cluster_names, archive, transition_size=5, minp=0, maxp=90, n=150, show=show, save=save, do_archive=do_archive)
    else:
        if 'environment' in toplot:
            plots.plot_grid(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
                            cluster_names, minp=0, maxp=100, n=5, show=show, save=save)
        if 'statistics' in toplot:
            archive = plots.plot_corners(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
                               cluster_names, archive, n=5, show=show, save=save, do_archive=do_archive)
    #################################
    if do_archive:
        if environment == 'EZM':
            archive['characteristics']['ezm_open_close_score'], archive['characteristics']['ezm_transition_score'],\
            archive['characteristics']['ezm_closed'], archive['characteristics']['ezm_transition'] = plots.get_ezm_score(archive['ROI_EZM'])
        elif environment == 'OFT':
            archive['characteristics']['of_corners_score'], archive['characteristics']['of_middle_score'],\
            archive['characteristics']['of_corners'], archive['characteristics']['of_middle'] = plots.get_of_score(archive['ROI_OF'])
pass
