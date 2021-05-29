
animal = '012' #one of the sets with one day
toplot = ['phase']  # subselection of: ['raw', 'classic', 'environment', 'transitions', 'statistics', 'phase']
             # for full archive need at least ['transitions', 'statistics', 'phase]

delete_plot_folder = False
show = True
save = False
do_archive = False
import copy
import pandas as pd
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

number_of_bins_transitions = 40 #in 10 second window around transitions
number_of_bins_phase = 8 #phaseplot
#factor = 360//number_of_bins_phase

animal_folder = data_folder + animal + '/'
experiment_names = os.listdir(animal_folder)
if 'circus' in experiment_names:
    experiment_names.remove('circus')

# delete old plots
if os.path.exists(all_plots):
    if delete_plot_folder:
        shutil.rmtree(all_plots, ignore_errors=True)
        time.sleep(5)
        os.mkdir(all_plots)
else:
    os.mkdir(all_plots)

cluster_names = np.load(target_folder + 'cluster_names.npy')
vHIP_pads = np.load(target_folder + 'phase_files/' + 'vHIP_pads.npy')

level_1 = ['characteristics' for _ in range(12)] \
          + ['ROI_EZM' for _ in range(8)] \
          + ['ROI_OF' for _ in range(9)] \
          + ['open_closed_entrytime' for _ in range(number_of_bins_transitions)] \
          + ['open_closed_exittime' for _ in range(number_of_bins_transitions)] \
          + ['closed_open_entrytime' for _ in range(number_of_bins_transitions)] \
          + ['closed_open_exittime' for _ in range(number_of_bins_transitions)]\
          + ['lingering_entrytime' for _ in range(number_of_bins_transitions)] \
          + ['lingering_exittime' for _ in range(number_of_bins_transitions)] \
          + ['prolonged_open_closed_entrytime' for _ in range(number_of_bins_transitions)] \
          + ['prolonged_open_closed_exittime' for _ in range(number_of_bins_transitions)] \
          + ['prolonged_closed_open_entrytime' for _ in range(number_of_bins_transitions)]\
          + ['prolonged_closed_open_exittime' for _ in range(number_of_bins_transitions)]\
          + ['withdraw_entrytime' for _ in range(number_of_bins_transitions)]\
          + ['withdraw_exittime' for _ in range(number_of_bins_transitions)] \
          + ['nosedip_starttime' for _ in range(number_of_bins_transitions)] \
          + ['nosedip_stoptime' for _ in range(number_of_bins_transitions)]

for pad in vHIP_pads:
    level_1 += ['theta_phase_OFT_' + str(pad) for _ in range(number_of_bins_phase)]
    level_1 += ['theta_phase_EZM_' + str(pad) for _ in range(number_of_bins_phase)]
    level_1 += ['theta_phase_before_' + str(pad) for _ in range(number_of_bins_phase)]
    level_1 += ['theta_phase_after_' + str(pad) for _ in range(number_of_bins_phase)]



five_sec_range = list(np.arange(number_of_bins_transitions))
transition_ranges = copy.copy(five_sec_range)
for i in range(13):
    transition_ranges.extend(five_sec_range)
degree360 = list(np.arange(number_of_bins_phase))
phase_ranges = copy.copy(degree360)
for i in range(4*len(vHIP_pads)):
    phase_ranges.extend(degree360)
level_2 = ['ezm_closed_score', 'ezm_transition_score', 'ezm_closed', 'ezm_transition', 'of_corners_score',
           'of_middle_score', 'of_corners', 'of_middle', 'mean_before', 'mean_after', 'mean_EZM', 'mean_OFT'] \
          + [i for i in range(8)] \
          + [i for i in range(9)] \
          + transition_ranges \
          + phase_ranges
tuples = list(zip(level_1, level_2))
columns = pd.MultiIndex.from_tuples(tuples)
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
    accomodation = 20
    if experiment_name in mouse_is_late:
        off = (mouse_is_late[experiment_name] + accomodation)*framerate
    else:
        off = 0

    if do_archive:
        archive.loc[:, ('characteristics', 'mean_'+environment)] = np.mean(raw_data[:, physio_trigger+off:], axis=1)
    #################################
    if 'phase' in toplot:
        archive = plots.plot_phase(vHIP_pads, target_folder, plot_folder, experiment_name, off,
                              physio_trigger,
                              cluster_names, archive, environment, number_of_bins = number_of_bins_phase, show=show, save=save, do_archive=do_archive)
    #################################
    if environment == 'EZM':
        if 'raw' in toplot:
            plots.plot_raw(environment, plot_folder, experiment_name, raw_data, events, video_trigger, off,
                           physio_trigger
                           , cluster_names, minp=0, maxp=90, n=150, show=show, save=save)
        #################################
        if 'classic' in toplot:
            plots.plot_classic(environment, plot_folder, experiment_name, raw_data, events, video_trigger, off,
                               physio_trigger, cluster_names, sigma=10, minp=0, maxp=95, n=150, show=show, save=save)
        if 'environment' in toplot:
            plots.plot_circle(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
                              cluster_names, n=360, sigma=-1, show=show, save=save)
        if 'transitions' in toplot:
            for mode in transition_keys:
                # plotmode is one of ['std', 'percent']
                archive = plots.plot_transitions(plot_folder, experiment_name, raw_data, events, cluster_names, video_trigger, archive,
                                       mode=mode, n=200, number_of_bins=number_of_bins_transitions, show=show, save=save, do_archive=do_archive)
        if 'statistics' in toplot:
            archive = plots.plot_arms(plot_folder, experiment_name, raw_data, events, video_trigger, off,
                            physio_trigger,
                            cluster_names, archive, transition_size=5, n=150, show=show, save=save, do_archive=do_archive)

    elif environment == 'OFT':
        if 'raw' in toplot:
            plots.plot_raw(environment, plot_folder, experiment_name, raw_data, events, video_trigger, off,
                           physio_trigger
                           , cluster_names, minp=0, maxp=90, n=150, show=show, save=save)
        #################################
        if 'classic' in toplot:
            plots.plot_classic(environment, plot_folder, experiment_name, raw_data, events, video_trigger, off,
                               physio_trigger, cluster_names, sigma=10, minp=0, maxp=95, n=150, show=show, save=save)
        if 'environment' in toplot:
            plots.plot_grid(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
                            cluster_names, minp=0, maxp=100, n=4, show=show, save=save)
        if 'statistics' in toplot:
            archive = plots.plot_corners(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
                               cluster_names, archive, n=4, show=show, save=save, do_archive=do_archive)

    #################################
    if do_archive:
        if environment == 'EZM':
            archive.loc[:, ('characteristics','ezm_closed_score')], archive.loc[:, ('characteristics', 'ezm_transition_score')],\
            archive.loc[:, ('characteristics', 'ezm_closed')], archive.loc[:, ('characteristics', 'ezm_transition')] = plots.get_ezm_score(archive.loc[:, 'ROI_EZM'].values)
        elif environment == 'OFT':
            archive.loc[:, ('characteristics', 'of_corners_score')], archive.loc[:, ('characteristics', 'of_middle_score')],\
            archive.loc[:, ('characteristics', 'of_corners')], archive.loc[:, ('characteristics', 'of_middle')] = plots.get_of_score(archive.loc[:, 'ROI_OF'].values)
if do_archive:
    archive.to_pickle(target_folder + 'archive')
pass
