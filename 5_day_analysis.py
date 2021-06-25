######################################
animal = '309'  # one of the sets with one day
toplot = []# ['raw', 'trace_filtered', 'trace', 'environment', 'transitions', 'statistics', 'phase']
# subselection of: ['raw', 'trace_filtered', 'trace', 'environment', 'transitions', 'statistics', 'phase']
# for full archive need at least ['transitions', 'statistics', 'phase']
max_duration = 1200  # seconds # if negative, crops from end, else from start

delete_plot_folder = False
show = False
save = False
do_archive = True
single_figures = False
multi_figure = False
alert_when_done = False
######################################
import copy
import pandas as pd
import os
import shutil
import time
import numpy as np
import pickle5 as pkl
import day_plots
from natsort import natsorted
import utils

make_path_visible = 0.0001

transition_keys = ['open_closed_entrytime', 'open_closed_exittime', 'closed_open_entrytime', 'closed_open_exittime',
                   'lingering_entrytime', 'lingering_exittime', 'prolonged_open_closed_entrytime',
                   'prolonged_open_closed_exittime',
                   'prolonged_closed_open_entrytime', 'prolonged_closed_open_exittime', 'withdraw_entrytime',
                   'withdraw_exittime',
                   'nosedip_starttime',
                   'nosedip_stoptime']
# subselection of ['open_closed_entrytime', 'open_closed_exittime', 'closed_open_entrytime', 'closed_open_exittime',
# 'lingering_entrytime', 'lingering_exittime', 'prolonged_open_closed_entrytime', 'prolonged_open_closed_exittime',
# 'prolonged_closed_open_entrytime', 'prolonged_closed_open_exittime', 'withdraw_entrytime', 'withdraw_exittime',
# 'nosedip_starttime', 'nosedip_stoptime']

start_time = time.time()
sorter = 'circus'
data_folder = r'E:/anxiety_ephys/'
target_folder = data_folder + animal + '/' + sorter + '/'
all_plots = target_folder + 'plots/'
frame_rate = 50
sampling_rate = 20000

number_of_bins_transitions = 20  # in 10 second window around transitions
number_of_bins_phase = 8  # phaseplot
# factor = 360//number_of_bins_phase

animal_folder = data_folder + animal + '/'
experiment_names = natsorted(os.listdir(animal_folder))
if 'circus' in experiment_names:
    experiment_names.remove('circus')



cluster_names = np.load(target_folder + 'utils/cluster_names.npy')
vHIP_pads = np.load(target_folder + 'utils/vHIP_pads.npy')

archive = pd.read_pickle(target_folder + 'archive.pkl')

for experiment_name in experiment_names:
    plot_folder = all_plots + experiment_name + '/'

    if experiment_name[-7] == 'M':
        environment = 'EZM'
    elif experiment_name[-7] == 'F':
        environment = 'OFT'
    elif experiment_name[-18:-16] == 'be':
        environment = 'before'
    elif experiment_name[-17:-15] == 'af':
        environment = 'after'
    else:
        continue

    eventfile = data_folder + animal + '/' + experiment_name + '/ephys_processed/' + experiment_name + '_events.pkl'
    spikes_50_file = target_folder + 'spikes_50/' + experiment_name + '.npy'
    # ptriggerfile = target_folder + experiment_name + '_trigger.npy'


    # physio_trigger = int(np.load(ptriggerfile) * frame_rate // 20000)

    # get events/movement
    with open(eventfile, 'rb') as f:
        events = pkl.load(f)

    spikes_50 = np.load(spikes_50_file) * frame_rate
    xy = np.load(target_folder + 'movement_files/' + experiment_name + '.npy')
    aligned = utils.create_aligned(spikes_50, xy, max_duration, make_path_visible)
    if do_archive:
        archive.loc[:, ('characteristics', 'mean_' + environment)] = np.mean(aligned[2:], axis=1)

#################################
    if 'phase' in toplot:
        #    np.save(target_folder + 'vHIP_phase/' + experiment_name, hilbert_phase)
        phase_aligned = (np.load(
            target_folder + 'vHIP_phase/' + experiment_name + '.npy') + 180) * number_of_bins_phase // 360
        spikes_20000_aligned = np.load(target_folder + 'spikes_20000/' + experiment_name + '.npy')

        if max_duration >= 0:
            phase_aligned = phase_aligned[:, :max_duration * sampling_rate]
            spikes_20000_aligned = spikes_20000_aligned[:, :max_duration * sampling_rate]

        else:
            phase_aligned = phase_aligned[:, max_duration * sampling_rate:]
            spikes_20000_aligned = spikes_20000_aligned[:, max_duration * sampling_rate:]

        archive = day_plots.plot_phase(phase_aligned, spikes_20000_aligned, vHIP_pads, plot_folder, experiment_name,
                                       cluster_names, archive,
                                       environment, number_of_bins=number_of_bins_phase, show=show,
                                       save=save, do_archive=do_archive, single_figures=single_figures,
                                       multi_figure=multi_figure)

#################################
    if environment == 'EZM':
        if 'trace' in toplot:
            day_plots.plot_trace(environment, plot_folder, experiment_name, aligned, cluster_names,
                                 single_figures=single_figures, multi_figure=multi_figure, sigma=10, minp=0, maxp=95, n=150,
                                 show=show, save=save, filter=False)
        if 'trace_filtered' in toplot:
            day_plots.plot_trace(environment, plot_folder, experiment_name, aligned, cluster_names,
                                 single_figures=single_figures, multi_figure=multi_figure, sigma=10, minp=0, maxp=95, n=150,
                                 show=show, save=save, filter=True)

        if 'environment' in toplot:
            day_plots.plot_circle(plot_folder, experiment_name, aligned, cluster_names, single_figures=single_figures,
                                  multi_figure=multi_figure,
                                  n=360, sigma=-1, show=show, save=save)  # sigma = -1 sets sigma matching n
        if 'transitions' in toplot:
            for mode in transition_keys:
                event_indices = events['transitions'][mode]
                archive = day_plots.plot_events(plot_folder, experiment_name, aligned, cluster_names, mode, event_indices,
                                                archive, single_figures, multi_figure,
                                                n=250, number_of_bins=20, show=show, save=save, do_archive=do_archive)
        if 'statistics' in toplot:
            archive = day_plots.plot_arms(plot_folder, experiment_name, aligned, cluster_names, archive, single_figures,
                                          multi_figure, transition_size=5, n=150, show=show, save=save,
                                          do_archive=do_archive)

#################################
    elif environment == 'OFT':
        if 'trace' in toplot:
            day_plots.plot_trace(environment, plot_folder, experiment_name, aligned, cluster_names,
                                 single_figures=single_figures, multi_figure=multi_figure, sigma=10, minp=0, maxp=95, n=150,
                                 show=show, save=save, filter=False)

        if 'trace_filtered' in toplot:
            day_plots.plot_trace(environment, plot_folder, experiment_name, aligned, cluster_names,
                                 single_figures=single_figures, multi_figure=multi_figure, sigma=10, minp=0, maxp=95, n=150,
                                 show=show, save=save, filter=True)

        if 'environment' in toplot:
            day_plots.plot_grid(plot_folder, experiment_name, aligned, cluster_names, single_figures=single_figures,
                                multi_figure=multi_figure, minp=0,
                                maxp=100, n=5, show=show, save=save)
        if 'statistics' in toplot:
            archive = day_plots.plot_corners(plot_folder, experiment_name, aligned, cluster_names, archive,
                                             single_figures=single_figures, multi_figure=multi_figure,
                                             n=4, show=show, save=save, do_archive=do_archive)

#################################
    if do_archive:
        if environment == 'EZM':
            archive.loc[:, ('characteristics', 'ezm_closed_score')], archive.loc[:,
                                                                     ('characteristics', 'ezm_transition_score')], \
            archive.loc[:, ('characteristics', 'ezm_closed')], archive.loc[:, ('characteristics',
                                                                               'ezm_transition')] = day_plots.get_ezm_score(
                archive.loc[:, 'ROI_EZM'].values)
        elif environment == 'OFT':
            archive.loc[:, ('characteristics', 'of_corners_score')], archive.loc[:,
                                                                     ('characteristics', 'of_middle_score')], \
            archive.loc[:, ('characteristics', 'of_corners')], archive.loc[:,
                                                               ('characteristics', 'of_middle')] = day_plots.get_of_score(
                archive.loc[:, 'ROI_OF'].values)
if do_archive:
    archive.to_pickle(target_folder + 'archive.pkl')
end_time = time.time()
print('Day_analysis for animal {} done! \nTime needed: {} minutes'.format(animal, (end_time-start_time)/60))
if alert_when_done:
    utils.alert()