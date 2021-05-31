animal = '211'  # one of the sets with one day
score = 'ezm_transition_score'  # one of:         ['ezm_closed_score', 'ezm_transition_score', 'of_corners_score', 'of_middle_score']
threshold = 0  # recommended: [0                     ,  x                    ,  x                ,  x               ]
data_separation = ['over_threshold_plus',
                   'over_threshold_minus']  # selection of ['under_threshold_all', 'over_threshold_all', 'under_threshold_plus', 'over_threshold_plus',
# 'under_threshold_minus', 'over_threshold_minus':]

#significance plus/minus: plus['ezm_closed_score': firing rate higer in closed area,
# 'ezm_transition_score': firing rate higher in transition zones, 'of_corners_score': firing rate higher in corners,
# 'of_middle_score': firing rate higher in the middle]

toplot = ['arms', 'circle']  # selection of: ['circle', 'grid', 'arms', 'corners', 'transitions', 'phase']
transition_modes = []  # selection of: ['open_closed_entrytime', 'open_closed_exittime', 'closed_open_entrytime', 'closed_open_exittime',
# 'lingering_entrytime', 'lingering_exittime', 'prolonged_open_closed_entrytime', 'prolonged_open_closed_exittime',
# 'prolonged_closed_open_entrytime', 'prolonged_closed_open_exittime', 'withdraw_entrytime', 'withdraw_exittime',
# 'nosedip_starttime', 'nosedip_stoptime']
phase_modes = []  # selection of: ['theta_phase_OFT', 'theta_phase_EZM', 'theta_phase_before', 'theta_phase_after']
delete_plot_folder = False
show = True
save = False

import copy
import pandas as pd
import os
import shutil
import time
import numpy as np
import pickle5 as pkl
import overall_plots

sorter = 'circus'
data_folder = r'E:/anxiety_ephys/'
target_folder = data_folder + animal + '/' + sorter + '/'
overall_plot_folder = target_folder + 'overall_plots/'
framerate = 50

if os.path.exists(overall_plot_folder):
    if delete_plot_folder:
        shutil.rmtree(overall_plot_folder, ignore_errors=True)
        os.mkdir(overall_plot_folder)
else:
    os.mkdir(overall_plot_folder)
time.sleep(5)
vHIP_pads = np.load(target_folder + 'phase_files/' + 'vHIP_pads.npy')

archive = pd.read_pickle(target_folder + 'archive.pkl')
plusminus = score[:-6]
# scorer = archive.loc[:,('characteristics', score)]
# n1 = archive.loc[:,('characteristics', score)].values < threshold
# n2 = archive.index != -1

under_threshold_all = archive.loc[np.logical_and(archive.loc[:,('characteristics', score)].values < threshold , archive.index != -1)]
under_threshold_plus = archive.loc[np.logical_and(np.logical_and(archive.loc[:,('characteristics', score)].values < threshold, archive.index != -1), archive.loc[:,
    ('characteristics', plusminus)] > 0)]
under_threshold_minus = archive.loc[np.logical_and(np.logical_and(archive.loc[:,('characteristics', score)].values < threshold, archive.index != -1), archive.loc[:,
    ('characteristics', plusminus)] <= 0)]
over_threshold_all = archive.loc[np.logical_and(archive.loc[:,('characteristics', score)].values >= threshold, archive.index != -1)]
over_threshold_plus = archive.loc[np.logical_and(np.logical_and(archive.loc[:,('characteristics', score)].values >= threshold, archive.index != -1), archive.loc[:,
    ('characteristics', plusminus)] > 0)]
over_threshold_minus = archive.loc[np.logical_and(np.logical_and(archive.loc[:,('characteristics', score)].values >= threshold, archive.index != -1), archive.loc[:,
    ('characteristics', plusminus)] <= 0)]

for name in data_separation:
    if name == 'under_threshold_all':
        data = under_threshold_all
    elif name == 'over_threshold_all':
        data = over_threshold_all
    elif name == 'under_threshold_plus':
        data = under_threshold_plus
    elif name == 'over_threshold_plus':
        data = over_threshold_plus
    elif name == 'under_threshold_minus':
        data = under_threshold_minus
    elif name == 'over_threshold_minus':
        data = over_threshold_minus
    else:
        raise Exception('invalid data seperation entry')
    if 'circle' in toplot:
        overall_plots.plot_circle(overall_plot_folder, animal, data.loc[:,'ROI_EZM'], name, show=show, save=save)
    if 'grid' in toplot:
        overall_plots.plot_grid(overall_plot_folder, animal, data.loc[:,'ROI_OF'], name, show=show, save=save)
    if 'arms' in toplot:
        overall_plots.plot_arms(overall_plot_folder, animal, data.loc[:,'ROI_EZM'], name, show=show, save=save)
    if 'corners' in toplot:
        overall_plots.plot_corners(overall_plot_folder, animal, data.loc[:,'ROI_OF'], name, show=show, save=save)
    if 'phase' in toplot:
        for phase_mode in phase_modes:
            for pad in vHIP_pads:
                overall_plots.plot_phase(overall_plot_folder, animal, data.loc[:,phase_mode + '_' + str(pad)], name,
                                         mode=phase_mode, show=show, save=show)
    if 'transitions' in toplot:
        for transition_mode in transition_modes:
            overall_plots.plot_transitions(overall_plot_folder, animal, data.loc[:,transition_mode], name,
                                           mode=transition_mode, show=show, save=save)


print('overall_analysis for animal {} done!'.format(animal))
