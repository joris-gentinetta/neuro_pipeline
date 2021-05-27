
animal = '211' #one of the sets with one day
score = [] #one of:         ['ezm_open_close_score', 'ezm_transition_score', 'of_corners_score', 'of_middle_score']
threshold =  #recommended: [x                     ,  x                    ,  x                ,  x               ]
data_separation = [] #selection of ['under_threshold_all', 'over_threshold_all', 'under_threshold_plus', 'over_threshold_plus',
                                  # 'under_threshold_minus', 'over_threshold_minus':]
                                  #todo
                                  # significance of plus minus:
toplot = [] #selection of: ['circle', 'grid', 'arms', 'corners', 'transitions', 'phase']
transition_modes =  []    #selection of: ['open_closed_entrytime', 'open_closed_exittime', 'closed_open_entrytime', 'closed_open_exittime',
                          # 'lingering_entrytime', 'lingering_exittime', 'prolonged_open_closed_entrytime', 'prolonged_open_closed_exittime',
                          # 'prolonged_closed_open_entrytime', 'prolonged_closed_open_exittime', 'withdraw_entrytime', 'withdraw_exittime',
                          # 'nosedip_starttime', 'nosedip_stoptime']
phase_modes = [] #selection of: ['theta_phase_OFT', 'theta_phase_EZM', 'theta_phase_before', 'theta_phase_after']
delete_plot_folder = False
show = False
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
overall_plots = target_folder + 'overall_plots/'
framerate = 50

if os.path.exists(overall_plots):
    if delete_plot_folder:
        shutil.rmtree(overall_plots, ignore_errors=True)
        os.mkdir(overall_plots)
else:
    os.mkdir(overall_plots)
time.sleep(5)
vHIP_pads = np.load(target_folder + 'phase_files/' + 'vHIP_pads')

archive = pd.read_pickle(target_folder + 'archive.pkl')
plusminus = score[:-6]

under_threshold_all = archive[archive[('characteristics', score)] < threshold and archive.index != -1]
under_threshold_plus = archive[archive[('characteristics', score)] < threshold and archive.index != -1 and archive[('characteristics', plusminus)] > 0]
under_threshold_minus = archive[archive[('characteristics', score)] < threshold and archive.index != -1 and archive[('characteristics', plusminus)] <= 0]
over_threshold_all= archive[archive[('characteristics', score)] >= threshold and archive.index != -1]
over_threshold_plus = archive[archive[('characteristics', score)] >= threshold and archive.index != -1 and archive[('characteristics', plusminus)] > 0]
over_threshold_minus = archive[archive[('characteristics', score)] >= threshold and archive.index != -1 and archive[('characteristics', plusminus)] <= 0]

for name in under_over:
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
    if 'circle' in toplot:
        overall_plots.plot_circle(overall_plots, animal, data.loc['ROI_EZM'], name, show=show, save=save)
    if 'grid' in toplot:
        overall_plots.plot_grid(overall_plots, animal, data.loc['ROI_OF'], name , show=show, save=save)
    if 'arms' in toplot:
        overall_plots.plot_arms(overall_plots, animal, data.loc['ROI_EZM'], name, show=show, save=save)
    if 'corners' in toplot:
        overall_plots.plot_corners(overall_plots, animal, data.loc['ROI_OF'], name, show=show, save=save)
    if 'phase' in toplot:
        for phase_mode in phase_modes:
            for pad in vHIP_pads:
                overall_plots.plot_phase(overall_plots, animal,  data.loc[phase_mode+'_'+str(pad)], name, mode=phase_mode, show=show, save=show)
    if 'transitions' in toplot:
        for transition_mode in transition_modes:
            overall_plots.plot_transitions(overall_plots, animal, data.loc[transition_mode], name, mode=transition_mode, show=show, save=save)
