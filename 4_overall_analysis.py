
animal = '211' #one of the sets with one day
score = [] #one of:         ['ezm_open_close_score', 'ezm_transition_score', 'of_corners_score', 'of_middle_score']
threshold =  #recommended: [x                     ,  x                    ,  x                ,  x               ]
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

archive = pd.read_pickle(target_folder + 'archive.pkl')
under_threshold = archive[archive[score] < threshold]
over_threshold = archive[archive[score] >= threshold]


overall_plots.plot_circle(overall_plots, animal, under_threshold.loc['ROI_EZM'], show=show, save=save)
