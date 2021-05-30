import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import math
from matplotlib import cm
import matplotlib as mpl
import copy
import pandas as pd


make_path_visible = 0.0001
idx = pd.IndexSlice
#controlled and commented
def plot_circle(overall_plots, animal, ROI, name, show=False, save=True):
    mode = 'circle'
    file_name = overall_plots + animal + '_' + mode + '_' + name
    colorcoding = np.empty(20)
    mean = np.mean(ROI, axis=0)
    colorcoding[2] = mean[0]
    colorcoding[7] = mean[1]
    colorcoding[12] = mean[2]
    colorcoding[17] = mean[3]
    colorcoding[3:7] = mean[4]
    colorcoding[8:12] = mean[5]
    colorcoding[13:17] = mean[6]
    colorcoding[18:] = mean[7]
    colorcoding[:2] = mean[7]



    colors = cm.jet(plt.Normalize()(colorcoding))
    fig = plt.figure(figsize=(5, 5))
    fig.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=mean.min(), vmax=mean.max()), cmap='jet'),
                 shrink=0.6)
    plt.pie(np.ones(20), colors=colors)
    my_circle = plt.Circle((0, 0), 0.8, color='white')
    fig.gca().add_artist(my_circle)
    if save:
        plt.savefig(file_name + '.jpg')
    if show:
        plt.title('ROI ' + animal )
        plt.show()
    plt.close(fig)
    return

#corrected and commented
#plots grid with mean values of ROI(OF) of selected unit
def plot_grid(overall_plots, animal, ROI, name, show=False, save=True):
    mode = 'grid'
    n = 5
    file_name = overall_plots + animal + '_' + mode + '_' + name
    grid = np.zeros((n, n), dtype=np.float32)
    mean = np.mean(ROI, axis=0)
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_bad(color='grey')
    indices = [6,8,8,8,4]
    for d in range(n):
        grid[1:n - 1, d] = mean[indices[d]]
    grid[0, 1:n - 1] = mean[5]
    grid[n, 1:n - 1] = mean[7]
    grid[0,0] = mean[1]
    grid[0,n] = mean[0]
    grid[n,0] = mean[2]
    grid[n,n] = mean[3]

    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(grid, cmap=cmap, origin='lower', interpolation='none')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save:
        plt.savefig(file_name + '.jpg')
    if show:
        plt.title('ROI ' + animal)
        plt.show()
    plt.close(fig)
    return

#corrected and commented
#barplot with mean of z scores of selected units for timebins around transition, error bars: SEM
def plot_transitions(overall_plots, animal, transition_window, name, mode, show=False, save=True):
    plotmode = 'std'
    file_name = overall_plots + animal + '_' + mode + '_' + name

    mean = np.mean(transition_window, axis=0)
    std = np.std(transition_window, axis=0)
    fig = plt.figure(figsize=(5, 5))
    n_of_samples = transition_window.shape[0]
    sem = std / np.sqrt(n_of_samples)  # compute SEM (standard error of the mean)

    toplot = mean
    errorbar = sem
    plt.bar(np.arange(transition_window.shape[1]), toplot, yerr=errorbar, width=1)
    if save:
        plt.savefig(file_name + '.jpg')
    if show:
        plt.title('transition ' + mode + ' ' + name)
        plt.show()
    plt.close(fig)
    return

#corrected and commented
#plots mean of selected units regarding the percent difference of ROI(EZM) mean from overall mean of unit
def plot_arms(overall_plots, animal, ROI, name, show=False, save=True):
    mode = 'arms'
    file_name = overall_plots + animal + '_' + mode + '_' + name

    mean = np.mean(ROI, axis=0)
    std = np.std(ROI, axis=0)
    n_of_samples = ROI.shape[0]
    sem = std / np.sqrt(n_of_samples)  # compute SEM (stadard error of the mean)
    fig = plt.figure(figsize=(5, 5))
    toplot = mean[[0, 1, 2, 3, 4, 6, 5, 7]]
    errorbar = sem[[0, 1, 2, 3, 4, 6, 5, 7]] #arrangement: ['top right', 'top left', 'bottom left',
            # 'bottom right', 'top (open)', 'bottom (open)', 'left (closed)', 'right (closed)']
    plt.bar(np.arange(8), toplot, yerr=errorbar, width=1)
    if save:
        plt.savefig(file_name + '.jpg')
    if show:
        plt.title(file_name)
        plt.show()
    plt.close(fig)
    return


#corrected and commented
#plots mean of selected units regarding the percent difference of ROI(OF) mean from overall mean of unit
def plot_corners(overall_plots, animal, ROI, name, show=False, save=True):
    mode = 'corners'
    file_name = overall_plots + animal + '_' + mode + '_' + name

    mean = np.mean(ROI, axis=0)
    std = np.std(ROI, axis=0)
    n_of_samples = ROI.shape[0]
    sem = std / np.sqrt(n_of_samples) # compute SEM (stadard error of the mean)

    fig = plt.figure(figsize=(5, 5))
    toplot = mean  #arrangement of the ROIs: {0: 'top right', 1: 'top left', 2: 'bottom left',
                                             # 3: 'bottom right', 4: 'right', 5: 'top', 6: 'left', 7: 'bottom', 8: 'middle'}
    errorbar = sem
    plt.bar(np.arange(8), toplot, yerr=errorbar, width=1)
    if save:
        plt.savefig(file_name + '.jpg')
    if show:
        plt.title(file_name)
        plt.show()
    plt.close(fig)
    return

#controlled and commented
#plots barplot of the mean phase of all included units errror bars are SEM
def plot_phase(overall_plots, animal, binned, name, mode, show=False, save=True):
    file_name = overall_plots + animal + '_' + mode + '_' + name
    normalized = binned / np.mean(binned, axis=1)[:, None] * np.mean(binned)
    mean = np.mean(normalized, axis=0)
    std = np.std(normalized, axis=0)
    n_of_samples = normalized.shape[0]
    sem = std / np.sqrt(n_of_samples) # compute SEM (stadard error of the mean)

    fig = plt.figure(figsize=(5, 5))
    toplot = mean
    errorbar = sem
    plt.bar(np.arange(mean.shape[1]), toplot, yerr=errorbar, width=1)
    if save:
        plt.savefig(file_name + '.jpg')
    if show:
        plt.title(file_name)
        plt.show()
    plt.close(fig)
    return