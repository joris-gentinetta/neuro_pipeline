import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
import math

make_path_visible = 0.0001
idx = pd.IndexSlice


# plot the mean firing rate per roi in the circle representation:
def plot_circle(plot_folder, ROI, data_separation, show=False, save=True):
    mode = 'circle'
    file_data_separation = plot_folder + mode + '_' + data_separation
    ##assign the the pie segments to the roi's:
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

    colors = cm.Blues(plt.Normalize()(colorcoding))
    fig = plt.figure(figsize=(5, 5))
    fig.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=mean.min(), vmax=mean.max()), cmap='Blues'),
                 shrink=0.6)
    plt.pie(np.ones(20), colors=colors)
    my_circle = plt.Circle((0, 0), 0.8, color='white')
    fig.gca().add_artist(my_circle)
    if save:
        plt.savefig(file_data_separation + '.jpg')
    if show:
        plt.title(mode + ', ' + data_separation)
        plt.show()
    plt.close(fig)
    return


# plots grid with mean values of ROI(OF) of selected unit
def plot_grid(plot_folder, ROI, data_separation, show=False, save=True):
    mode = 'grid'
    n = 5
    file_data_separation = plot_folder + mode + '_' + data_separation
    grid = np.zeros((n, n), dtype=np.float32)
    mean = np.mean(ROI, axis=0)
    cmap = copy.copy(mpl.cm.get_cmap("Blues"))
    cmap.set_bad(color='grey')

    ##assign the grid tiles to the rois
    indices = [6, 8, 8, 8, 4]
    for d in range(n):
        grid[1:n - 1, d] = mean[indices[d]]
    grid[0, 1:n - 1] = mean[5]
    grid[n - 1, 1:n - 1] = mean[7]
    grid[0, 0] = mean[1]
    grid[0, n - 1] = mean[0]
    grid[n - 1, 0] = mean[2]
    grid[n - 1, n - 1] = mean[3]

    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(grid, cmap=cmap, origin='lower', interpolation='none')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save:
        plt.savefig(file_data_separation + '.jpg')
    if show:
        plt.title(mode + ', ' + data_separation)
        plt.show()
    plt.close(fig)
    return


# barplot with mean of z scores of selected units for timebins around transition, error bars: SEM
def plot_transitions(plot_folder, transition_window, data_separation, mode, show=False, save=True):
    file_data_separation = plot_folder + mode + '_' + data_separation

    mean = np.mean(transition_window, axis=0)
    std = np.std(transition_window, axis=0)
    fig = plt.figure(figsize=(25, 5))
    n_of_samples = transition_window.shape[0]
    sem = std / np.sqrt(n_of_samples)  # compute SEM (standard error of the mean)

    toplot = mean
    errorbar = sem
    number_of_bins = transition_window.shape[1]
    timeframe = 10
    plt.bar(np.arange(number_of_bins) + 0.5, toplot, yerr=errorbar, width=1)
    plt.xticks(np.arange(number_of_bins + 1),
               np.arange(-number_of_bins // 2, number_of_bins // 2 + 1) * timeframe / number_of_bins)
    if save:
        plt.savefig(file_data_separation + '.jpg')
    if show:
        plt.title(mode + ', ' + data_separation)
        plt.show()
    plt.close(fig)
    return


# plots mean of selected units regarding the percent difference of ROI(EZM) mean from overall mean of unit
def plot_arms(plot_folder, ROI, data_separation, show=False, save=True):
    mode = 'arms'
    file_data_separation = plot_folder + mode + '_' + data_separation

    mean = np.mean(ROI, axis=0)
    std = np.std(ROI, axis=0)
    n_of_samples = ROI.shape[0]
    sem = std / np.sqrt(n_of_samples)  # compute SEM (stadard error of the mean)
    fig = plt.figure(figsize=(5, 5))
    toplot = mean[[0, 1, 2, 3, 4, 6, 5, 7]]
    errorbar = sem[[0, 1, 2, 3, 4, 6, 5, 7]]  # arrangement: ['top right', 'top left', 'bottom left',
    # 'bottom right', 'top (open)', 'bottom (open)', 'left (closed)', 'right (closed)']
    plt.bar(np.arange(8), toplot, yerr=errorbar, width=1)
    plt.xticks(np.arange(8), [0, 1, 2, 3, 4, 6, 5, 7])

    if save:
        plt.savefig(file_data_separation + '.jpg')
    if show:
        plt.title(mode + ', ' + data_separation)
        plt.show()
    plt.close(fig)
    return


# plots mean of selected units regarding the percent difference of ROI(OF) mean from overall mean of unit
def plot_corners(plot_folder, ROI, data_separation, show=False, save=True):
    mode = 'corners'
    file_data_separation = plot_folder + mode + '_' + data_separation

    mean = np.mean(ROI, axis=0)
    std = np.std(ROI, axis=0)
    n_of_samples = ROI.shape[0]
    sem = std / np.sqrt(n_of_samples)  # compute SEM (stadard error of the mean)

    fig = plt.figure(figsize=(5, 5))
    toplot = mean  # arrangement of the ROIs: {0: 'top right', 1: 'top left', 2: 'bottom left',
    # 3: 'bottom right', 4: 'right', 5: 'top', 6: 'left', 7: 'bottom', 8: 'middle'}
    errorbar = sem
    plt.bar(np.arange(9), toplot, yerr=errorbar, width=1)
    plt.xticks(np.arange(9), np.arange(9))
    if save:
        plt.savefig(file_data_separation + '.jpg')
    if show:
        plt.title(mode + ', ' + data_separation)
        plt.show()
    plt.close(fig)
    return


# plots barplot of the mean phaseplots of all included units, error bars are SEM
def plot_phase(plot_folder, binned, data_separation, mode, show=False, save=True):  # todo binned is already mean??
    binned = binned
    file_data_separation = plot_folder + mode + '_' + data_separation
    unit_mean = np.mean(binned, axis=1)
    normalized = (binned - unit_mean[:, None]) / unit_mean[:, None]
    mean = np.mean(normalized, axis=0)
    std = np.std(normalized.astype('float'), axis=0)
    n_of_samples = normalized.shape[0]
    sem = std / np.sqrt(n_of_samples)  # compute SEM (standard error of the mean)

    fig = plt.figure(figsize=(5, 5))
    toplot = mean
    errorbar = sem
    number_of_bins = mean.shape[0]
    # plt.bar(np.arange(number_of_bins) + 0.5, toplot, yerr=errorbar, width=1)
    # plt.xticks(np.arange(number_of_bins + 1),
    #            np.arange(-number_of_bins // 2, number_of_bins // 2 + 1) * 180 * 2 // number_of_bins)
    plt.polar(np.arange(-number_of_bins // 2, number_of_bins // 2 + 1) * math.pi * 2 / number_of_bins,
              [*toplot, toplot[0]])
    if save:
        plt.savefig(file_data_separation + '.jpg')
    if show:
        plt.title(mode + ', ' + data_separation)
        plt.show()
    plt.close(fig)
    return


# plots barplot of the phase of all included units on all vHIP pads error bars are SEM
def plot_phase_all_pads(plot_folder, all_pads, pad_columns, data_separation, mode, show, save):
    mode = mode + '_all_pads'
    file_data_separation = plot_folder + mode + '_' + data_separation

    fig, axs = plt.subplots(8, 4, sharex=True, sharey=True, subplot_kw=dict(polar=True))
    fig.set_figheight(15)
    fig.set_figwidth(15)
    for pad_name in pad_columns:
        pad_number = int(pad_name[-2:]) - 33
        binned = all_pads.loc[:, pad_name].values
        unit_mean = np.mean(binned, axis=1)
        normalized = (binned - unit_mean[:, None]) / unit_mean[:, None]
        mean = np.mean(normalized, axis=0)
        std = np.std(normalized.astype('float'), axis=0)
        n_of_samples = normalized.shape[0]
        sem = std / np.sqrt(n_of_samples)  # compute SEM (standard error of the mean)

        toplot = mean
        errorbar = sem
        number_of_bins = mean.shape[0]
        # axs[pad_number // 4, pad_number % 4].bar(np.arange(number_of_bins) + 0.5, toplot, yerr=errorbar, width=1)
        # axs[pad_number // 4, pad_number % 4].set_xticks(np.arange(number_of_bins + 1))
        # axs[pad_number // 4, pad_number % 4].set_xticklabels(
        #     np.arange(-number_of_bins // 2, number_of_bins // 2 + 1) * 180 * 2 // number_of_bins)
        axs[pad_number // 4, pad_number % 4].plot(np.arange(-number_of_bins // 2, number_of_bins // 2 + 1) * math.pi * 2
                                                  / number_of_bins, [*toplot, toplot[0]])

        axs[pad_number // 4, pad_number % 4].set_title(pad_number + 33, loc='right')
    if save:
        plt.savefig(file_data_separation + '.jpg')
    if show:
        fig.suptitle(mode + ', ' + data_separation)
        plt.show()
    plt.close(fig)
    return
