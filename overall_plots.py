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

def plot_circle(overall_plots, animal, ROI, name, show=False, save=True):
    mode = 'circle'
    file_name = overall_plots + animal + '_' + mode + '_' + name
    colorcoding = np.empty(20)

    colorcoding[2] = ROI[0]
    colorcoding[7] = ROI[1]
    colorcoding[12] = ROI[2]
    colorcoding[17] = ROI[3]
    colorcoding[3:7] = ROI[4]
    colorcoding[8:12] = ROI[5]
    colorcoding[13:17] = ROI[6]
    colorcoding[18:] = ROI[7]
    colorcoding[:2] = ROI[7]



    colors = cm.jet(plt.Normalize()(colorcoding))
    fig = plt.figure(figsize=(5, 5))
    fig.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=ROI.min(), vmax=ROI.max()), cmap='jet'),
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


def plot_grid(overall_plots, animal, ROI, name, show=False, save=True):
    mode = 'grid'
    n = 5
    file_name = overall_plots + animal + '_' + mode + '_' + name
    grid = np.zeros((n, n), dtype=np.float32)

    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_bad(color='grey')
    indices = [6,8,8,8,4]
    for d in range(n):
        grid[1:n - 1, d] = ROI[indices[d]]
    grid[0, 1:n - 1] = ROI[5]
    grid[n, 1:n - 1] = ROI[7]
    grid[0,0] = ROI[1]
    grid[0,n] = ROI[0]
    grid[n,0] = ROI[2]
    grid[n,n] = ROI[3]

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


def plot_transitions(overall_plots, animal, transition_window, name, mode, show=False, save=True):
    plotmode = 'std'
    file_name = overall_plots + animal + '_' + mode + '_' + name
    n = 500
    x = np.arange(-n, n + 1)

    sum_mean = np.mean(transition_window, axis=0)
    sum_std = np.std(transition_window, axis=0)
    fig = plt.figure(figsize=(5, 5))

    if plotmode == 'std':
        plt.bar(x, sum_mean, yerr=sum_std, width=25)
    elif plotmode == 'percent':
        mean = np.mean(sum_mean)
        plt.bar(x, (sum_mean - mean) * 100 / mean, width=25)
    if save:
        plt.savefig(file_name + '.jpg')
    plt.title('transition ' + mode + ' ' + name)
    if show:
        plt.show()
    plt.close(fig)
    return


def plot_arms(overall_plots, animal, ROI, name, show=False, save=True):
    mode = 'arms'
    file_name = overall_plots + animal + '_' + mode + '_' + name

    ROImean = np.mean(ROI, axis=0)
    ROIstd = np.std(ROI, axis=0)

    fig = plt.figure(figsize=(5, 5))
    toplot = ROImean[[0, 1, 2, 3, 4, 6, 5, 7]]
    errorbar = ROIstd[[0, 1, 2, 3, 4, 6, 5, 7]]
    plt.bar(np.arange(8), toplot, yerr=errorbar, width=1)
    if save:
        plt.savefig(file_name + '.jpg')
    if show:
        plt.title(file_name)
        plt.show()
    plt.close(fig)
    return


def plot_corners(overall_plots, animal, ROI, name, show=False, save=True):
    mode = 'corners'
    file_name = overall_plots + animal + '_' + mode + '_' + name

    ROImean = np.mean(ROI, axis=0)
    ROIstd = np.std(ROI, axis=0)

    fig = plt.figure(figsize=(5, 5))
    toplot = ROImean
    errorbar = ROIstd
    plt.bar(np.arange(8), toplot, yerr=errorbar, width=1)
    if save:
        plt.savefig(file_name + '.jpg')
    if show:
        plt.title(file_name)
        plt.show()
    plt.close(fig)
    return


def plot_phase(overall_plots, animal, ROI, name, mode, show=False, save=True):
    file_name = overall_plots + animal + '_' + mode + '_' + name
    normalized = ROI / np.mean(ROI, axis=1)[:, None]
    ROImean = np.mean(normalized, axis=0)
    ROIstd = np.std(ROI, axis=0)

    fig = plt.figure(figsize=(5, 5))
    toplot = ROImean
    errorbar = ROIstd
    plt.bar(np.arange(ROImean.shape[1]), toplot, yerr=errorbar, width=1)
    if save:
        plt.savefig(file_name + '.jpg')
    if show:
        plt.title(file_name)
        plt.show()
    plt.close(fig)
    return