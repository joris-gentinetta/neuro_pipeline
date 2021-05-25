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

def plot_circle(overallplots, animal, ROI, show=False, save=True):
    mode = 'circle'
    file_name = overallplots + animal + '_' + mode
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
        plt.title('ROI' + animal )
        plt.show()
    plt.close(fig)
    return


def plot_grid(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger, cluster_names, minp=0,
              maxp=100, n=5, show=False, save=True):
    movement = events['movement']
    mode = 'grid'
    file_name = plot_folder + experiment_name + '_' + mode + '_n' + str(n) + '_minp' + str(minp) + '_maxp' + str(
        maxp) + '_'

    data = raw_data
    grid = np.zeros((n, n, data.shape[0]), dtype=np.float32)

    xyv = np.empty(
        (3, min(len(movement['calib_traj_x'].index) - video_trigger - off, data.shape[1] - physio_trigger - off)),
        dtype=np.float32)
    xyv[0] = movement['calib_traj_x'][video_trigger + off: xyv.shape[1] + video_trigger + off]
    xyv[1] = movement['calib_traj_y'][video_trigger + off: xyv.shape[1] + video_trigger + off]
    sx = sy = 350
    xyv[0] += 6
    xyv[1] += 6

    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_bad(color='grey')
    for i in range(data.shape[0]):
        xyv[2] = data[i][physio_trigger + off: xyv.shape[1] + physio_trigger + off] + make_path_visible

        for x in range(n):
            for y in range(n):
                boolean = np.logical_and(np.logical_and(xyv[0] > x / n * sx, xyv[0] < (x + 1) / n * sx),
                                         np.logical_and(xyv[1] > y / n * sy, xyv[1] < (y + 1) / n * sy))
                bsum = np.sum(boolean)
                if bsum != 0:
                    grid[x, y, i] = np.sum(xyv[2][boolean]) / bsum

        for d in range(n):
            grid[1:n - 1, d, i] = np.mean(grid[1:n - 1, d, i])

        for d in range(n):
            grid[d, 1:n - 1, i] = np.mean(grid[d, 1:n - 1, i])
        vmin = np.percentile(grid[:, :, i][np.where(grid[:, :, i] > 0)], minp)
        vmax = np.percentile(grid[:, :, i][np.where(grid[:, :, i] > 0)], maxp)

        fig = plt.figure(figsize=(5, 5))
        im = plt.imshow(np.ma.masked_where(grid[:, :, i] == 0, grid[:, :, i]).T, cmap=cmap, origin='upper',
                        interpolation='none', vmin=vmin, vmax=vmax)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        if save:
            plt.savefig(file_name + str(cluster_names[i]) + '.jpg')

        if i != 0:
            plt.title('firing rate unit ' + str(cluster_names[i]))
        else:
            plt.title('firing rate all mua')
        if show:
            plt.show()
        plt.close(fig)

    ROI = np.sum(grid[:, :, 1:], axis=2)
    vmin = np.percentile(ROI[np.where(ROI > 0)], minp)
    vmax = np.percentile(ROI[np.where(ROI > 0)], maxp)

    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(np.ma.masked_where(ROI == 0, ROI).T, cmap=cmap, origin='upper', interpolation='none',
                    vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save:
        plt.savefig(file_name + 'all_units' + '.jpg')
    plt.title('firing rate all units')
    if show:
        plt.show()
    plt.close(fig)
    return


def plot_transitions(plot_folder, experiment_name, raw_data, events, cluster_names, video_trigger, archive, mode,
                     plotmode='percent', n=500, m=5, show=False, save=True, do_archive=True):
    file_name = plot_folder + experiment_name + '_' + mode + '_n' + str(n) + '_'

    data = raw_data
    x = np.arange(-n, n + 1, (2 * n + 1) // (m - 1))
    transitions = events['transitions']
    transition_indices = [transition_index + video_trigger for transition_index in transitions[mode] if
                          transition_index + video_trigger + n + 1 <= data.shape[
                              1] and transition_index + video_trigger - n >= 0]
    grid = [np.zeros((len(transition_indices), m), dtype=np.float32) for _ in range(data.shape[0])]

    for tindex, transition_index in enumerate(transition_indices):
        for unit in range(data.shape[0]):
            for timeindex in range(m):
                grid[unit][tindex, timeindex] = np.mean(data[unit, transition_index - n:transition_index + n + 1][
                                                        (2 * n + 1) * timeindex // n: (2 * n + 1) * (
                                                                timeindex + 1) // n]) - np.mean(data[unit])

    mean_std = [np.zeros((2, m)) for _ in range(data.shape[0])]
    for unit in range(data.shape[0]):
        mean_std[unit][0] = np.mean(grid[unit], axis=0)
        mean_std[unit][1] = np.std(grid[unit], axis=0)

    if do_archive:
        archive_transition_indices = [transition_index + video_trigger for transition_index in transitions[mode] if
                              transition_index + video_trigger + 501 <= data.shape[
                                  1] and transition_index + video_trigger - 500 >= 0]
        tobearchived = np.zeros((data.shape[0], 1001, len(archive_transition_indices)))
        for tindex, transition_index in enumerate(archive_transition_indices):
            for unit in range(data.shape[0]):
                tobearchived[unit,:, tindex] = data[unit, transition_index - 500:transition_index + 501]
        tobearchived = np.mean(tobearchived, axis=2) - np.mean(data, axis=1)[:,None]
        archive.loc[:,  idx[mode,:]] = tobearchived
    if show or save:
        # for i in range(grid.shape[0]):
        #    grid[i] = gaussian_filter1d(grid[i], sigma=sigma)
        for i in range(data.shape[0]):
            fig = plt.figure(figsize=(5, 5))
            if plotmode == 'std':
                plt.bar(x, mean_std[i][0], yerr=mean_std[i][1], width=25)
            elif plotmode == 'percent':
                plt.bar(x, ((np.mean(data[i]) != 0) * mean_std[i][0] / np.mean(data[i]) + 0) * 100, width=25)
            if save:
                plt.savefig(file_name + str(cluster_names[i]) + '.jpg')

            if i != 0:
                plt.title('firing rate unit ' + str(cluster_names[i]))
            else:
                plt.title('firing rate all mua')
            if show:
                plt.show()
            plt.close(fig)

        ROI = sum(grid[1:])
        sum_mean = np.mean(ROI, axis=0)
        sum_std = np.std(ROI, axis=0)
        if plotmode == 'std':
            plt.bar(x, sum_mean, yerr=sum_std, width=25)
        elif plotmode == 'percent':
            plt.bar(x, ((np.mean(data) != 0) * sum_mean / np.mean(data) + 0) * 100, width=25)
        if save:
            plt.savefig(file_name + 'all_units' + '.jpg')
        plt.title('firing rate all units')
        if show:
            plt.show()
        plt.close(fig)
    return archive


def plot_arms(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
              cluster_names, archive, transition_size=2, minp=0, maxp=90, n=150, show=False, save=True, do_archive=True):
    movement = events['movement']
    mode = 'arms'
    file_name = plot_folder + experiment_name + '_' + mode + '_n' + str(n) + '_minp' + str(minp) + '_maxp' + str(
        maxp) + '_'

    data = raw_data
    grid = np.zeros((n, n, data.shape[0]), dtype=np.float32)
    xyv = np.empty(
        (3, min(len(movement['calib_traj_x'].index) - video_trigger - off, data.shape[1] - physio_trigger - off)),
        dtype=np.float32)
    xyv[0] = movement['calib_traj_x'][video_trigger + off: xyv.shape[1] + video_trigger + off]
    xyv[1] = movement['calib_traj_y'][video_trigger + off: xyv.shape[1] + video_trigger + off]

    xyv[1] -= 5
    sx = 400
    sy = 400
    masks = [np.zeros((n, n), dtype=np.bool) for _ in range(8)]
    ROI = np.zeros((data.shape[0], 8))
    for x in range(n):
        for y in range(n):
            angle = np.arctan2((y - n / 2), x - n / 2)
            angle += (angle < 0) * 2 * math.pi
            for quadrant in range(4):
                target = 40 + quadrant * 90
                degree = angle * 360 // (2 * math.pi)
                if degree in range(target - transition_size, target + transition_size):
                    masks[quadrant][x, n - y - 1] = 1
                if degree in np.arange(target + transition_size, target + 90 - transition_size) % 360:
                    masks[quadrant + 4][x, n - y - 1] = 1

    # for i in range(8):
    #     f = plt.figure()
    #     plt.imshow(masks[i].T)
    #     plt.show()
    #     plt.close(f)

    for i in range(data.shape[0]):
        xyv[2] = data[i][physio_trigger + off: xyv.shape[1] + physio_trigger + off] + make_path_visible

        for x in range(n):
            for y in range(n):
                boolean = np.logical_and(np.logical_and(xyv[0] > x / n * sx, xyv[0] < (x + 1) / n * sx),
                                         np.logical_and(xyv[1] > y / n * sy, xyv[1] < (y + 1) / n * sy))
                bsum = np.sum(boolean)
                if bsum != 0:
                    grid[x, y, i] = np.sum(xyv[2][boolean]) / bsum

        for quadrant in range(8):
            valid_values_in_quadrant = grid[:, :, i][np.logical_and(masks[quadrant] == 1, grid[:, :, i] != 0)]
            # fig=plt.figure()
            # plt.imshow(grid[:,:,i].T)
            # plt.imshow(masks[quadrant].T, alpha=0.5)
            # plt.show()
            # plt.close(fig)
            ROI[i, quadrant] = (valid_values_in_quadrant.mean() - grid[:, :, i][grid[:, :, i] != 0].mean()) / \
                               grid[:, :, i][grid[:, :, i] != 0].mean() * 100

        if save or show:
            fig = plt.figure(figsize=(5, 5))
            toplot = ROI[i][[0, 1, 2, 3, 4, 6, 5, 7]]
            plt.bar(np.arange(8), toplot)
            if save:
                plt.savefig(file_name + str(cluster_names[i]) + '.jpg')
            if i != 0:
                plt.title('firing rate unit ' + str(cluster_names[i]))
            else:
                plt.title('firing rate all mua')
            if show:
                plt.show()
            plt.close(fig)
    if do_archive:
        archive.loc[:, idx['ROI_EZM', :]] = ROI


    if save or show:
        ROImean = np.mean(ROI[1:], axis=0)
        fig = plt.figure(figsize=(5, 5))
        toplot = ROImean[[0, 1, 2, 3, 4, 6, 5, 7]]
        plt.bar(np.arange(8), toplot, width=1)
        if save:
            plt.savefig(file_name + 'all_units' + '.jpg')
        plt.title('firing rate all units')
        if show:
            plt.show()
        plt.close(fig)
    return archive


def plot_corners(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger, cluster_names, archive,
                 n=5, show=False, save=True, do_archive=True):
    movement = events['movement']
    mode = 'corners'
    file_name = plot_folder + experiment_name + '_' + mode + '_n' + str(n) + '_'

    data = raw_data
    grid = np.zeros((n, n, data.shape[0]), dtype=np.float32)

    xyv = np.empty(
        (3, min(len(movement['calib_traj_x'].index) - video_trigger - off, data.shape[1] - physio_trigger - off)),
        dtype=np.float32)
    xyv[0] = movement['calib_traj_x'][video_trigger + off: xyv.shape[1] + video_trigger + off]
    xyv[1] = movement['calib_traj_y'][video_trigger + off: xyv.shape[1] + video_trigger + off]
    sx = sy = 350
    xyv[0] += 6
    xyv[1] += 6
    ROI = np.zeros((data.shape[0], 9))

    for i in range(data.shape[0]):
        xyv[2] = data[i][physio_trigger + off: xyv.shape[1] + physio_trigger + off] + make_path_visible

        for x in range(n):
            for y in range(n):
                boolean = np.logical_and(np.logical_and(xyv[0] > x / n * sx, xyv[0] < (x + 1) / n * sx),
                                         np.logical_and(xyv[1] > y / n * sy, xyv[1] < (y + 1) / n * sy))
                bsum = np.sum(boolean)
                if bsum != 0:
                    grid[x, y, i] = np.sum(xyv[2][boolean]) / bsum

        for d in range(n):
            grid[1:n - 1, d, i] = np.mean(grid[1:n - 1, d, i])

        for d in range(n):
            grid[d, 1:n - 1, i] = np.mean(grid[d, 1:n - 1, i])
        takenfrom = [(n - 1, 0), (0, 0), (0, n - 1), (n - 1, n - 1), (n - 1, 1), (1, 0), (0, 1), (1, n - 1), (1, 1)]
        for index in range(9):
            ROI[i, index] = (grid[:, :, i][takenfrom[index]] - grid[:,:,i].mean()) * 100 / grid[:,:,i].mean()

        if save or show:

            fig = plt.figure(figsize=(5, 5))
            plt.bar(np.arange(9), ROI[i])
            if save:
                plt.savefig(file_name + str(cluster_names[i]) + '.jpg')

            if i != 0:
                plt.title('firing rate unit ' + str(cluster_names[i]))
            else:
                plt.title('firing rate all mua')
            if show:
                plt.show()
            plt.close(fig)
    if do_archive:
        archive.loc[:, idx['ROI_OF', :]] = ROI
    if save or show:
        ROI = (np.mean(ROI[1:], axis=0) - grid.mean()) * 100 / grid.mean()
        fig = plt.figure(figsize=(5, 5))
        plt.bar(np.arange(9), ROI)
        if save:
            plt.savefig(file_name + 'all_units' + '.jpg')
        plt.title('firing rate all units')
        if show:
            plt.show()
        plt.close(fig)
    return archive

def get_ezm_score(rois):
    mean = np.mean(rois, axis=1)
    rois = (rois - mean[:,None]) / mean[:,None]
    a1 = 0.25 * (np.abs(rois[:,5]-rois[:,4]) + np.abs(rois[:,5]-rois[:,6]) + np.abs(rois[:,7]-rois[:,7]) + np.abs(rois[:,7]-rois[:,6]))
    b1 = 0.5 * (np.abs(rois[:,5]-rois[:,7]) + np.abs(rois[:,4]-rois[:,6]))
    open_close = (a1-b1)/(a1+b1)

    a2 = 1/16 * (np.abs(rois[:,0]-rois[:,4]) + np.abs(rois[:,0]-rois[:,7])
                 + np.abs(rois[:,0]-rois[:,6]) + np.abs(rois[:,0]-rois[:,5])
                 + np.abs(rois[:,1]-rois[:,4]) + np.abs(rois[:,1]-rois[:,7])
                 + np.abs(rois[:,1]-rois[:,6]) + np.abs(rois[:,1]-rois[:,5])
                 + np.abs(rois[:,2]-rois[:,4]) + np.abs(rois[:,2]-rois[:,7])
                 + np.abs(rois[:,2]-rois[:,6]) + np.abs(rois[:,2]-rois[:,5])
                 + np.abs(rois[:,3]-rois[:,4]) + np.abs(rois[:,3]-rois[:,7])
                 + np.abs(rois[:,3]-rois[:,6]) + np.abs(rois[:,3]-rois[:,5]))
    b2 = 1/12 * (np.abs(rois[:,0]-rois[:,1]) + np.abs(rois[:,1]-rois[:,3])
                 + np.abs(rois[:,1]-rois[:,2]) + np.abs(rois[:,0]-rois[:,3])
                 + np.abs(rois[:,0]-rois[:,2]) + np.abs(rois[:,2]-rois[:,3])
                 + np.abs(rois[:,4]-rois[:,7]) + np.abs(rois[:,4]-rois[:,6])
                 + np.abs(rois[:,4]-rois[:,5]) + np.abs(rois[:,5]-rois[:,7])
                 + np.abs(rois[:,5]-rois[:,6]) + np.abs(rois[:,6]-rois[:,7]))
    crossing = (a2-b2)/(a2+b2)
    closed = (rois[:, 5] + rois[:, 7])/2
    transition = (rois[:, 0]+ rois[:, 1] + rois[:, 2]+ rois[:, 3])/4
    return open_close, crossing, closed, transition


def get_of_score(rois):
    mean = np.mean(rois, axis=1)
    rois = (rois - mean[:, None]) / mean[:, None]

    a1 = 1 / 20 * (np.abs(rois[:, 0] - rois[:, 4]) + np.abs(rois[:, 0] - rois[:, 5])
                   + np.abs(rois[:, 0] - rois[:, 6]) + np.abs(rois[:, 0] - rois[:, 7])
                   + np.abs(rois[:, 0] - rois[:, 8]) + np.abs(rois[:, 1] - rois[:, 4])
                   + np.abs(rois[:, 1] - rois[:, 5]) + np.abs(rois[:, 1] - rois[:, 6])
                   + np.abs(rois[:, 1] - rois[:, 7]) + np.abs(rois[:, 1] - rois[:, 8])
                   + np.abs(rois[:, 2] - rois[:, 4]) + np.abs(rois[:, 2] - rois[:, 5])
                   + np.abs(rois[:, 2] - rois[:, 6]) + np.abs(rois[:, 2] - rois[:, 7])
                   + np.abs(rois[:, 2] - rois[:, 8]) + np.abs(rois[:, 3] - rois[:, 4])
                   + np.abs(rois[:, 3] - rois[:, 5]) + np.abs(rois[:, 3] - rois[:, 6])
                   + np.abs(rois[:, 3] - rois[:, 7]) + np.abs(rois[:, 3] - rois[:, 8]))

    b1 = 1 / 15 * (np.abs(rois[:, 0] - rois[:, 1]) + np.abs(rois[:, 1] - rois[:, 3])
                   + np.abs(rois[:, 1] - rois[:, 2]) + np.abs(rois[:, 0] - rois[:, 3])
                   + np.abs(rois[:, 0] - rois[:, 2]) + np.abs(rois[:, 2] - rois[:, 3])
                   + np.abs(rois[:, 4] - rois[:, 7]) + np.abs(rois[:, 4] - rois[:, 6])
                   + np.abs(rois[:, 4] - rois[:, 5]) + np.abs(rois[:, 5] - rois[:, 7])
                   + np.abs(rois[:, 5] - rois[:, 6]) + np.abs(rois[:, 6] - rois[:, 7])
                   + np.abs(rois[:, 4] - rois[:, 8]) + np.abs(rois[:, 5] - rois[:, 8])
                   + np.abs(rois[:, 6] - rois[:, 8]))

    of_corners_score = (a1 - b1) / (a1 + b1)
    of_corners = (rois[:, 0] + rois[:, 1] + rois[:, 2] + rois[:, 3]) / 4
    a2 = 1 / 8 * (np.abs(rois[:, 8] - rois[:, 4]) + np.abs(rois[:, 8] - rois[:, 5])
                   + np.abs(rois[:, 8] - rois[:, 6]) + np.abs(rois[:, 8] - rois[:, 7])
                   + np.abs(rois[:, 0] - rois[:, 8]) + np.abs(rois[:, 1] - rois[:, 8])
                   + np.abs(rois[:, 2] - rois[:, 8]) + np.abs(rois[:, 3] - rois[:, 8]))


    b2 = 1 / 28 * (np.abs(rois[:, 0] - rois[:, 1]) + np.abs(rois[:, 0] - rois[:, 2])
                   + np.abs(rois[:, 0] - rois[:, 3]) + np.abs(rois[:, 0] - rois[:, 4])
                   + np.abs(rois[:, 0] - rois[:, 5]) + np.abs(rois[:, 0] - rois[:, 6])
                   + np.abs(rois[:, 0] - rois[:, 7]) + np.abs(rois[:, 1] - rois[:, 2])
                   + np.abs(rois[:, 1] - rois[:, 3]) + np.abs(rois[:, 1] - rois[:, 4])
                   + np.abs(rois[:, 1] - rois[:, 5]) + np.abs(rois[:, 1] - rois[:, 6])
                   + np.abs(rois[:, 1] - rois[:, 7]) + np.abs(rois[:, 2] - rois[:, 3])
                   + np.abs(rois[:, 2] - rois[:, 4]) + np.abs(rois[:, 2] - rois[:, 5])
                   + np.abs(rois[:, 2] - rois[:, 6]) + np.abs(rois[:, 2] - rois[:, 7])
                   + np.abs(rois[:, 3] - rois[:, 4]) + np.abs(rois[:, 3] - rois[:, 5])
                   + np.abs(rois[:, 3] - rois[:, 6]) + np.abs(rois[:, 3] - rois[:, 7])
                   + np.abs(rois[:, 4] - rois[:, 5]) + np.abs(rois[:, 4] - rois[:, 6])
                   + np.abs(rois[:, 4] - rois[:, 7]) + np.abs(rois[:, 5] - rois[:, 6])
                   + np.abs(rois[:, 5] - rois[:, 7]) + np.abs(rois[:, 6] - rois[:, 7]))


    of_middle_score = (a2 - b2) / (a2 + b2)
    of_middle = rois[:, 8]
    return of_corners_score, of_middle_score, of_corners, of_middle

def plot_phase(circus, plot_folder, experiment_name, off, physio_trigger, cluster_names, archive, environment,
          show, save, do_archive):
    offset = (physio_trigger + off) * 20000 // 50
    mode = 'phase'
    key = 'theta_phase_' + environment
    file_name = plot_folder + experiment_name + '_' + mode + '_'
    number_of_bins = 8
    factor = 360//number_of_bins
    phase = np.load(circus + 'phase_files/' + experiment_name + '.npy')[:,offset:]//factor
    original_data = np.load(circus + 'original_' + experiment_name + '.npy')[:,offset:]
    print(original_data[0])
    masked = np.ma.masked_array(phase, mask = np.invert(original_data))
    phase_distribution = np.zeros((original_data.shape[0], number_of_bins))
    for angle in range(-180//factor, 180//factor):
        phase_distribution[:, angle+180//factor] = np.sum(masked == angle, axis=1)
    mean = phase_distribution.mean(axis=1)
    phase_distribution -= mean[:, None]
    phase_distribution = phase_distribution * 100 / mean[:, None]
    if do_archive:
        archive.loc[:, idx[key, :]] = phase_distribution
    if save or show:
        for unit in range(phase_distribution.shape[0]):
            fig = plt.figure(figsize=(5, 5))
            plt.bar(np.arange(-180//factor, 180//factor), phase_distribution[unit], width=1)
            if save:
                plt.savefig(file_name + str(cluster_names[unit]) + '.jpg')
            if show:
                if unit != 0:
                    plt.title('firing rate unit ' + str(cluster_names[unit]))
                else:
                    plt.title('firing rate all mua')
                plt.show()
            plt.close(fig)

        fig = plt.figure(figsize=(5, 5))
        plt.bar(np.arange(-180 // factor, 180 // factor), phase_distribution.mean(axis=0), width=1)
        if save:
            plt.savefig(file_name + 'all_units' + '.jpg')
        plt.title('firing rate all units')
        if show:
            plt.show()
        plt.close(fig)
    return archive