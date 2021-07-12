import copy
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.filters import gaussian_filter1d

idx = pd.IndexSlice


# creates the marker cross to show the transitions in classic and raw plots
# 2D mask, 1 in the transition zones, 0 in the rest
def get_ezm_mask(n):
    mask = np.zeros((n, n), dtype=np.float32)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            angle = np.arctan2((y - n / 2), x - n / 2)
            angle += (angle < 0) * 2 * math.pi
            for quadrant in range(4):
                target = 40 + quadrant * 90
                degree = angle * 360 // (2 * math.pi)
                if degree in range(target - 1, target + 1):
                    mask[x, n - y - 1] = 1
    return mask


# creates a list of the masks for the different ROIs, for the indices of the ROIs in the list, see the thesis
# n is the size of the 2D grid (nxn), transition areas are defined as open-closed borders +-transition_size
def get_ezm_ROI_masks(n, transition_size):
    masks = [np.zeros((n, n), dtype=np.bool) for _ in range(8)]  # create mask for every ROI
    for x in range(n):
        for y in range(n):
            angle = np.arctan2((y - n / 2), x - n / 2)  # get angle in range -pi, pi
            angle += (angle < 0) * 2 * math.pi  # convert to range 0, 2pi
            for quadrant in range(4):
                target = 40 + quadrant * 90  # the EZM is tilted by 40 degrees
                degree = angle * 360 // (2 * math.pi)  # convert radian to degree
                if degree in range(target - transition_size, target + transition_size):  # create transition ROIs
                    masks[quadrant][x, n - y - 1] = 1  # n - y - 1 is taking into account that y axis goes downward
                if degree in np.arange(target + transition_size,
                                       target + 90 - transition_size) % 360:  # create open and closed ROIs
                    masks[quadrant + 4][x, n - y - 1] = 1
    return masks


# plots 2d firing rate, temporally filtered
def plot_trace(environment, plot_folder, experiment_name, aligned, cluster_names, single_figures, multi_figure,
               sigma=10, minp=0, maxp=95, n=150, show=False, save=True, filter=False):
    ##make file name:
    mode = 'trace_filter_' + str(filter)
    file_name = plot_folder + experiment_name + '_' + mode
    if filter:
        file_name += '_sigma' + str(sigma)
    file_name += '_n' + str(n) + '_minp' + str(
        minp) + '_maxp' + str(maxp) + '_'

    content = np.copy(aligned)  # do not modify aligned directly
    if filter:
        content[2:] = gaussian_filter1d(content[2:], sigma=sigma, axis=1)  # filter temporally (gaussian)

    number_of_units = content.shape[0] - 2
    grid = np.zeros((n, n, number_of_units),
                    dtype=np.float32)  # 2d grid containing firing rate at coordinates given by row(y) and column(x)

    ##center the apparatus:
    if environment == 'EZM':
        sx = sy = 400  # max x/y coordinate
        content[1] -= 5  # shift upwards
        mask = get_ezm_mask(n)
    elif environment == 'OFT':
        sx = sy = 350
        content[0] += 6  # shift downward
        content[1] += 6  # shift right

    cmap = copy.copy(mpl.cm.get_cmap('Blues'))
    cmap.set_bad(color='grey')

    ##create the 2D grid:
    for unit in range(number_of_units):
        for x in range(n):
            for y in range(n):
                boolean = np.logical_and(np.logical_and(content[0] > x / n * sx, content[0] < (x + 1) / n * sx),
                                         np.logical_and(content[1] > y / n * sy, content[1] < (y + 1) / n * sy))
                bsum = np.sum(boolean)
                if bsum != 0:
                    grid[x, y, unit] = np.sum(content[unit + 2][boolean]) / bsum

        ##plot individual figures for all units
        if single_figures:
            vmin = np.percentile(grid[:, :, unit][np.where(grid[:, :, unit] > 0)],
                                 minp)  # minp percent of the values are below vmin
            vmax = np.percentile(grid[:, :, unit][np.where(grid[:, :, unit] > 0)], maxp)
            fig = plt.figure(figsize=(5, 5))
            im = plt.imshow(np.ma.masked_where(grid[:, :, unit] == 0, grid[:, :, unit]).T, cmap=cmap, origin='upper',
                            interpolation='none', vmin=vmin, vmax=vmax)
            if environment == 'EZM':
                plt.imshow(mask.T, cmap='Greys', origin='upper',
                           interpolation='none', alpha=0.1)

            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.axis('off')
            if save:
                plt.savefig(file_name + str(cluster_names[unit]) + '.jpg')
            if unit != 0:
                plt.title('firing rate unit ' + str(cluster_names[unit]))
            else:
                plt.title('firing rate all mua')
            if show:
                plt.show()
            plt.close(fig)

    ##plot all units on one figure:
    if multi_figure:
        fig, axs = plt.subplots(number_of_units // 4 + 1, 4)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        for unit in range(number_of_units):
            vmin = np.percentile(grid[:, :, unit][np.where(grid[:, :, unit] > 0)],
                                 minp)  # minp percent of the values are below vmin
            vmax = np.percentile(grid[:, :, unit][np.where(grid[:, :, unit] > 0)], maxp)

            im = axs[unit // 4, unit % 4].imshow(np.ma.masked_where(grid[:, :, unit] == 0, grid[:, :, unit]).T,
                                                 cmap=cmap, origin='upper',
                                                 interpolation='none', vmin=vmin, vmax=vmax)
            fig.colorbar(im, fraction=0.046, pad=0.04, ax=axs[unit // 4, unit % 4])
            axs[unit // 4, unit % 4].axis('off')

            axs[unit // 4, unit % 4].set_title(cluster_names[unit], loc='right')
        axes_to_delete = axs[(number_of_units - 1) // 4, (number_of_units - 1) % 4 + 1:]
        for ax_to_delete in axes_to_delete:
            fig.delaxes(ax_to_delete)
        if save:
            plt.savefig(file_name + 'multiple' + '.jpg')
        if show:
            fig.suptitle(file_name)
            plt.show()
        plt.close(fig)

    unit_mean = np.mean(grid[:, :, 1:], axis=2)
    vmin = np.percentile(unit_mean[np.where(unit_mean > 0)], minp)
    vmax = np.percentile(unit_mean[np.where(unit_mean > 0)], maxp)

    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(np.ma.masked_where(unit_mean == 0, unit_mean).T, cmap=cmap, origin='upper', interpolation='none',
                    vmin=vmin, vmax=vmax)
    if environment == 'EZM':
        plt.imshow(mask.T, cmap='Greys', origin='upper',
                   interpolation='none', alpha=0.1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    if save:
        plt.savefig(file_name + 'all_units' + '.jpg')

    if show:
        plt.title('firing rate all units')
        plt.show()
    plt.close(fig)
    return


# plots spatially filtered circle (abstraction of the EZM) for every unit
def plot_circle(plot_folder, experiment_name, aligned, cluster_names, single_figures, multi_figure,
                n=360, sigma=-1, show=False, save=True):
    mode = 'circle'
    if sigma < 0:  # set default sigma such that +-2 sigma equals 1/8 of the circle
        sigma = n / 8 / 4
    file_name = plot_folder + experiment_name + '_' + mode + '_sigma' + str(sigma) + '_n' + str(n) + '_'
    content = np.copy(aligned)
    number_of_units = content.shape[0] - 2
    grid = np.zeros((number_of_units, n),
                    dtype=np.float32)  # rows: unit, columns: angle, angle=0 is on positive x axis

    content[1] -= 5  # shift upward
    sx = 400  # max x coordinate
    sy = 400
    middle_x = sx / 2
    middle_y = sy / 2
    content[0] -= middle_x  # shift center to zero
    content[1] -= middle_y
    content[1] *= -1  # flip y axis (x=0,y=0 was originally in top left corner)
    angle = np.arctan2(content[1], content[0])  # get angle in range -pi, -pi
    angle += (angle < 0) * 2 * math.pi  # shift range to 0, 2pi

    ## assign mean firingrate to every angle:
    for r in range(n):
        boolean = np.logical_and(angle > math.pi * 2 * r / n, angle < math.pi * 2 * (r + 1) / n)
        mask = np.tile(np.invert(boolean), (number_of_units, 1))
        masked = np.ma.masked_array(content[2:], mask=mask)
        grid[:, r] = np.mean(masked, axis=1)

    grid = gaussian_filter1d(grid, sigma=sigma, mode='wrap', axis=1)  # gaussian filter, wraps around at n, 0

    # plot individual figures for all units:
    if single_figures:
        for unit in range(number_of_units):
            fig = plt.figure(figsize=(5, 5))
            colors = cm.jet(plt.Normalize()(grid[unit]))
            for quadrant in range(4):
                colors[(40 + quadrant * 90) * n // 360] = [0, 0, 0, 1]  # mark transitons
            plt.pie(np.ones(n), colors=colors)
            plt.colorbar(
                cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=grid[unit].min(), vmax=grid[unit].max()), cmap='jet'),
                shrink=0.6)
            my_circle = plt.Circle((0, 0), 0.8, color='white')
            fig.gca().add_artist(my_circle)  # make the inside of the pieplot white
            if save:
                plt.savefig(file_name + str(cluster_names[unit]) + '.jpg')
            if unit != 0:
                plt.title('firing rate unit ' + str(cluster_names[unit]))
            else:
                plt.title('firing rate all mua')
            if show:
                plt.show()
            plt.close(fig)

    ##plot one figure with all units:
    if multi_figure:
        fig, axs = plt.subplots(number_of_units // 4 + 1, 4)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        for unit in range(number_of_units):
            colors = cm.jet(plt.Normalize()(grid[unit]))
            for quadrant in range(4):
                colors[(40 + quadrant * 90) * n // 360] = [0, 0, 0, 1]  # mark transitons
            axs[unit // 4, unit % 4].pie(np.ones(n), colors=colors)
            divider = make_axes_locatable(axs[unit // 4, unit % 4])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(
                cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=grid[unit].min(), vmax=grid[unit].max()), cmap='jet'),
                cax=cax, orientation='vertical',
                shrink=0.8)
            my_circle = plt.Circle((0, 0), 0.8, color='white')

            axs[unit // 4, unit % 4].add_artist(my_circle)  # make the inside of the pieplot white
            axs[unit // 4, unit % 4].axis('off')
            axs[unit // 4, unit % 4].set_title(cluster_names[unit], loc='right')

        axes_to_delete = axs[(number_of_units - 1) // 4, (number_of_units - 1) % 4 + 1:]
        for ax_to_delete in axes_to_delete:
            fig.delaxes(ax_to_delete)
        if save:
            plt.savefig(file_name + 'multiple' + '.jpg')
        if show:
            fig.suptitle(file_name)
            plt.show()
        plt.close(fig)

    ## plot mean of all units:
    unit_mean = np.mean(grid[1:] - grid[1:].mean(axis=1)[:, None], axis=0)
    colors = cm.jet(plt.Normalize()(unit_mean))
    for quadrant in range(4):
        colors[(40 + quadrant * 90) * n // 360] = [0, 0, 0, 1]
    fig = plt.figure(figsize=(5, 5))
    fig.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=unit_mean.min(), vmax=unit_mean.max()), cmap='jet'),
                 shrink=0.6)
    plt.pie(np.ones(n), colors=colors)
    my_circle = plt.Circle((0, 0), 0.8, color='white')
    fig.gca().add_artist(my_circle)
    if save:
        plt.savefig(file_name + 'all_units' + '.jpg')
    if show:
        plt.title('firing rate all units')
        plt.show()
    plt.close(fig)
    return


# plots 2d image with mean firing rate for every roi
def plot_grid(plot_folder, experiment_name, aligned, cluster_names, single_figures, multi_figure, minp=0,
              maxp=100, n=5, show=False, save=True):
    mode = 'grid'
    file_name = plot_folder + experiment_name + '_' + mode + '_n' + str(n) + '_minp' + str(minp) + '_maxp' + str(
        maxp) + '_'

    content = np.copy(aligned)
    number_of_units = content.shape[0] - 2
    grid = np.zeros((n, n, number_of_units), dtype=np.float32)

    sx = sy = 350
    content[0] += 6
    content[1] += 6
    ##make nxn (default 5x5) grid with the mean firing rate:
    for x in range(n):
        for y in range(n):
            boolean = np.logical_and(np.logical_and(content[0] > x / n * sx, content[0] < (x + 1) / n * sx),
                                     np.logical_and(content[1] > y / n * sy, content[1] < (y + 1) / n * sy))
            mask = np.tile(np.invert(boolean), (number_of_units, 1))
            masked = np.ma.masked_array(content[2:], mask=mask)
            grid[x, y, :] = np.mean(masked, axis=1)
    ##the firing rate per roi is the mean of the firing rates of the grid compartments in the roi #todo
    grid[1:n - 1, :, :] = np.mean(grid[1:n - 1, :, :], axis=0)[None, :, :]
    grid[:, 1:n - 1, :] = np.mean(grid[:, 1:n - 1, :], axis=1)[:, None, :]
    cmap = copy.copy(mpl.cm.get_cmap('Blues'))
    cmap.set_bad(color='grey')
    if single_figures:
        for unit in range(number_of_units):
            vmin = np.percentile(grid[:, :, unit][np.where(grid[:, :, unit] > 0)], minp)
            vmax = np.percentile(grid[:, :, unit][np.where(grid[:, :, unit] > 0)], maxp)

            fig = plt.figure(figsize=(5, 5))
            im = plt.imshow(np.ma.masked_where(grid[:, :, unit] == 0, grid[:, :, unit]).T, cmap=cmap, origin='upper',
                            interpolation='none', vmin=vmin, vmax=vmax)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            if save:
                plt.savefig(file_name + str(cluster_names[unit]) + '.jpg')
            if unit != 0:
                plt.title('firing rate unit ' + str(cluster_names[unit]))
            else:
                plt.title('firing rate all mua')
            if show:
                plt.show()
            plt.close(fig)

    if multi_figure:
        fig, axs = plt.subplots(number_of_units // 4 + 1, 4, sharex=True, sharey=True)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        for unit in range(number_of_units):
            vmin = np.percentile(grid[:, :, unit][np.where(grid[:, :, unit] > 0)], minp)
            vmax = np.percentile(grid[:, :, unit][np.where(grid[:, :, unit] > 0)], maxp)
            im = axs[unit // 4, unit % 4].imshow(np.ma.masked_where(grid[:, :, unit] == 0, grid[:, :, unit]).T,
                                                 cmap=cmap, origin='upper',
                                                 interpolation='none', vmin=vmin, vmax=vmax)
            # fig.colorbar(im, fraction=0.046, pad=0.04)
            divider = make_axes_locatable(axs[unit // 4, unit % 4])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical',
                         shrink=0.8)
            axs[unit // 4, unit % 4].axis('off')
            axs[unit // 4, unit % 4].set_title(cluster_names[unit], loc='right')

        axes_to_delete = axs[(number_of_units - 1) // 4, (number_of_units - 1) % 4 + 1:]
        for ax_to_delete in axes_to_delete:
            fig.delaxes(ax_to_delete)
        if save:
            plt.savefig(file_name + 'multiple' + '.jpg')
        if show:
            fig.suptitle(file_name)
            plt.show()
        plt.close(fig)

    unit_mean = np.mean(grid[:, :, 1:], axis=2)
    vmin = np.percentile(unit_mean[np.where(unit_mean > 0)], minp)
    vmax = np.percentile(unit_mean[np.where(unit_mean > 0)], maxp)

    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(np.ma.masked_where(unit_mean == 0, unit_mean).T, cmap=cmap, origin='upper', interpolation='none',
                    vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save:
        plt.savefig(file_name + 'all_units' + '.jpg')
    plt.title('firing rate all units')
    if show:
        plt.show()
    plt.close(fig)
    return


# plots the z score of the firing rate for every unit, and the mean of the z scores of all units
def plot_events(plot_folder, experiment_name, aligned, cluster_names, mode, event_indices, archive, single_figures,
                multi_figure,
                n=250, number_of_bins=20, show=False, save=True, do_archive=True):  # +- n frames around event
    file_name = plot_folder + experiment_name + '_' + mode + '_n' + str(n) + '_'
    content = np.copy(aligned)
    number_of_units = content.shape[0] - 2
    ##downsample aligned to binsize 2*n//number_of_bins:
    downsampled = np.empty((number_of_units, content.shape[1] // (2 * n) * number_of_bins))
    for step in range(downsampled.shape[1]):
        downsampled[:, step] = np.mean(
            aligned[2:, step * (2 * n) // number_of_bins:(step + 1) * (2 * n) // number_of_bins], axis=1)
    ##get the valid transition indices:
    transition_indices = [transition_index for transition_index in event_indices if
                          transition_index + n <= content.shape[1] and transition_index - n >= 0]
    if not transition_indices:  # check if there are any valid events
        return archive
    binned = np.zeros((number_of_units, number_of_bins, len(transition_indices)),
                      dtype=np.float32)  # 1.dim: unit, 2.dim: bin, 3.dim: transition_index
    for tindex, transition_index in enumerate(transition_indices):
        for bar in range(number_of_bins):
            # take mean of all firingrates within bins(bars)
            binned[:, bar, tindex] = np.mean(content[2:, transition_index - n:transition_index + n][:,
                                             (2 * n) * bar // number_of_bins: (2 * n) * (bar + 1) // number_of_bins],
                                             axis=1)
    mean_of_population = np.mean(binned, axis=2)  # mean of all transitons
    mean_of_all = np.mean(downsampled, axis=1)[:, None]  # mean of recording
    std_of_population = np.std(binned, axis=2)  # std of all transitions
    std_of_all = np.std(downsampled, axis=1)[:, None]  # std of recording
    n_of_samples = binned.shape[2]
    z_scores = (mean_of_population - mean_of_all) * np.sqrt(n_of_samples) / std_of_all  # compute z score
    sem = std_of_population / np.sqrt(n_of_samples)  # compute SEM (standard error of the mean)

    if do_archive:
        archive.loc[:, idx[mode, :]] = z_scores  # save z scores to archive
    if show or save:

        if single_figures:
            for unit in range(number_of_units):
                fig = plt.figure(figsize=(25, 5))
                plt.bar(np.arange(number_of_bins) + 0.5, z_scores[unit], yerr=sem[unit], width=1)
                plt.xticks(np.arange(number_of_bins + 1),
                           np.linspace(-number_of_bins // 2, number_of_bins // 2,
                                       number_of_bins + 1) * n / number_of_bins / 50)
                if save:
                    plt.savefig(file_name + str(cluster_names[unit]) + '.jpg')
                if unit != 0:
                    plt.title('z-score unit ' + str(cluster_names[unit]))
                else:
                    plt.title('z-score all mua')
                if show:
                    plt.show()
                plt.close(fig)

        if multi_figure:
            #  fig, axs = plt.subplots(number_of_units , sharex=True)
            fig, axs = plt.subplots(number_of_units // 4 + 1, 4, sharex=True)
            fig.set_figheight(15)
            fig.set_figwidth(15)
            for unit in range(number_of_units):
                # axs[unit].bar(np.arange(number_of_bins)+0.5, z_scores[unit], yerr=sem[unit], width=1)
                # axs[unit].set_title(cluster_names[unit], loc='right')
                # axs[unit].set_xticks(np.arange(number_of_bins+1))
                # axs[unit].set_xticklabels(np.linspace(-number_of_bins // 2, number_of_bins // 2, number_of_bins+1)*n/number_of_bins/50)
                axs[unit // 4, unit % 4].bar(np.arange(number_of_bins) + 0.5, z_scores[unit],
                                             width=1)  # , yerr=sem[unit]
                axs[unit // 4, unit % 4].set_title(cluster_names[unit], loc='right')
                axs[unit // 4, unit % 4].set_xticks(np.linspace(0, number_of_bins + 1, 5))
                axs[unit // 4, unit % 4].set_xticklabels(np.linspace(-number_of_bins // 2, number_of_bins // 2,
                                                                     5) * n / number_of_bins / 50)
            axes_to_delete = axs[(number_of_units - 1) // 4, (number_of_units - 1) % 4 + 1:]
            for ax_to_delete in axes_to_delete:
                fig.delaxes(ax_to_delete)
            if save:
                plt.savefig(file_name + 'multiple' + '.jpg')
            if show:
                fig.suptitle(file_name)
                plt.show()
            plt.close(fig)

        fig = plt.figure(figsize=(25, 5))
        mean_of_units = np.mean(z_scores, axis=0)
        std_of_units = np.std(z_scores, axis=0)
        n_of_units = z_scores.shape[0]
        sem_of_units = std_of_units / np.sqrt(n_of_units)
        plt.bar(np.arange(number_of_bins) + 0.5, mean_of_units, yerr=sem_of_units, width=1)
        plt.xticks(np.arange(number_of_bins + 1),
                   np.linspace(-number_of_bins // 2, number_of_bins // 2, number_of_bins + 1) * n / 50 / number_of_bins)
        if save:
            plt.savefig(file_name + 'all_units' + '.jpg')

        if show:
            plt.title('z-score all units')
            plt.show()
        plt.close(fig)
    return archive


# used for EZM
# plots bar diagramm with one bar per ROI and unit, indicating percent difference in firingrate in ROI to mean firingrate of unit
# plots the mean of all single unit plots
# fills archive['ROI_EZM']
def plot_arms(plot_folder, experiment_name, aligned, cluster_names, archive, single_figures, multi_figure,
              transition_size=2, n=150, show=False, save=True, do_archive=True):
    mode = 'arms'
    file_name = plot_folder + experiment_name + '_' + mode + '_n' + str(n) + '_'
    content = np.copy(aligned)
    number_of_units = content.shape[0] - 2
    grid = np.zeros((n, n, number_of_units), dtype=np.float32)  # 2d grid, indicating firing rate per coordinate

    content[1] -= 5  # shift 5 units upward (y axis is positive downward)
    sx = 400  # max x coordinate
    sy = 400  # max y coordinate
    ROI = np.zeros((number_of_units, 8))  # row indicates unit, column ROI

    masks = get_ezm_ROI_masks(n, transition_size)

    for unit in range(number_of_units):
        for x in range(n):
            for y in range(n):
                boolean = np.logical_and(np.logical_and(content[0] > x / n * sx, content[0] < (x + 1) / n * sx),
                                         np.logical_and(content[1] > y / n * sy, content[1] < (y + 1) / n * sy))
                mask = np.tile(np.invert(boolean), (number_of_units, 1))
                masked = np.ma.masked_array(content[2:], mask=mask)
                grid[x, y, :] = np.mean(masked, axis=1)  # assign mean firing rate per coordinate)

        for quadrant in range(8):
            valid_values_in_quadrant = grid[:, :, unit][np.logical_and(masks[quadrant] == 1, grid[:, :,
                                                                                             ## to visulalize the masks:                                                                                unit] != 0)]  # 1d array of all values in quadrant and visited
                                                                                             # fig=plt.figure()
                                                                                             # plt.imshow(grid[:,:,unit].T)
                                                                                             # plt.imshow(masks[quadrant].T, alpha=0.5)
                                                                                             # plt.show()
                                                                                             # plt.close(fig)
                                                                                             mean_in_quadrant = valid_values_in_quadrant.mean()
            mean_of_unit = grid[:, :, unit][grid[:, :, unit] != 0].mean()
            if mean_of_unit != 0:
                ROI[unit, quadrant] = (
                                              mean_in_quadrant - mean_of_unit) * 100 / mean_of_unit  # percent difference to mean of unit
            else:
                raise Exception('unit mean is zero, code to ctrl-F for: 23456')

        if (save or show) and single_figures:
            fig = plt.figure(figsize=(5, 5))
            toplot = ROI[unit][[0, 1, 2, 3, 4, 6, 5, 7]]  # arrangement: ['top right', 'top left', 'bottom left',
            # 'bottom right', 'top (open)', 'bottom (open)', 'left (closed)', 'right (closed)']
            plt.bar(np.arange(8), toplot)
            plt.xticks(np.arange(8), [0, 1, 2, 3, 4, 6, 5, 7])
            if save:
                plt.savefig(file_name + str(cluster_names[unit]) + '.jpg')  # cluster_names[0] = 255 -> mua
            if unit != 0:
                plt.title('firing rate unit ' + str(cluster_names[unit]))
            else:
                plt.title('firing rate all mua')
            if show:
                plt.show()
            plt.close(fig)

    if (save or show) and multi_figure:
        fig, axs = plt.subplots(number_of_units // 4 + 1, 4, sharex=True)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        for unit in range(number_of_units):
            toplot = ROI[unit][[0, 1, 2, 3, 4, 6, 5, 7]]  # arrangement: ['top right', 'top left', 'bottom left',
            # 'bottom right', 'top (open)', 'bottom (open)', 'left (closed)', 'right (closed)']

            axs[unit // 4, unit % 4].bar(np.arange(8), toplot)
            axs[unit // 4, unit % 4].set_xticks(np.arange(8))
            axs[unit // 4, unit % 4].set_xticklabels([0, 1, 2, 3, 4, 6, 5, 7])
            axs[unit // 4, unit % 4].set_title(cluster_names[unit], loc='right')

        # axes_to_delete = axs[(number_of_units - 1) // 4, (number_of_units - 1) % 4 + 1:] #todo
        # for ax_to_delete in axes_to_delete:
        #     fig.delaxes(ax_to_delete)
        if save:
            plt.savefig(file_name + 'multiple' + '.jpg')
        if show:
            fig.suptitle(file_name)
            plt.show()
        plt.close(fig)

    if do_archive:
        archive.loc[:, idx['ROI_EZM', :]] = ROI  # add to archive

    if save or show:  # plot mean of all single unit plots
        ROImean = np.mean(ROI[1:], axis=0)
        fig = plt.figure(figsize=(5, 5))
        toplot = ROImean[[0, 1, 2, 3, 4, 6, 5, 7]]
        plt.bar(np.arange(8), toplot, width=1)
        plt.xticks(np.arange(8), [0, 1, 2, 3, 4, 6, 5, 7])
        if save:
            plt.savefig(file_name + 'all_units' + '.jpg')
        plt.title('firing rate all units')
        if show:
            plt.show()
        plt.close(fig)
    return archive


# used for OF
# plots bar diagramm with one bar per ROI and unit, indicating percent difference in firing rate in ROI to mean firing rate of unit
# plots the mean of all single unit plots
# fills archive['ROI_OF']
def plot_corners(plot_folder, experiment_name, aligned, cluster_names, archive, single_figures, multi_figure,
                 n=5, show=False, save=True, do_archive=True):
    mode = 'corners'
    file_name = plot_folder + experiment_name + '_' + mode + '_n' + str(n) + '_'
    content = np.copy(aligned)
    number_of_units = content.shape[0] - 2
    grid = np.zeros((n, n, number_of_units), dtype=np.float32)

    sx = sy = 350
    content[0] += 6
    content[1] += 6
    ROI = np.zeros((number_of_units, 9))  # rows: units, columns: ROIs
    for x in range(n):
        for y in range(n):
            boolean = np.logical_and(np.logical_and(content[0] > x / n * sx, content[0] < (x + 1) / n * sx),
                                     np.logical_and(content[1] > y / n * sy, content[1] < (y + 1) / n * sy))
            mask = np.tile(np.invert(boolean), (number_of_units, 1))
            masked = np.ma.masked_array(content[2:], mask=mask)
            grid[x, y, :] = np.mean(masked, axis=1)
    grid[1:n - 1, :, :] = np.mean(grid[1:n - 1, :, :], axis=0)[None, :, :]
    grid[:, 1:n - 1, :] = np.mean(grid[:, 1:n - 1, :], axis=1)[:, None, :]

    for unit in range(number_of_units):
        # now every point within a ROI contains the mean firingrate of the ROI it belongs tog
        # take one point from every ROI:
        takenfrom = [(n - 1, 0), (0, 0), (0, n - 1), (n - 1, n - 1), (n - 1, 1), (1, 0), (0, 1), (1, n - 1), (1, 1)]
        for index in range(9):
            mean = grid[:, :, unit].mean()
            if mean == 0:
                mean = 1
            ROI[unit, index] = (grid[:, :, unit][takenfrom[
                index]] - mean) * 100 / mean  # assign percent difference to unit mean to every ROI
        if (save or show) and single_figures:

            fig = plt.figure(figsize=(5, 5))
            plt.bar(np.arange(9),
                    ROI[unit])  # arrangement of the ROIs: {0: 'top right', 1: 'top left', 2: 'bottom left',
            # 3: 'bottom right', 4: 'right', 5: 'top', 6: 'left', 7: 'bottom', 8: 'middle'}
            plt.xticks(np.arange(9), np.arange(9))
            if save:
                plt.savefig(file_name + str(cluster_names[unit]) + '.jpg')
            if show:
                if unit != 0:
                    plt.title('firing rate unit ' + str(cluster_names[unit]))
                else:
                    plt.title('firing rate all mua')
                plt.show()
            plt.close(fig)
    if (save or show) and multi_figure:
        fig, axs = plt.subplots(number_of_units // 4 + 1, 4, sharex=True)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        for unit in range(number_of_units):
            axs[unit // 4, unit % 4].bar(np.arange(9), ROI[
                unit])  # arrangement of the ROIs: {0: 'top right', 1: 'top left', 2: 'bottom left',
            # 3: 'bottom right', 4: 'right', 5: 'top', 6: 'left', 7: 'bottom', 8: 'middle'}
            axs[unit // 4, unit % 4].set_xticks(np.arange(9))
            axs[unit // 4, unit % 4].set_xticklabels(np.arange(9))
            axs[unit // 4, unit % 4].set_title(cluster_names[unit], loc='right')

        # axes_to_delete = axs[(number_of_units - 1) // 4, (number_of_units - 1) % 4 + 1:] #todo
        # for ax_to_delete in axes_to_delete:
        #     fig.delaxes(ax_to_delete)
        if save:
            plt.savefig(file_name + 'multiple' + '.jpg')
        if show:
            fig.suptitle(file_name)
            plt.show()
        plt.close(fig)

    if do_archive:
        archive.loc[:, idx['ROI_OF', :]] = ROI
    if save or show:  # plot mean of all single unit plots
        unit_sum = (np.mean(ROI[1:], axis=0))
        fig = plt.figure(figsize=(5, 5))
        plt.bar(np.arange(9), unit_sum)
        plt.xticks(np.arange(9), np.arange(9))
        if save:
            plt.savefig(file_name + 'all_units' + '.jpg')
        plt.title('firing rate all units')
        if show:
            plt.show()
        plt.close(fig)
    return archive


# computes the EZM score
def get_ezm_score(rois):
    a1 = 0.25 * (np.abs(rois[:, 5] - rois[:, 4]) + np.abs(rois[:, 5] - rois[:, 6]) + np.abs(
        rois[:, 7] - rois[:, 4]) + np.abs(rois[:, 7] - rois[:, 6]))
    b1 = 0.5 * (np.abs(rois[:, 5] - rois[:, 7]) + np.abs(rois[:, 4] - rois[:, 6]))
    open_close = (a1 - b1) / (a1 + b1)
    closed = (rois[:, 5] + rois[:, 7]) / 2 - (rois[:, 4] + rois[:, 6]) / 2

    a2 = 1 / 16 * (np.abs(rois[:, 0] - rois[:, 4]) + np.abs(rois[:, 0] - rois[:, 7])
                   + np.abs(rois[:, 0] - rois[:, 6]) + np.abs(rois[:, 0] - rois[:, 5])
                   + np.abs(rois[:, 1] - rois[:, 4]) + np.abs(rois[:, 1] - rois[:, 7])
                   + np.abs(rois[:, 1] - rois[:, 6]) + np.abs(rois[:, 1] - rois[:, 5])
                   + np.abs(rois[:, 2] - rois[:, 4]) + np.abs(rois[:, 2] - rois[:, 7])
                   + np.abs(rois[:, 2] - rois[:, 6]) + np.abs(rois[:, 2] - rois[:, 5])
                   + np.abs(rois[:, 3] - rois[:, 4]) + np.abs(rois[:, 3] - rois[:, 7])
                   + np.abs(rois[:, 3] - rois[:, 6]) + np.abs(rois[:, 3] - rois[:, 5]))
    b2 = 1 / 12 * (np.abs(rois[:, 0] - rois[:, 1]) + np.abs(rois[:, 1] - rois[:, 3])
                   + np.abs(rois[:, 1] - rois[:, 2]) + np.abs(rois[:, 0] - rois[:, 3])
                   + np.abs(rois[:, 0] - rois[:, 2]) + np.abs(rois[:, 2] - rois[:, 3])
                   + np.abs(rois[:, 4] - rois[:, 7]) + np.abs(rois[:, 4] - rois[:, 6])
                   + np.abs(rois[:, 4] - rois[:, 5]) + np.abs(rois[:, 5] - rois[:, 7])
                   + np.abs(rois[:, 5] - rois[:, 6]) + np.abs(rois[:, 6] - rois[:, 7]))
    crossing = (a2 - b2) / (a2 + b2)
    transition = (rois[:, 0] + rois[:, 1] + rois[:, 2] + rois[:, 3]) / 4 - (
                rois[:, 5] + rois[:, 7] + rois[:, 4] + rois[:, 6]) / 4
    return open_close, crossing, closed, transition


# computes the OF score
def get_of_score(rois):
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
    of_corners = (rois[:, 0] + rois[:, 1] + rois[:, 2] + rois[:, 3]) / 4 - (
                rois[:, 4] + rois[:, 5] + rois[:, 6] + rois[:, 7] + rois[:, 8]) / 5
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
    of_middle = rois[:, 8] - (
                rois[:, 0] + rois[:, 1] + rois[:, 2] + rois[:, 3] + rois[:, 4] + rois[:, 5] + rois[:, 6] + rois[:,
                                                                                                           7]) / 8
    return of_corners_score, of_middle_score, of_corners, of_middle


# plot number of spikes per range of phases for every unit
# plot plot average number of spikes per range of phases, where every unit contributes the same
# makes theta_phase of archive
def plot_phase(phase_aligned, original_aligned, vHIP_pads, plot_folder, experiment_name, cluster_names, archive,
               environment, number_of_bins,
               show, save, do_archive, single_figures, multi_figure):
    mode = 'phase'
    key = 'theta_phase_' + environment
    unit_keys = [key + '_' + str(vHIP_pad) for vHIP_pad in vHIP_pads]  # column of archive to save to
    file_name = plot_folder + experiment_name + '_' + mode + '_'
    # phase angle in range(0,number of bins):
    phase = phase_aligned

    for i, unit in enumerate(cluster_names):
        mask = np.tile(np.invert(original_aligned[i]), (phase.shape[0], 1))  # mask phase values where no spike ocurred
        masked = np.ma.masked_array(phase, mask=mask)  # todo
        binned = np.zeros((masked.shape[0], number_of_bins))
        for bin in range(number_of_bins):
            binned[:, bin] = np.sum(masked == bin, axis=1)  # slow part
        unit_mean = np.mean(binned, axis=1)
        normalized = (binned - unit_mean[:, None]) / unit_mean[:, None]
        if do_archive:
            archive.loc[unit, idx[unit_keys, :]] = np.reshape(binned, -1).astype(np.uint32)
        if save or show:
            if single_figures:
                for row, vHIP_pad in enumerate(vHIP_pads):
                    fig = plt.figure(figsize=(5, 5))
                    # plt.bar(np.arange(number_of_bins) + 0.5, normalized[row], width=1)
                    ##todo:
                    plt.polar(np.arange(-number_of_bins // 2, number_of_bins // 2 + 1) * math.pi * 2 // number_of_bins, normalized[row])

                    # plt.xticks(np.arange(number_of_bins + 1),
                    #            np.arange(-number_of_bins // 2, number_of_bins // 2 + 1) * 180 * 2 // number_of_bins)
                    if save:
                        plt.savefig(file_name + 'unit_' + str(unit) + '_pad_' + str(vHIP_pad) + '.jpg')
                    if show:
                        if unit != -1:
                            plt.title('phase plot unit ' + str(unit) + ', pad ' + str(vHIP_pad))
                        else:
                            plt.title('phase plot all mua, pad ' + str(vHIP_pad))
                        plt.show()
                    plt.close(fig)
            if multi_figure:
                fig, axs = plt.subplots(8, 4, sharex=True, sharey=True)
                fig.set_figheight(15)
                fig.set_figwidth(15)
                for row, vHIP_pad in enumerate(vHIP_pads):
                    toplot = normalized[row]
                    pad_number = vHIP_pad - 33
                    number_of_bins = toplot.shape[0]
                    axs[pad_number // 4, pad_number % 4].bar(np.arange(number_of_bins) + 0.5, toplot, width=1)
                    axs[pad_number // 4, pad_number % 4].set_xticks(np.arange(number_of_bins + 1))
                    axs[pad_number // 4, pad_number % 4].set_xticklabels(
                        np.arange(-number_of_bins // 2, number_of_bins // 2 + 1) * 180 * 2 // number_of_bins)
                    axs[pad_number // 4, pad_number % 4].set_title(pad_number + 33, loc='right')

                if save:
                    plt.savefig(file_name + 'unit_' + str(unit) + '_pad_all' + '.jpg')
                if show:
                    fig.suptitle(key + 'unit_' + str(unit))
                    plt.show()
                plt.close(fig)

            fig = plt.figure(figsize=(5, 5))
            plt.bar(np.arange(number_of_bins) + 0.5, normalized.mean(axis=0), width=1)
            plt.xticks(np.arange(number_of_bins + 1),
                       np.arange(-number_of_bins // 2, number_of_bins // 2 + 1) * 180 * 2 // number_of_bins)

            if save:
                plt.savefig(file_name + '_unit_' + str(unit) + 'all_pads.jpg')
            if show:
                plt.title('phase plot all pads, unit ' + str(unit))
                plt.show()
            plt.close(fig)
    return archive
