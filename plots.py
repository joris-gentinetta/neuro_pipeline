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
#plots 2d firing rate, temporally filtered
def plot_classic(environment, plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
                 cluster_names, sigma=10, minp=0, maxp=95, n=150, show=False, save=True):
    movement = events['movement']
    mode = 'classic'
    file_name = plot_folder + experiment_name + '_' + mode + '_sigma' + str(sigma) + '_n' + str(n) + '_minp' + str(
        minp) + '_maxp' + str(maxp) + '_'
    data = np.empty(raw_data.shape, dtype=np.float32)

    for i in range(raw_data.shape[0]):
        data[i] = gaussian_filter1d(raw_data[i], sigma=sigma) #filter temporally (gaussian)
    grid = np.zeros((n, n, data.shape[0]), dtype=np.float32) #2d grid containing firing rate at coordinates given by row(y) and column(x)
    xyv = np.empty(
        (3, min(len(movement['calib_traj_x'].index) - video_trigger - off, data.shape[1] - physio_trigger - off)),
        dtype=np.float32)
    xyv[0] = movement['calib_traj_x'][video_trigger + off: xyv.shape[1] + video_trigger + off] #x coordinates
    xyv[1] = movement['calib_traj_y'][video_trigger + off: xyv.shape[1] + video_trigger + off] #y coordinates
    if environment == 'EZM':
        xyv[1] -= 5 #shift upwards
        sx = 400 # max x coordinate
        sy = 400
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
    elif environment == 'OFT':
        sx = sy = 350
        xyv[0] += 6 #shift downward
        xyv[1] += 6 #shift right

    cmap = copy.copy(mpl.cm.get_cmap('jet'))
    cmap.set_bad(color='grey')
    for unit in range(data.shape[0]):
        xyv[2] = data[unit][physio_trigger + off: xyv.shape[1] + physio_trigger + off] + make_path_visible

        for x in range(n):
            for y in range(n):
                boolean = np.logical_and(np.logical_and(xyv[0] > x / n * sx, xyv[0] < (x + 1) / n * sx),
                                         np.logical_and(xyv[1] > y / n * sy, xyv[1] < (y + 1) / n * sy))
                bsum = np.sum(boolean)
                if bsum != 0:
                    grid[x, y, unit] = np.sum(xyv[2][boolean]) / bsum

        vmin = np.percentile(grid[:, :, unit][np.where(grid[:, :, unit] > 0)], minp) #minp percent of the values are below vmin
        vmax = np.percentile(grid[:, :, unit][np.where(grid[:, :, unit] > 0)], maxp)

        fig = plt.figure(figsize=(5, 5))

        im = plt.imshow(np.ma.masked_where(grid[:, :, unit] == 0, grid[:, :, unit]).T, cmap=cmap, origin='upper',
                        interpolation='none', vmin=vmin, vmax=vmax)
        if environment == 'EZM':
            plt.imshow(mask.T, cmap='Greys', origin='upper',
                       interpolation='none', alpha=0.1)

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
    if save: plt.savefig(file_name + 'all_units' + '.jpg')
    plt.title('firing rate all units')
    if show:
        plt.show()
    plt.close(fig)
    return

#controlled, take comments from plot_classic
#plots 2d firing rate
def plot_raw(environment, plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
             cluster_names, minp=0, maxp=90, n=150, show=False, save=True):
    movement = events['movement']
    mode = 'raw'
    file_name = plot_folder + experiment_name + '_' + mode + '_n' + str(n) + '_minp' + str(minp) + '_maxp' + str(
        maxp) + '_'

    data = raw_data
    grid = np.zeros((n, n, data.shape[0]), dtype=np.float32)
    xyv = np.empty(
        (3, min(len(movement['calib_traj_x'].index) - video_trigger - off, data.shape[1] - physio_trigger - off)),
        dtype=np.float32)
    xyv[0] = movement['calib_traj_x'][video_trigger + off: xyv.shape[1] + video_trigger + off]
    xyv[1] = movement['calib_traj_y'][video_trigger + off: xyv.shape[1] + video_trigger + off]
    if environment == 'EZM':
        xyv[1] -= 5
        sx = 400
        sy = 400
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

    elif environment == 'OFT':
        sx = sy = 350
        xyv[0] += 6
        xyv[1] += 6

    cmap = copy.copy(mpl.cm.get_cmap("Blues"))
    cmap.set_bad(color='grey')
    for unit in range(data.shape[0]):
        xyv[2] = data[unit][physio_trigger + off: xyv.shape[1] + physio_trigger + off] + make_path_visible

        for x in range(n):
            for y in range(n):
                boolean = np.logical_and(np.logical_and(xyv[0] > x / n * sx, xyv[0] < (x + 1) / n * sx),
                                         np.logical_and(xyv[1] > y / n * sy, xyv[1] < (y + 1) / n * sy))
                bsum = np.sum(boolean)
                if bsum != 0:
                    grid[x, y, unit] = np.sum(xyv[2][boolean]) / bsum

        vmin = np.percentile(grid[:, :, unit][np.where(grid[:, :, unit] > 0)], minp)
        vmax = np.percentile(grid[:, :, unit][np.where(grid[:, :, unit] > 0)], maxp)

        fig = plt.figure(figsize=(5, 5))
        im = plt.imshow(np.ma.masked_where(grid[:, :, unit] == 0, grid[:, :, unit]).T, cmap=cmap, origin='upper',
                        interpolation='none', vmin=vmin, vmax=vmax)
        if environment == 'EZM':
            plt.imshow(mask.T, cmap='Greys', origin='upper',
                       interpolation='none', alpha=0.1)
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
    if save:
        plt.savefig(file_name + 'all_units' + '.jpg')
    plt.title('firing rate all units')
    if show:
        plt.show()
    plt.close(fig)
    return

#controlled and commented
#plots spatially filtered circle (abstraction of the EZM) for every unit
def plot_circle(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger, cluster_names,
                n=360, sigma=-1, show=False, save=True):
    movement = events['movement']
    mode = 'circle'
    if sigma < 0: #set default sigma such that +-2 sigma equals 1/8 of the circle
        sigma = n / 8 / 4
    file_name = plot_folder + experiment_name + '_' + mode + '_sigma' + str(sigma) + '_n' + str(n) + '_'

    data = raw_data #data is not temporally filtered
    grid = np.zeros((data.shape[0], n), dtype=np.float32) #rows: unit, columns: position in circle, 0 is on positive x axis
    xyv = np.empty(
        (4, min(len(movement['calib_traj_x'].index) - video_trigger - off, data.shape[1] - physio_trigger - off)))
    xyv[0] = movement['calib_traj_x'][video_trigger + off: xyv.shape[1] + video_trigger + off] #x coordinates
    xyv[1] = movement['calib_traj_y'][video_trigger + off: xyv.shape[1] + video_trigger + off] #y coordinates
    xyv[1] -= 5 #shift upward
    sx = 400 #max x coordinate
    sy = 400
    middle_x = sx / 2
    middle_y = sy / 2
    xyv[0] -= middle_x #shift center to zero
    xyv[1] -= middle_y
    xyv[1] *= -1 #flip y axis (x=0,y=0 was originally in top left corner)
    xyv[3] = np.arctan2(xyv[1], xyv[0]) #get angle in range -pi, -pi
    xyv[3] += (xyv[3] < 0) * 2 * math.pi #shift range to 0, 2pi

    for unit in range(data.shape[0]):
        xyv[2] = data[unit][physio_trigger + off: xyv.shape[1] + physio_trigger + off] #assign firingrates of unit

        for r in range(n): #assign mean firingrate to every angle
            boolean = np.logical_and(xyv[3] > math.pi * 2 * r / n, xyv[3] < math.pi * 2 * (r + 1) / n)
            grid[unit, r] = np.mean(xyv[2][boolean])
            bsum = np.sum(boolean)
            if bsum != 0:
                grid[unit, r] = np.sum(xyv[2][boolean]) / bsum
        grid[unit] = gaussian_filter1d(grid[unit], sigma=sigma, mode='wrap') #gaussian filter, wraps around at n, 0
        fig = plt.figure(figsize=(5, 5))
        colors = cm.jet(plt.Normalize()(grid[unit]))
        for quadrant in range(4):
            colors[(40 + quadrant * 90) * n // 360] = [0, 0, 0, 1] #mark transitons
        plt.pie(np.ones(n), colors=colors)
        plt.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=grid[unit].min(), vmax=grid[unit].max()), cmap='jet'),
                     shrink=0.6)
        my_circle = plt.Circle((0, 0), 0.8, color='white')
        fig.gca().add_artist(my_circle) #make the inside of the pieplot white
        if save:
            plt.savefig(file_name + str(cluster_names[unit]) + '.jpg')
        if unit != 0:
            plt.title('firing rate unit ' + str(cluster_names[unit]))
        else:
            plt.title('firing rate all mua')
        if show:
            plt.show()
        plt.close(fig)
    #plot mean of all units:
    unit_mean = np.mean(grid[1:]-grid[1:].mean(axis=1)[:, None], axis=0)
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

#controlled
#plots 2d image with mean firing rate for every roi
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

    cmap = copy.copy(mpl.cm.get_cmap('Blues'))
    cmap.set_bad(color='grey')
    for unit in range(data.shape[0]):
        xyv[2] = data[unit][physio_trigger + off: xyv.shape[1] + physio_trigger + off] + make_path_visible

        for x in range(n):
            for y in range(n):
                boolean = np.logical_and(np.logical_and(xyv[0] > x / n * sx, xyv[0] < (x + 1) / n * sx),
                                         np.logical_and(xyv[1] > y / n * sy, xyv[1] < (y + 1) / n * sy))
                bsum = np.sum(boolean)
                if bsum != 0:
                    grid[x, y, unit] = np.sum(xyv[2][boolean]) / bsum

        for d in range(n):
            grid[1:n - 1, d, unit] = np.mean(grid[1:n - 1, d, unit])

        for d in range(n):
            grid[d, 1:n - 1, unit] = np.mean(grid[d, 1:n - 1, unit])
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


def plot_transitions(plot_folder, experiment_name, raw_data, events, cluster_names, video_trigger, archive, mode,
                     plotmode='percent', n=500, m=5, show=False, save=True, do_archive=True):
    file_name = plot_folder + experiment_name + '_' + mode + '_n' + str(n) + '_'

    data = raw_data
    x = np.arange(-n, n + 1, (2 * n + 1) // (m - 1))
    transitions = events['transitions']
    transition_indices = [transition_index + video_trigger for transition_index in transitions[mode] if
                          transition_index + video_trigger + n + 1 <= data.shape[
                              1] and transition_index + video_trigger - n >= 0] #create valid indices
    grid = [np.zeros((len(transition_indices), m), dtype=np.float32) for _ in range(data.shape[0])] #list, every entry corresponds to one unit and contains a nparray with:
    # rows: indices, columns: m pooled values around indices

    for tindex, transition_index in enumerate(transition_indices): #fill grid with values
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
        tobearchived = np.zeros((data.shape[0], 1001, len(archive_transition_indices))) #unit, 1001, index
        for tindex, transition_index in enumerate(archive_transition_indices):
            for unit in range(data.shape[0]):
                tobearchived[unit, :, tindex] = data[unit, transition_index - 500:transition_index + 501]
        tobearchived = np.mean(tobearchived, axis=2) - np.mean(data, axis=1)[:, None]
        archive.loc[:, idx[mode, :]] = tobearchived
    if show or save:
        # for unit in range(grid.shape[0]):
        #    grid[unit] = gaussian_filter1d(grid[unit], sigma=sigma)
        for unit in range(data.shape[0]):
            fig = plt.figure(figsize=(5, 5))
            if plotmode == 'std':
                plt.bar(x, mean_std[unit][0], yerr=mean_std[unit][1], width=25)
            elif plotmode == 'percent':
                mean = np.mean(mean_std[unit][0])
                mean = np.where(mean != 0, mean, 1)
                toplot = (mean_std[unit][0] - mean) * 100 / mean

                plt.bar(x, toplot, width=25)
            if save:
                plt.savefig(file_name + str(cluster_names[unit]) + '.jpg')

            if unit != 0:
                plt.title('firing rate unit ' + str(cluster_names[unit]))
            else:
                plt.title('firing rate all mua')
            if show:
                plt.show()
            plt.close(fig)

        unit_sum = sum(grid[1:])
        sum_mean = np.mean(unit_sum, axis=0)
        sum_std = np.std(unit_sum, axis=0)
        mean = np.mean(sum_mean)
        if mean == 0:
            mean = 1
        if plotmode == 'std':
            plt.bar(x, sum_mean, yerr=sum_std, width=25)
        elif plotmode == 'percent':
            plt.bar(x, (sum_mean - mean) * 100 / mean, width=25)
        if save:
            plt.savefig(file_name + 'all_units' + '.jpg')
        plt.title('firing rate all units')
        if show:
            plt.show()
        plt.close(fig)
    return archive

# controlled and well commented
# plots bar diagramm with one bar per ROI and unit, indicating percent difference in firingrate in ROI to mean firingrate of unit
# fills archive['ROI_EZM']
def plot_arms(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
              cluster_names, archive, transition_size=2, n=150, show=False, save=True, do_archive=True):
    movement = events['movement']
    mode = 'arms'
    file_name = plot_folder + experiment_name + '_' + mode + '_n' + str(n) + '_'

    data = raw_data #unfiltered data
    grid = np.zeros((n, n, data.shape[0]), dtype=np.float32) #2d grid, indicating firing rate per coordinate
    xyv = np.empty(
        (3, min(len(movement['calib_traj_x'].index) - video_trigger - off, data.shape[1] - physio_trigger - off)),
        dtype=np.float32)
    xyv[0] = movement['calib_traj_x'][video_trigger + off: xyv.shape[1] + video_trigger + off] #x component of the trajectory
    xyv[1] = movement['calib_traj_y'][video_trigger + off: xyv.shape[1] + video_trigger + off] #y component of the trajectoru

    xyv[1] -= 5 # shift 5 units upward (y axis is positive downward)
    sx = 400 #max x coordinate
    sy = 400 #max y coordinate
    masks = [np.zeros((n, n), dtype=np.bool) for _ in range(8)] #create mask for every ROI
    ROI = np.zeros((data.shape[0], 8)) # row indicates unit, column ROI
    for x in range(n):
        for y in range(n):
            angle = np.arctan2((y - n / 2), x - n / 2) #get angle in range -pi, pi
            angle += (angle < 0) * 2 * math.pi #convert to range 0, 2pi
            for quadrant in range(4):
                target = 40 + quadrant * 90 #the EZM is tilted by 40 degrees
                degree = angle * 360 // (2 * math.pi) #convert radian to degree
                if degree in range(target - transition_size, target + transition_size): #create transition ROIs
                    masks[quadrant][x, n - y - 1] = 1 #n - y - 1 is taking into account that y axis goes downward
                if degree in np.arange(target + transition_size, target + 90 - transition_size) % 360: #create open and closed ROIs
                    masks[quadrant + 4][x, n - y - 1] = 1

    # for unit in range(8): #test orientation of the masks
    #     f = plt.figure()
    #     plt.imshow(masks[unit].T)
    #     plt.show()
    #     plt.close(f)

    for unit in range(data.shape[0]):
        xyv[2] = data[unit][physio_trigger + off: xyv.shape[1] + physio_trigger + off] + make_path_visible #assign firing rate to coordinates

        for x in range(n):
            for y in range(n):
                boolean = np.logical_and(np.logical_and(xyv[0] > x / n * sx, xyv[0] < (x + 1) / n * sx),
                                         np.logical_and(xyv[1] > y / n * sy, xyv[1] < (y + 1) / n * sy))
                bsum = np.sum(boolean)
                if bsum != 0:
                    grid[x, y, unit] = np.sum(xyv[2][boolean]) / bsum #assign sum of firing rate of all times visited divided
                    # by number of times visited (mean firing rate per coordinate)

        for quadrant in range(8):
            valid_values_in_quadrant = grid[:, :, unit][np.logical_and(masks[quadrant] == 1, grid[:, :, unit] != 0)] #1d array of all values in quadrant and visited
            # fig=plt.figure()
            # plt.imshow(grid[:,:,unit].T)
            # plt.imshow(masks[quadrant].T, alpha=0.5)
            # plt.show()
            # plt.close(fig)
            mean_in_quadrant = valid_values_in_quadrant.mean()
            mean_of_unit = grid[:, :, unit][grid[:, :, unit] != 0].mean()
            if mean_of_unit != 0:
                ROI[unit, quadrant] = (mean_in_quadrant - mean_of_unit) * 100 / mean_of_unit #percent difference to mean of unit
            else:
                raise Exception('unit mean is zero, line 477')

        if save or show:
            fig = plt.figure(figsize=(5, 5))
            toplot = ROI[unit][[0, 1, 2, 3, 4, 6, 5, 7]] #arrangement: ['top right', 'top left', 'bottom left',
            # 'bottom right', 'top (open)', 'bottom (open)', 'left (closed)', 'right (closed)']
            plt.bar(np.arange(8), toplot)
            if save:
                plt.savefig(file_name + str(cluster_names[unit]) + '.jpg') # cluster_names[0] = 255 -> mua
            if unit != 0:
                plt.title('firing rate unit ' + str(cluster_names[unit]))
            else:
                plt.title('firing rate all mua')
            if show:
                plt.show()
            plt.close(fig)
    if do_archive:
        archive.loc[:, idx['ROI_EZM', :]] = ROI # add to archive

    if save or show: #plot mean of all single unit plots
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


def plot_corners(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger, cluster_names,
                 archive,
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

    for unit in range(data.shape[0]):
        xyv[2] = data[unit][physio_trigger + off: xyv.shape[1] + physio_trigger + off] + make_path_visible

        for x in range(n):
            for y in range(n):
                boolean = np.logical_and(np.logical_and(xyv[0] > x / n * sx, xyv[0] < (x + 1) / n * sx),
                                         np.logical_and(xyv[1] > y / n * sy, xyv[1] < (y + 1) / n * sy))
                bsum = np.sum(boolean)
                if bsum != 0:
                    grid[x, y, unit] = np.sum(xyv[2][boolean]) / bsum

        for d in range(n):
            grid[1:n - 1, d, unit] = np.mean(grid[1:n - 1, d, unit])

        for d in range(n):
            grid[d, 1:n - 1, unit] = np.mean(grid[d, 1:n - 1, unit])
        takenfrom = [(n - 1, 0), (0, 0), (0, n - 1), (n - 1, n - 1), (n - 1, 1), (1, 0), (0, 1), (1, n - 1), (1, 1)]
        for index in range(9):
            mean = grid[:, :, unit].mean()
            if mean == 0:
                mean = 1
            ROI[unit, index] = (grid[:, :, unit][takenfrom[index]] - mean) * 100 / mean

        if save or show:

            fig = plt.figure(figsize=(5, 5))
            plt.bar(np.arange(9), ROI[unit])
            if save:
                plt.savefig(file_name + str(cluster_names[unit]) + '.jpg')

            if unit != 0:
                plt.title('firing rate unit ' + str(cluster_names[unit]))
            else:
                plt.title('firing rate all mua')
            if show:
                plt.show()
            plt.close(fig)
    if do_archive:
        archive.loc[:, idx['ROI_OF', :]] = ROI
    if save or show:
        mean = grid.mean()
        if mean == 0:
            mean = 1
        unit_sum = (np.mean(ROI[1:], axis=0) - grid.mean()) * 100 / mean
        fig = plt.figure(figsize=(5, 5))
        plt.bar(np.arange(9), unit_sum)
        if save:
            plt.savefig(file_name + 'all_units' + '.jpg')
        plt.title('firing rate all units')
        if show:
            plt.show()
        plt.close(fig)
    return archive


def get_ezm_score(rois):
    mean = np.mean(rois, axis=1)
    mean = np.where(mean != 0, mean, 1)
    rois = (rois - mean[:, None]) / mean[:, None]
    a1 = 0.25 * (np.abs(rois[:, 5] - rois[:, 4]) + np.abs(rois[:, 5] - rois[:, 6]) + np.abs(
        rois[:, 7] - rois[:, 7]) + np.abs(rois[:, 7] - rois[:, 6]))
    b1 = 0.5 * (np.abs(rois[:, 5] - rois[:, 7]) + np.abs(rois[:, 4] - rois[:, 6]))
    open_close = (a1 - b1) / (a1 + b1)

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
    closed = (rois[:, 5] + rois[:, 7]) / 2
    transition = (rois[:, 0] + rois[:, 1] + rois[:, 2] + rois[:, 3]) / 4
    return open_close, crossing, closed, transition


def get_of_score(rois):
    mean = np.mean(rois, axis=1)
    mean = np.where(mean != 0, mean, 1)
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


def plot_phase(vHIP_pads, circus, plot_folder, experiment_name, off, physio_trigger, cluster_names, archive,
               environment,
               show, save, do_archive):
    offset = (physio_trigger + off) * 20000 // 50
    mode = 'phase'
    key = 'theta_phase_' + environment
    unit_keys = [key + '_' + str(vHIP_pad) for vHIP_pad in vHIP_pads]
    file_name = plot_folder + experiment_name + '_' + mode + '_'
    number_of_bins = 8
    factor = 360 // number_of_bins
    phase = np.load(circus + 'phase_files/' + experiment_name + '.npy')[:, offset:] // factor
    original_data = np.load(circus + 'original_' + experiment_name + '.npy')[:, offset:]
    for i, unit in enumerate(cluster_names):
        mask = np.tile(np.invert(original_data[i]), (phase.shape[0], 1))
        masked = np.ma.masked_array(phase, mask=mask)
        phase_distribution = np.zeros((masked.shape[0], number_of_bins))
        for angle in range(-180 // factor, 180 // factor):
            phase_distribution[:, angle + 180 // factor] = np.sum(masked == angle, axis=1)
        mean = phase_distribution.mean(axis=1)
        mean = np.where(mean != 0, mean, 1)
        # phase_distribution -= mean[:, None]
        phase_distribution = phase_distribution / mean[:, None]
        if do_archive:
            archive.loc[unit, idx[unit_keys, :]] = np.reshape(phase_distribution, -1)
        if save or show:
            for row, vHIP_pad in enumerate(vHIP_pads):
                fig = plt.figure(figsize=(5, 5))
                plt.bar(np.arange(-180 // factor, 180 // factor), phase_distribution[row], width=1)
                if save:
                    plt.savefig(file_name + 'unit_' + str(unit) + '_pad_' + str(vHIP_pad) + '.jpg')
                if show:
                    if unit != -1:
                        plt.title('phase plot unit ' + str(unit) + ', pad ' + str(vHIP_pad))
                    else:
                        plt.title('phase plot all mua, pad ' + str(vHIP_pad))
                    plt.show()
                plt.close(fig)

            fig = plt.figure(figsize=(5, 5))
            plt.bar(np.arange(-180 // factor, 180 // factor), phase_distribution.mean(axis=0), width=1)
            if save:
                plt.savefig(file_name + '_unit_' + str(unit) + 'all_pads.jpg')
            if show:
                plt.title('phase plot all pads, unit ' + str(unit))
                plt.show()
            plt.close(fig)
    return archive
