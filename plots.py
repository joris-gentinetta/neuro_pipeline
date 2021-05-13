import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import math
from matplotlib import cm
import matplotlib as mpl
import copy

make_path_visible = 0.0001


def plot_classic(environment, plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
                 cluster_names, sigma=10, minp=0, maxp=95, n=150, show=False, save=True):
    movement = events['movement']
    mode = 'classic'
    file_name = plot_folder + experiment_name + '_' + mode + '_sigma' + str(sigma) + '_n' + str(n) + '_minp' + str(
        minp) + '_maxp' + str(maxp) + '_'
    data = np.empty(raw_data.shape, dtype=np.float32)

    for i in range(raw_data.shape[0]):
        data[i] = gaussian_filter1d(raw_data[i], sigma=sigma)
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

    cmap = copy.copy(mpl.cm.get_cmap('jet'))
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

        vmin = np.percentile(grid[:, :, i][np.where(grid[:, :, i] > 0)], minp)
        vmax = np.percentile(grid[:, :, i][np.where(grid[:, :, i] > 0)], maxp)

        fig = plt.figure(figsize=(5, 5))

        im = plt.imshow(np.ma.masked_where(grid[:, :, i] == 0, grid[:, :, i]).T, cmap=cmap, origin='upper',
                        interpolation='none', vmin=vmin, vmax=vmax)
        if environment == 'EZM':
            plt.imshow(mask, cmap='Greys', origin='upper',
                       interpolation='none', alpha=0.1)

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

    unit_sum = np.sum(grid[:, :, 1:], axis=2)
    vmin = np.percentile(unit_sum[np.where(unit_sum > 0)], minp)
    vmax = np.percentile(unit_sum[np.where(unit_sum > 0)], maxp)

    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(np.ma.masked_where(unit_sum == 0, unit_sum).T, cmap=cmap, origin='upper', interpolation='none',
                    vmin=vmin, vmax=vmax)
    if environment == 'EZM':
        plt.imshow(mask, cmap='Greys', origin='upper',
                   interpolation='none', alpha=0.1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save: plt.savefig(file_name + 'all_units' + '.jpg')
    plt.title('firing rate all units')
    if show:
        plt.show()
    plt.close(fig)
    return


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
    for i in range(data.shape[0]):
        xyv[2] = data[i][physio_trigger + off: xyv.shape[1] + physio_trigger + off] + make_path_visible

        for x in range(n):
            for y in range(n):
                boolean = np.logical_and(np.logical_and(xyv[0] > x / n * sx, xyv[0] < (x + 1) / n * sx),
                                         np.logical_and(xyv[1] > y / n * sy, xyv[1] < (y + 1) / n * sy))
                bsum = np.sum(boolean)
                if bsum != 0:
                    grid[x, y, i] = np.sum(xyv[2][boolean]) / bsum

        vmin = np.percentile(grid[:, :, i][np.where(grid[:, :, i] > 0)], minp)
        vmax = np.percentile(grid[:, :, i][np.where(grid[:, :, i] > 0)], maxp)

        fig = plt.figure(figsize=(5, 5))
        im = plt.imshow(np.ma.masked_where(grid[:, :, i] == 0, grid[:, :, i]).T, cmap=cmap, origin='upper',
                        interpolation='none', vmin=vmin, vmax=vmax)
        if environment == 'EZM':
            plt.imshow(mask.T, cmap='Greys', origin='upper',
                       interpolation='none', alpha=0.1)
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

    unit_sum = np.sum(grid[:, :, 1:], axis=2)
    vmin = np.percentile(unit_sum[np.where(unit_sum > 0)], minp)
    vmax = np.percentile(unit_sum[np.where(unit_sum > 0)], maxp)

    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(np.ma.masked_where(unit_sum == 0, unit_sum).T, cmap=cmap, origin='upper', interpolation='none',
                    vmin=vmin, vmax=vmax)
    if environment == 'EZM':
        plt.imshow(mask, cmap='Greys', origin='upper',
                   interpolation='none', alpha=0.1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save:
        plt.savefig(file_name + 'all_units' + '.jpg')
    plt.title('firing rate all units')
    if show:
        plt.show()
    plt.close(fig)
    return


def plot_circle(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger, cluster_names,
                n=360, sigma=-1, show=False, save=True):
    movement = events['movement']
    mode = 'circle'
    if sigma < 0:
        sigma = n / 8 / 4
    file_name = plot_folder + experiment_name + '_' + mode + '_sigma' + str(sigma) + '_n' + str(n) + '_'

    data = raw_data
    grid = np.zeros((data.shape[0], n), dtype=np.float32)
    xyv = np.empty(
        (4, min(len(movement['calib_traj_x'].index) - video_trigger - off, data.shape[1] - physio_trigger - off)))
    xyv[0] = movement['calib_traj_x'][video_trigger + off: xyv.shape[1] + video_trigger + off]
    xyv[1] = movement['calib_traj_y'][video_trigger + off: xyv.shape[1] + video_trigger + off]
    xyv[1] -= 5
    sx = 400
    sy = 400
    middle_x = sx / 2
    middle_y = sy / 2
    xyv[0] -= middle_x
    xyv[1] -= middle_y
    xyv[1] *= -1
    xyv[3] = np.arctan2(xyv[1], xyv[0])
    xyv[3] += (xyv[3] < 0) * 2 * math.pi

    for i in range(data.shape[0]):
        xyv[2] = data[i][physio_trigger + off: xyv.shape[1] + physio_trigger + off]

        for r in range(n):
            boolean = np.logical_and(xyv[3] > r * math.pi * 2 / n, xyv[3] < (r + 1) * math.pi * 2 / n)
            bsum = np.sum(boolean)
            if bsum != 0:
                grid[i, r] = np.sum(xyv[2][boolean]) / bsum
        grid[i] = gaussian_filter1d(grid[i], sigma=sigma, mode='wrap')
        fig = plt.figure(figsize=(5, 5))
        colors = cm.jet(plt.Normalize()(grid[i]))
        for quadrant in range(4):
            colors[(40 + quadrant * 90) * n // 360] = [0, 0, 0, 1]
        plt.pie(np.ones(n), colors=colors)
        plt.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=grid[i].min(), vmax=grid[i].max()), cmap='jet'),
                     shrink=0.6)
        my_circle = plt.Circle((0, 0), 0.8, color='white')
        fig.gca().add_artist(my_circle)
        if save:
            plt.savefig(file_name + str(cluster_names[i]) + '.jpg')
        if i != 0:
            plt.title('firing rate unit ' + str(cluster_names[i]))
        else:
            plt.title('firing rate all mua')
        if show:
            plt.show()
        plt.close(fig)

    unit_sum = np.sum(grid[1:], axis=0)
    colors = cm.jet(plt.Normalize()(unit_sum))
    for quadrant in range(4):
        colors[(40 + quadrant * 90) * n // 360] = [0, 0, 0, 1]
    fig = plt.figure(figsize=(5, 5))
    fig.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=unit_sum.min(), vmax=unit_sum.max()), cmap='jet'),
                 shrink=0.6)
    plt.pie(np.ones(n), colors=colors)
    my_circle = plt.Circle((0, 0), 0.8, color='white')
    fig.gca().add_artist(my_circle)
    if save:
        plt.savefig(file_name + 'all_units' + '.jpg')
    plt.title('firing rate all units')
    if show:
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

    unit_sum = np.sum(grid[:, :, 1:], axis=2)
    vmin = np.percentile(unit_sum[np.where(unit_sum > 0)], minp)
    vmax = np.percentile(unit_sum[np.where(unit_sum > 0)], maxp)

    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(np.ma.masked_where(unit_sum == 0, unit_sum).T, cmap=cmap, origin='upper', interpolation='none',
                    vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save:
        plt.savefig(file_name + 'all_units' + '.jpg')
    plt.title('firing rate all units')
    if show:
        plt.show()
    plt.close(fig)
    return


def plot_transitions(plot_folder, experiment_name, raw_data, events, cluster_names, video_trigger, mode,
                     plotmode='percent', n=200, m=5, show=False, save=True):
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

    unit_sum = sum(grid[1:])
    sum_mean = np.mean(unit_sum, axis=0)
    sum_std = np.std(unit_sum, axis=0)
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
    return


def plot_arms(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger,
              cluster_names, transition_size=2, minp=0, maxp=90, n=150, show=False, save=True):
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

        fig = plt.figure(figsize=(5, 5))

        for quadrant in range(8):
            valid_values_in_quadrant = grid[:, :, i][np.logical_and(masks[quadrant] == 1, grid[:, :, i] != 0)]
            # fig=plt.figure()
            # plt.imshow(grid[:,:,i].T)
            # plt.imshow(masks[quadrant].T, alpha=0.5)
            # plt.show()
            # plt.close(fig)
            ROI[i, quadrant] = (valid_values_in_quadrant.mean() - grid[:, :, i][grid[:, :, i] != 0].mean()) / \
                               grid[:, :, i][grid[:, :, i] != 0].mean() * 100

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

    fig = plt.figure(figsize=(5, 5))

    ROImean = np.mean(ROI[1:], axis=0)
    toplot = ROImean[[0, 1, 2, 3, 4, 6, 5, 7]]
    plt.bar(np.arange(8), toplot, width=1)
    if save:
        plt.savefig(file_name + 'all_units' + '.jpg')
    plt.title('firing rate all units')
    if show:
        plt.show()
    plt.close(fig)
    return


def plot_corners(plot_folder, experiment_name, raw_data, events, video_trigger, off, physio_trigger, cluster_names,
                 n=5, show=False, save=True):
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
        ROI = np.zeros((data.shape[0], 9))
        takenfrom = [(n - 1, 0), (0, 0), (0, n - 1), (n - 1, n - 1), (n - 1, 1), (1, 0), (0, 1), (1, n - 1), (1, 1)]
        for index in range(9):
            ROI[i, index] = (grid[:, :, i][takenfrom[index]] - grid.mean()) * 100 / grid.mean()

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

    unit_sum = (np.mean(ROI[1:], axis=0) - grid.mean()) * 100 / grid.mean()

    fig = plt.figure(figsize=(5, 5))
    plt.bar(np.arange(9), unit_sum)
    if save:
        plt.savefig(file_name + 'all_units' + '.jpg')
    plt.title('firing rate all units')
    if show:
        plt.show()
    plt.close(fig)
    return
