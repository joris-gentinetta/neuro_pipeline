import copy
import os
import winsound
from time import sleep

import numpy as np
import pandas as pd


# audio alert used to alert the user when a script is done
# freq: Hz, duration: milliseconds
def alert(freq=700, duration=600, number_of_beeps=4):
    for _ in range(number_of_beeps):
        winsound.Beep(freq, duration)
        sleep(0.2)


# creates all the subdirectories of the circus folder
def create_directories(target_folder):
    os.mkdir(target_folder)
    os.mkdir(target_folder + 'dat_files')
    os.mkdir(target_folder + 'vHIP_phase')
    os.mkdir(target_folder + 'dat_files_mod')
    os.mkdir(target_folder + 'mPFC_raw')
    os.mkdir(target_folder + 'vHIP_raw')
    os.mkdir(target_folder + 'mPFC_spike_range')
    os.mkdir(target_folder + 'movement_files')
    os.mkdir(target_folder + 'transition_files')
    os.mkdir(target_folder + 'spikes_50')
    os.mkdir(target_folder + 'spikes_20000')
    os.mkdir(target_folder + 'utils')


# columns: frames (50Hz), first row: x coordinates, second row: y coord, rest: units
def create_aligned(spikes_50, xy, max_duration, make_path_visible):
    frame_rate = 50
    aligned = np.empty(
        (2 + spikes_50.shape[0], min(xy.shape[1], spikes_50.shape[1])),
        dtype=np.float32)

    aligned[0] = xy[0][: aligned.shape[1]]  # x coordinates
    aligned[1] = xy[1][: aligned.shape[1]]  # y coordinates
    aligned[2:] = spikes_50[:, : aligned.shape[1]] + make_path_visible

    # crop to desired length
    if max_duration >= 0:
        aligned = aligned[:, :max_duration * frame_rate]
    else:
        aligned = aligned[:, max_duration * frame_rate:]
    return aligned


# creates the empty archive DataFrame
def create_archive(vHIP_pads, cluster_names, number_of_bins_transitions, number_of_bins_phase, number_of_bins_isi):
    level_1 = ['characteristics' for _ in range(17)] \
              + ['mean_waveform' for _ in range(60)] \
              + ['isi' for _ in range(number_of_bins_isi)] \
              + ['ROI_EZM' for _ in range(8)] \
              + ['ROI_OF' for _ in range(9)] \
              + ['open_closed_entrytime' for _ in range(number_of_bins_transitions)] \
              + ['open_closed_exittime' for _ in range(number_of_bins_transitions)] \
              + ['closed_open_entrytime' for _ in range(number_of_bins_transitions)] \
              + ['closed_open_exittime' for _ in range(number_of_bins_transitions)] \
              + ['lingering_entrytime' for _ in range(number_of_bins_transitions)] \
              + ['lingering_exittime' for _ in range(number_of_bins_transitions)] \
              + ['prolonged_open_closed_entrytime' for _ in range(number_of_bins_transitions)] \
              + ['prolonged_open_closed_exittime' for _ in range(number_of_bins_transitions)] \
              + ['prolonged_closed_open_entrytime' for _ in range(number_of_bins_transitions)] \
              + ['prolonged_closed_open_exittime' for _ in range(number_of_bins_transitions)] \
              + ['withdraw_entrytime' for _ in range(number_of_bins_transitions)] \
              + ['withdraw_exittime' for _ in range(number_of_bins_transitions)] \
              + ['nosedip_starttime' for _ in range(number_of_bins_transitions)] \
              + ['nosedip_stoptime' for _ in range(number_of_bins_transitions)]

    for pad in vHIP_pads:
        level_1 += ['theta_phase_OFT_' + str(pad) for _ in range(number_of_bins_phase)]
        level_1 += ['theta_phase_EZM_' + str(pad) for _ in range(number_of_bins_phase)]
        level_1 += ['theta_phase_before_' + str(pad) for _ in range(number_of_bins_phase)]
        level_1 += ['theta_phase_after_' + str(pad) for _ in range(number_of_bins_phase)]

    five_sec_range = list(np.arange(number_of_bins_transitions))
    transition_ranges = copy.copy(five_sec_range)
    for i in range(13):
        transition_ranges.extend(five_sec_range)
    degree360 = list(np.arange(number_of_bins_phase))
    phase_ranges = copy.copy(degree360)
    for i in range(4 * len(vHIP_pads) - 1):
        phase_ranges.extend(degree360)
    level_2 = ['pad', 'data_row', 'amplitude', 'overall_firing_rate', 'purity', 'ezm_closed_score',
               'ezm_transition_score', 'ezm_closed', 'ezm_transition', 'of_corners_score',
               'of_middle_score', 'of_corners', 'of_middle', 'mean_before', 'mean_after', 'mean_EZM', 'mean_OFT'] \
              + [i for i in range(60)] \
              + [i for i in range(number_of_bins_isi)] \
              + [i for i in range(8)] \
              + [i for i in range(9)] \
              + transition_ranges \
              + phase_ranges
    tuples = list(zip(level_1, level_2))
    columns = pd.MultiIndex.from_tuples(tuples)
    archive = pd.DataFrame(index=cluster_names, columns=columns, dtype=np.float32)
    return archive
