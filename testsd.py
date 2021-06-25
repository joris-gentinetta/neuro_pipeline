from time import sleep
sleep(10)



######################################
animal = '109'
alert_when_done = False
######################################

import numpy as np
from load_intan_rhd_format.load_intan_rhd_format import read_data
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import subprocess
import shlex
from natsort import natsorted
import pickle5 as pkl
from utils import alert
from scipy.signal import hilbert
from scipy.signal import butter, sosfiltfilt

butter_order = 3
theta_band = [4, 12]
spike_band = [300, 3000]
sampling_rate = 20000
frame_rate = 50

start_time = time.time()
# folder preparation
animal_folder = r'E:/anxiety_ephys/' + animal + '/'
sub_folder = 'circus'
target_folder = animal_folder + sub_folder + '/'
experiment_names = natsorted(os.listdir(animal_folder))
if 'circus' in experiment_names:
    experiment_names.remove('circus')
data_folders = [animal_folder + experiment_name + '/' for experiment_name in experiment_names]
templates = r'E:/pipe/spykingcircus_sorter_2/templates/'
circus_entrypoint = target_folder + 'dat_files/' + experiment_names[0] + '_0.dat'


mouse_is_late = {'2021-02-19_mBWfus010_EZM_ephys': 70,
                 '2021-02-19_mBWfus009_EZM_ephys': 42,
                 '2021-02-26_mBWfus012_EZM_ephys': 35}
accomodation = 20

logbook = np.zeros(len(experiment_names))
index = 1
experiment_name = '2021-02-19_mBWfus009_EZM_ephys'
eventfile = animal_folder + experiment_name + '/ephys_processed/' + experiment_name + '_events.pkl'
#get events/movement
with open(eventfile, 'rb') as f:
    events = pkl.load(f)
movement = events['movement']
xy = np.array([movement['calib_traj_x'], movement['calib_traj_y']], dtype=np.float32)

movement_trigger_file = animal_folder + '/' + experiment_name + '/log.txt'
with open(movement_trigger_file, 'r') as file:
    data = file.read().replace('\n', ' ')
point = data.index('.')
movement_trigger = int(float(data[point - 2:point + 3]) * frame_rate)

if experiment_name in mouse_is_late:
    off = (mouse_is_late[experiment_name] + accomodation) * frame_rate
else:
    off = 0
movement_trigger += off
xy = xy[:,movement_trigger:]
np.save(target_folder + 'movement_files/' + experiment_name, xy.astype(np.float32))
