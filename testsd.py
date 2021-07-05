from time import sleep

sleep(10)

######################################
animal = '112'
alert_when_done = False
######################################

import time
import pandas as pd

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

pisi = target_folder + 'utils/p_values_isi.pkl'
pearisi = target_folder + 'utils/pearson_coefficients_isi.pkl'

pwa = target_folder + 'utils/p_values_waveform.pkl'
pearwa = target_folder + 'utils/pearson_coefficients_waveform.pkl'

d = pd.read_pickle(pearwa)
print(d)
