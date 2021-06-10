import trajectory_process as tp
import pickle5 as pkl
import os

data_folder = 'E:/anxiety_ephys/'
animal = '011'
session = '2021-03-13_mBWfus011_EZM_ephys'
mtriggerfile = data_folder + animal + '/' + session + '/log.txt'
behavior = None
if session[-7] == 'M':
    behavior = 'ezm'
with open(mtriggerfile, 'r') as file:
    data = file.read().replace('\n', ' ')
point = data.index('.')
behavior_trigger = float(data[point - 2:point + 3])
duration = 60

events = tp.traj_process(animal, session, behavior_trigger, duration, behavior)
with open(data_folder + animal + '/' + session + '/ephys_processed/' + session + '_events.pkl', 'wb') as f:
    pkl.dump(events, f)
