import trajectory_process as tp
import pickle5 as pkl
import os

data_folder = 'E:/anxiety_ephys/'
animals = os.listdir(data_folder)
for animal in animals:
    sessions = os.listdir(data_folder + animal)
    if 'circus' in sessions:
        sessions.remove('circus')
    for session in sessions:
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
