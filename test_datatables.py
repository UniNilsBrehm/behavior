from datatable import (dt, f, by, ifelse, update, sort,
                       count, min, max, mean, sum, rowsum)
from IPython import embed
import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_distance_moved(x, y):
    # input: two lists of Y and Y coordinates per sample point
    # Compute Euclidian distance
    d_moved = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return d_moved


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SETTINGS
smoothing = 'txt'
dir_path = f'C:/Uni Freiburg/Behavior/kurs2021/'  # get current working dir
all_groups = ['Group1', 'Group2', 'Group3', 'Group4']
final_data = dict.fromkeys(all_groups)
time_course_all = dict.fromkeys(all_groups)
remove_fish_in_time_course = False
remove_fish_in_distance = True
remove_fish_in_group_well = [
    ['A7'],
    ['A1', 'A3', 'A4', 'B3', 'B4', 'B6', 'B7', 'C1', 'C5', 'F4'],
    ['A3', 'A5', 'B1', 'B3', 'C5', 'D8'],
    ['B4', 'B6', 'C6', 'C7', 'D7', 'D8', 'F7']
                             ]
for group_i, group_v in enumerate(tqdm(all_groups)):
    raw_data_path = f'rawdata/{group_v}/{smoothing}'
    save_path = f'{dir_path}/analysis/{group_v}'
    filenames = next(os.walk(f'{dir_path}{raw_data_path}'), (None, None, []))[2]  # [] if no file
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # GET DATA

    # get data files (txt files that start with 'Track')
    data_files_names = [i for i in filenames if i.startswith('Track')]
    # Get protocol file (We will take just the first one since all are the same)
    protocol_file_name = [i for i in filenames if i.startswith('Trial')][0]

    # Load txt files using datatable package
    file_path = f'{dir_path}{raw_data_path}/'
    for i, v in enumerate(tqdm(data_files_names)):
        # Find number of head lines (stored in each txt file)
        skip_rows = dt.fread(f'{file_path}{data_files_names[i]}', max_nrows=1, fill=True)[:, 1]
        # Find well labels (A1...F8)
        area_name = dt.fread(f'{file_path}{data_files_names[i]}', max_nrows=10, fill=True)[5, 1]
        # Convert to int
        skip_rows[:, :] = dt.int32
        df = dt.fread(f'{file_path}{data_files_names[i]}', na_strings=['-'], fill=True, skip_to_line=skip_rows[0, 0]-1)
        # delete units row
        del df[0, :]
        # Convert strings to float values
        # df[:, ['X center', 'Y center', 'Distance moved', 'Velocity']] = dt.float32
        # Compute distance moved
        # distance_moved = compute_distance_moved(df['X center'].to_numpy()[:, 0], df['Y center'].to_numpy()[:, 0])
        # Select Distance moved
        if i == 0:
            distance = dt.Frame(df[:, ['Recording time', 'Distance moved']])
            distance.names = ['Time', area_name]
            distance[:, :] = dt.float32
        else:
            distance[area_name] = df['Distance moved']
            distance[area_name] = dt.float32

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # LOAD PROTOCOL AND FIND STIMULI

    # Load txt files
    # Find number of head lines (stored in each txt file)
    skip_rows = dt.fread(f'{file_path}{protocol_file_name}', max_nrows=1, fill=True)[:, 1]

    # Convert to int
    skip_rows[:, :] = dt.int32
    protocol = dt.fread(f'{file_path}{protocol_file_name}', na_strings=['-'], fill=True, skip_to_line=skip_rows[0, 0]-1)

    # delete units row
    del protocol[0, :]

    # Find stimulus tags and remove the last one, which should be "Stop track"
    tags = protocol[:-1, ['Recording time', 'Action'], by((f.Action != '') & (f.Event =='becomes active'))].to_list()
    stimulus_tags = dict.fromkeys(np.array(tags[2])[tags[0]])
    for i, tag_name in enumerate(stimulus_tags.keys()):
        stimulus_tags[tag_name] = np.array(tags[1], dtype='float32')[tags[0]][i]

    # Load treatments
    treatments = dt.fread(f'{dir_path}/analysis/wells.csv', na_strings=[''], fill=True, max_nrows=4)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # DATA MANIPULATIONS

    # Cut out stimulus time points and sum distance moved
    # Then put all data into one data frame in long format (stacked data)
    time_interval = [60, 30, 30] + [1] * (len(stimulus_tags)-3)  # in secs; has to be set manually
    entry_id = 0
    list_container = []
    cc = 0
    time_course = {}
    for st_i, stimulus_name in enumerate(stimulus_tags):
        treatments_group = treatments.copy()
        # print(f'{stimulus_name}: {stimulus_tags[stimulus_name]}')
        dummy = distance[:, :, by((f.Time >= stimulus_tags[stimulus_name]) & (f.Time <= stimulus_tags[stimulus_name] +
                                                                              time_interval[cc]))]
        # Now sum distance over time:
        summed = dummy[dummy[:, 0], 2:].sum()
        # Store time course:
        tc = dummy[dummy[:, 0], 2:]

        # Delete miss-tracked animals:
        if remove_fish_in_time_course:
            del treatments_group[:, remove_fish_in_group_well[group_i]]
            del tc[:, remove_fish_in_group_well[group_i]]
        if remove_fish_in_distance:
            del summed[:, remove_fish_in_group_well[group_i]]

        # change labels to A1(treatment)....:
        if remove_fish_in_time_course:
            labels = []
            for k, v in enumerate(list(tc.names)):
                labels.append(f'{v}({list(treatments_group[int(group_v[-1]) - 1, 1:].to_numpy()[0])[k]})')
            tc.names = labels
        time_course[stimulus_name] = tc
        for k in summed.keys():
            if stimulus_name.startswith('Tap'):
                trial = stimulus_name[-1]
                new_stimulus_name = stimulus_name[:-2]
                new_row = [entry_id, f'{group_v[-1]}_{k}', int(group_v[-1]), k, treatments[int(group_v[-1]) - 1, k],
                           new_stimulus_name,
                           time_interval[cc], summed[0, k], trial]
            else:
                new_row = [entry_id, f'{group_v[-1]}_{k}', int(group_v[-1]), k, treatments[int(group_v[-1])-1, k], stimulus_name,
                           time_interval[cc], summed[0, k], 1]
            list_container.append(new_row)
            entry_id += 1
        cc += 1
    data_long = dt.Frame(np.array(list_container))
    data_long.names = ['ID', 'FishID', 'Group', 'Well', 'Treatment', 'Stimulus', 'Duration', 'Distance', 'Trial']
    data_long['Distance'] = dt.float32
    data_long['Duration'] = dt.float32
    data_long['Trial'] = dt.float32
    final_data[group_v] = data_long
    time_course_all[group_v] = time_course

# Combine all groups into one data frame
data_all_groups = dt.rbind(final_data['Group1'], final_data['Group2'], final_data['Group3'], final_data['Group4'])
# Store data to HDD:
pickle.dump(data_all_groups, open(f'{dir_path}/analysis/data/summed_distances.pkl', "wb"))
pickle.dump(time_course_all, open(f'{dir_path}/analysis/data/time_courses.pkl', "wb"))

print('All data stored to HDD')


