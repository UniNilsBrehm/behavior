from datatable import (dt, f, by, ifelse, update, sort,
                       count, min, max, mean, sum, rowsum)
from IPython import embed
import os
import numpy as np


def compute_distance_moved(x, y):
    # input: two lists of Y and Y coordinates per sample point
    # Compute Euclidian distance
    d_moved = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return d_moved


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Just some testing
p = 'C:/Users/Nils/Desktop/test.csv'
df = dt.fread(p)

# WORKING WITH LONG FORMAT AND DATATABLE PACKAGE
# Find idx of all "tap4_x" stimuli:
idx = [x for x, v in enumerate(df['stimulus'].to_list()[0]) if v.startswith('tap4')]
tap4 = df[idx, :]

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Get mean distance for each stimuli and treatment group
means = df[:, dt.mean(f[:]), by('stimulus', 'treatment')]

# IMPORT TXT DATA FILES
group = 'Group2'
smoothing = 'txt'
dir_path = f'{os.getcwd()}/kurs2021/'  # get current working dir
raw_data_path = f'rawdata/{group}/{smoothing}'
save_path = f'{dir_path}/analysis/{group}'
filenames = next(os.walk(f'{dir_path}{raw_data_path}'), (None, None, []))[2]  # [] if no file

# get data files (txt files that start with 'Track')
data_files_names = [i for i in filenames if i.startswith('Track')]

# Get protocol file (We will take just the first one since all are the same)
protocol_file_name = [i for i in filenames if i.startswith('Trial')][0]

# Load txt files using datatable package
file_path = f'{dir_path}{raw_data_path}/'
for i, v in enumerate(data_files_names):
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

# Load protocol txt file
# Find number of head lines (stored in each txt file)
skip_rows = dt.fread(f'{file_path}{protocol_file_name}', max_nrows=1, fill=True)[:, 1]
# Convert to int
skip_rows[:, :] = dt.int32
protocol = dt.fread(f'{file_path}{protocol_file_name}', na_strings=['-'], fill=True, skip_to_line=skip_rows[0, 0]-1)
# delete units row
del protocol[0, :]
# Find stimulus tags
tags = protocol[:, ['Recording time', 'Action'], by(f.Action != '')].to_list()
stimulus_tags = [np.array(tags[1], dtype='float32')[tags[0]], np.array(tags[2])[tags[0]]]

# Cut out stimulus time points
time_interval = 1  # in secs
a = distance[:, :, by((f.Time >= stimulus_tags[0][10]) & (f.Time <= stimulus_tags[0][10] + time_interval))]
embed()
exit()

