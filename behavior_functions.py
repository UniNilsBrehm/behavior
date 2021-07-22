import csv
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from IPython import embed


def import_pickle(file):
    p = pd.read_pickle(file)
    return p


def import_csv(file):
    p = pd.read_csv(file, sep=';', header=0).fillna(value='nofish')
    return p


def pandas_import_excel(xl_name):
    xl = pd.ExcelFile(xl_name, header=None, skiprows=34)
    # xl = pd.read_excel(xl_name, header=None, skiprows=34)
    xl_sheets = xl.sheet_names
    sheet = xl.parse(xl_sheets[0])


def pandas_import_csv(csv_name):
    # READ HEADER:
    h = pd.read_csv(csv_name, sep=',', header=0, quoting=2, na_values='-', nrows=1, encoding='utf16')
    number_of_header_lines = int(h.keys()[1])
    header = pd.read_csv(csv_name, sep=',', header=0, quoting=2, na_values='-', nrows=number_of_header_lines-3
                         , encoding='utf16')
    skip = list(np.arange(0, number_of_header_lines-2))
    skip.append(number_of_header_lines-1)
    cf = pd.read_csv(csv_name, sep=',', header=0, skiprows=skip, quoting=2, na_values='-', encoding='utf16')
    return cf, header


def pandas_import_protocol(csv_name):
    skip = list(np.arange(0, 34))
    skip.append(35)
    cf = pd.read_csv(csv_name, sep=',', header=0, skiprows=skip, quoting=2, na_values='-', encoding='utf16')
    return cf


def compute_distance_moved(x, y):
    # input: two lists of Y and Y coordinates per sample point
    d_moved = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return d_moved


# DATA IMPORT ---------------------------------------------------
def convert_raw_data(path):
    raw_data_path = f'{path}/rawdata/txt/'
    save_path = f'{path}/analysis/'
    filenames = next(os.walk(f'{raw_data_path}'), (None, None, []))[2]  # [] if no file

    # get data files (txt files that start with 'Track')
    data_files_names = [i for i in filenames if i.startswith('Track')]

    # Get protocol file (We will take just the first one since all are the same)
    protocol_file_name = [i for i in filenames if i.startswith('Trial')][0]

    # Open CSV data files
    data_set = {}  # dict of all panda data frames
    data_header = {}
    data_list = []

    for count, val in enumerate(data_files_names):
        well_nr = val[-16:-14]  # This the number of the specific well (exp. 'D4')
        csv_path = f'{raw_data_path}/{data_files_names[count]}'
        data_set[well_nr], data_header[well_nr] = pandas_import_csv(csv_path)
        data_list.append(data_set[well_nr])

    # Open Protocol CSV file
    protocol_path = f'{raw_data_path}/{protocol_file_name}'
    protocol = pandas_import_protocol(protocol_path)
    # protocol = pd.read_csv('protocol.csv', sep=';', header=0, quoting=2, na_values='-')

    # ANALYSIS -------------------------------------------------------
    sampling_rate = 30  # in Hz
    time_after_stimulus = 1  # in secs
    samples_after_stimulus = int(sampling_rate * time_after_stimulus)

    # Compute Distance Moved ===========================================================================================
    distance_moved = {}
    for k in data_set:
        X = data_set[k]['X center']
        Y = data_set[k]['Y center']
        dummy = compute_distance_moved(X, Y)
        distance_moved[k] = np.insert(dummy, 0, 0)  # add a zero since there is no diff at the start

    # Find Stimulus Time Points
    idx_stimulus = protocol['Action'].notnull()
    stimuli_non_unique = pd.concat([protocol[idx_stimulus]['Recording time'], protocol[idx_stimulus]['Action']], axis=1)
    stimuli = stimuli_non_unique.drop_duplicates(subset='Recording time')

    # PUT ALL DATA INTO ONE EXCEL FILE
    # Time; Stimulus, Distance Moved (A1) ...

    data_wells = list(distance_moved.keys())
    rows = len(distance_moved[data_wells[0]])
    cols = len(data_wells)
    data_export = np.zeros([rows, cols])

    # Add Recording Time
    recording_time = data_set[data_wells[0]]['Recording time']
    recording_time_df = pd.DataFrame(recording_time, columns=['Time'])
    # data_export[:, 0] = recording_time

    # Add Stimulus onset times
    index_stimulus = []
    stimulus_col = [np.nan] * rows
    for i in stimuli['Recording time']:
        index_stimulus.append(np.where(recording_time == i))

    count2 = 0
    for i in stimuli['Action']:
        stimulus_col[index_stimulus[count2][0][0]] = i
        count2 += 1
    stimulus_col = pd.DataFrame(stimulus_col, columns=['Stimulus'])

    count = 0
    for k in distance_moved:
        data_export[:, count] = distance_moved.get(k)
        count += 1

    # Convert to Panda Data Frame
    col_names = ['Time', 'Stimulus'] + data_wells
    data_frame = pd.DataFrame(data_export, columns=data_wells)
    data_frame = pd.concat([recording_time, stimulus_col, data_frame], axis=1)

    # Save as excel files (Takes some time...)
    print('FOUND STIMULI:')
    print(stimuli)
    print('--------------------------------------')
    print('PLEASE WAIT WHILE SAVING DATA TO DISC')
    with pd.ExcelWriter(f'{save_path}/behavior_distance_moved.xlsx') as writer:
        data_frame.to_excel(writer, sheet_name='Distance_Moved')
        protocol.to_excel(writer, sheet_name='Protocol')
    # data_tapping_export.to_excel("Tapping_DistanceMoved.xlsx")
    print('SAVED DATA TO EXCEL FILE')

    # Save to pickle file
    # np.save(f'{save_path}/raw_data.pkl', data_set)
    pickle.dump(data_set, open(f'{save_path}/raw_data.pkl', "wb"))
    data_frame.to_pickle(f'{save_path}/behavior_distance_moved.pkl')
    protocol.to_pickle(f'{save_path}/behavior_protocol.pkl')
    print('SAVED DATA TO PICKLE FILE')
