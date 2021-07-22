import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed


def import_excel(xl_name, save, export):
    xl = pd.ExcelFile(xl_name)
    # xl = pd.read_excel(xl_name, header=None, skiprows=34)
    xl_sheets = xl.sheet_names

    # Put all in one data frame
    df = pd.concat([xl.parse(xl_sheets[0]), xl.parse(xl_sheets[1]), xl.parse(xl_sheets[2]), xl.parse(xl_sheets[3])],
                   axis=0)
    if export:
        df.to_pickle(f'{save}/all_groups_distance_moved.pkl')
        for k in xl_sheets:
            d = xl.parse(k)
            d.to_pickle(f'{save}/{k}.pkl')

    return df


def import_pickle(file):
    p = pd.read_pickle(file)
    return p


dir_path = os.getcwd()  # get current working dir
path = f'{dir_path}/keshia/20210622/analysis/'

file_name = 'all_groups.xlsx'
save_path = f'{path}/pkl'
xl_file = f'{path}{file_name}'

# Load pickled data
all_data = import_pickle(f'{save_path}/all_groups_distance_moved.pkl')
data = []
for k in range(4):
    data.append(import_pickle(f'{save_path}/G{k+1}.pkl'))

# Load Well Labels
well_labels = pd.read_excel(f'{path}/well_labels.xlsx', nrows=4).T
labels = well_labels.keys()[1:]
treatments = ['control', 'cu1uM', 'cu10uM', 'cu50uM', 'cu100um',
              'neo50uM', 'neo100uM', 'neo200um', 'neo400uM']

mapping = dict.fromkeys([0, 1, 2, 3])
for i in mapping:
    mapping[i] = dict.fromkeys(treatments)
mapping_data = mapping.copy()

# Find mapping from well label to treatments
for k in range(well_labels.shape[1]):  # loop through all groups
    for tr in treatments:  # loop through all treatments
        idx = well_labels[k] == tr
        mapping[k][tr] = well_labels[k].index[idx]
        mapping_data[k][tr] = pd.concat([data[k]['Recording time'], data[k]['Stimulus'], data[k][mapping[k][tr]]], axis=1)
# Combine all groups into one data frame
df = dict.fromkeys(treatments)
for tr in treatments:
    df[tr] = pd.concat([mapping_data[0][tr], mapping_data[1][tr], mapping_data[2][tr], mapping_data[3][tr]], axis=1)

embed()
exit()
