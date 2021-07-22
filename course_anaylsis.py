
import csv
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
import matplotlib.cbook as cbook
import time
import progressbar
import matplotlib.pyplot as plt
from IPython import embed
from collections import defaultdict


'''
This script will load converted data from the Danio Vision Apparatus and creates several different figures.
For more detail please see the comments for each figure below.

Nils Brehm - 2021
'''

# FUNCTIONS ============================================================================================================


def plot_settings():
    # Font:
    # matplotlib.rc('font',**{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    # matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.size'] = 10

    # Ticks:
    matplotlib.rcParams['xtick.major.pad'] = '2'
    matplotlib.rcParams['ytick.major.pad'] = '2'
    matplotlib.rcParams['ytick.major.size'] = 4
    matplotlib.rcParams['xtick.major.size'] = 4

    # Title Size:
    matplotlib.rcParams['axes.titlesize'] = 10

    # Axes Label Size:
    matplotlib.rcParams['axes.labelsize'] = 10

    # Axes Line Width:
    matplotlib.rcParams['axes.linewidth'] = 1

    # Tick Label Size:
    matplotlib.rcParams['xtick.labelsize'] = 9
    matplotlib.rcParams['ytick.labelsize'] = 9

    # Line Width:
    matplotlib.rcParams['lines.linewidth'] = 1
    matplotlib.rcParams['lines.color'] = 'k'

    # Marker Size:
    matplotlib.rcParams['lines.markersize'] = 2

    # Error Bars:
    matplotlib.rcParams['errorbar.capsize'] = 0

    # Legend Font Size:
    matplotlib.rcParams['legend.fontsize'] = 6

    # Set pcolor shading
    matplotlib.rcParams['pcolor.shading'] = 'auto'

    return matplotlib.rcParams


def sort_stimuli_data(n, fish, stimulus_name, position, t):
    all_taps = dict.fromkeys(fish)
    for i_a, v_a in enumerate(fish):
        taps_pos = []
        for i in n[v_a].keys():
            if position == 'end':
                taps_pos.append(i.endswith(stimulus_name))
            else:
                taps_pos.append(i.startswith(stimulus_name))
        taps_keys = np.array(list(n[v_a].keys()))[taps_pos]
        all_taps[v_a] = [n[v_a].get(key) for key in taps_keys]

    # treatment_groups = dict.fromkeys(treats)
    # flat_treatment_groups = dict.fromkeys(treats)
    list_treatment_groups = []
    for g in t:
        # treatment_groups[g] = [v for k, v in all_taps.items() if k.startswith(g)]
        # flat_treatment_groups[g] = [item for sublist in treatment_groups[g] for item in sublist]
        list_treatment_groups.append([item for sublist in [v for k, v in all_taps.items() if k.startswith(g)] for item in sublist])
    return list_treatment_groups


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


def sort_data(data_checked, wells, window_time, sampling_rate, threshold, threshold2):
    # Rename well label to treatments
    labels = ['Time', 'Stimulus'] + list(wells.to_numpy())
    distance_moved = data_checked.set_axis(labels, axis=1)

    # Remove empty wells with ('nofish'):
    distance_moved = distance_moved.drop(columns='nofish')

    # Find Stimuli:
    tags = distance_moved['Stimulus'].notnull()
    stim_set = distance_moved[tags]
    idx_all_tags = stim_set.index  # All tags
    idx_tags = idx_all_tags[0:-1]  # Remove STOP RECORDING Tag
    stimulus_labels = stim_set[0:-1]['Stimulus'].to_list()

    # Mean of cols of different dfs:
    # pd.concat([a, a]).groupby(level=0).mean()

    # Cut out data corresponding to all the stimuli tags
    window_samples = int(window_time * sampling_rate)
    distance_moved_per_tag = []
    for k, v_tags in enumerate(idx_tags):
        # threshold = 200
        # idx_filter = distance_moved[v_tags:v_tags + window_samples].sum() > threshold
        # a = idx_filter.keys() == 'Time'
        # idx_filter = idx_filter[~a]
        # remove_idx = idx_filter.keys()[idx_filter]
        # filterd = distance_moved[v_tags:v_tags + window_samples].drop(columns=remove_idx)
        distance_moved_per_tag.append(distance_moved[v_tags:v_tags + window_samples])

    # Get all treatment names from data
    treatments = distance_moved.keys()[2:]

    # dict with unique treatment names as keys
    sorted_data = dict.fromkeys(treatments.unique())
    sorted_data_sum = dict.fromkeys(treatments.unique())

    for k, v_treatment in enumerate(treatments):
        # each element of that list corresponds to position in "stimulus_labels"
        sorted_data[v_treatment] = [[]] * len(idx_tags)
        sorted_data_sum[v_treatment] = [[]] * len(idx_tags)
        for i, w in enumerate(distance_moved_per_tag):
            sorted_data[v_treatment][i] = w[v_treatment]
            # Take the sum over time for each animal per stimulus
            # filter out summed distances that are over threshold
            dummy = w[v_treatment].sum()
            if dummy.size == 1:
                if dummy > threshold or dummy < threshold2:
                    dummy = np.nan
                sorted_data_sum[v_treatment][i] = dummy
            else:
                for h, g in enumerate(dummy):
                    if g > threshold or g < threshold2:
                        dummy[h] = np.nan
                sorted_data_sum[v_treatment][i] = dummy

    # Find repeats of stimuli
    sorted_stimuli = []
    for k, v in enumerate(stimulus_labels):
        pos = v.find('_')
        if pos > 0:
            sorted_stimuli.append(v[0:pos])
        else:
            sorted_stimuli.append(v)

    # This dict will contain for each unique stimulus name the repeated presentations
    stimuli_idx = dict.fromkeys(np.unique(sorted_stimuli))
    for k, v in enumerate(np.unique(sorted_stimuli)):
        stimuli_idx[v] = np.array(sorted_stimuli) == v

    # After this we have a dict with each unique treatment holding all the stimuli with their repetitions
    final_data = dict.fromkeys(treatments.unique())
    final_data_sum = dict.fromkeys(treatments.unique())
    for k, v in enumerate(treatments.unique()):
        final_data[v] = dict.fromkeys(np.unique(sorted_stimuli))
        final_data_sum[v] = dict.fromkeys(np.unique(sorted_stimuli))
        for i, w in enumerate(np.unique(sorted_stimuli)):
            final_data[v][w] = np.array(sorted_data[v])[stimuli_idx[w]]
            final_data_sum[v][w] = np.array(sorted_data_sum[v])[stimuli_idx[w]]

    treatment_groups = list(final_data.keys())

    return final_data, final_data_sum, treatment_groups, np.unique(sorted_stimuli)


def remove_miss_tracked_fish(original_data, miss_tracked_idx, wells):
    data_c = original_data.drop(columns=miss_tracked_idx)
    wells = wells.drop(index=miss_tracked_idx)
    return data_c, wells


def find_stimuli(data_input):
    # Find stimuli labels and time points in "distance_moved" data
    tags = data_input['Stimulus'].notnull()
    stim_set = data_input[tags]
    idx_all_tags = stim_set.index  # All tags
    idx_tags = idx_all_tags[0:-1]  # Remove STOP RECORDING Tag
    stimulus_labels = stim_set[0:-1]['Stimulus'].to_list()

    result = {'tags': tags, 'idx_all': idx_all_tags, 'idx': idx_tags, 'labels': stimulus_labels}
    return result


# ======================================================================================================================
# ======================================================================================================================
# SETTINGS:
# *********************************************
check_tracking = False

distance_moved_per_stimulus = False

boxplots = False

removing_miss_tracked_fish = False

plot_settings()
# *********************************************

# set sample rate of the recording:
sampling_rate = 30  # in Hz
# get current working dir
dir_path = os.getcwd()
# set path for data and exporting figures
path = f'{dir_path}/kurs2021/analysis/'
# set groups
groups = ['Group1', 'Group2', 'Group3', 'Group4']
remove_fish_in_group = [True, True, True, True]
remove_fish_in_group_well = [
    ['A7'],
    ['A1', 'A3', 'A4', 'B3', 'B4', 'B6', 'B7', 'C1', 'C5', 'F4'],
    ['A3', 'A5', 'B1', 'B3', 'C5', 'D8'],
    ['B4', 'B6', 'C6', 'C7', 'D7', 'D8', 'F7']
                             ]

# ======================================================================================================================
# START OF SCRIPT ======================================================================================================
# DATA HANDLING ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
window_time = [30, 30, 30, 1]  # in secs
print('Starting Analysis')
data_sorted = dict.fromkeys(groups)
data_sorted_sum = dict.fromkeys(groups)
treatment_names = dict.fromkeys(groups)
stimuli = dict.fromkeys(groups)

# VISUALLY CHECK TRACKING PERFORMANCE
if check_tracking:
    for kk, gg in enumerate(groups):
        print(' ---- ')
        print(f'Finished {kk} / {len(groups)}')
        print(' ---- ')

        # Load pickled data
        group = gg
        with open(f'{path}/{group}/raw_data.pkl', "rb") as input_file:
            raw_data = pickle.load(input_file)
        data = import_pickle(f'{path}/{group}/behavior_distance_moved.pkl')
        protocol = import_pickle(f'{path}/{group}/behavior_protocol.pkl')
        all_wells = pd.read_csv(f'{path}/wells.csv', sep=';', header=0, nrows=len(groups)).fillna(value='nofish')
        wells_csv = all_wells.loc[kk][1:]

        # Rename well label to treatments
        labels1 = ['Time', 'Stimulus'] + list(wells_csv.to_numpy())
        # find wells with no fish in it:
        idx_nofish = np.where(np.array(labels1[2:]) == 'nofish')

        # CHECK TRACKING:
        # find stimuli tags:
        stimuli = find_stimuli(data)
        idx_taps = stimuli['idx']

        # tap_checks = dict.fromkeys(np.array(stimuli['labels'])[idx_bool_taps])
        labels = np.array(stimuli['labels'])
        tag_checks = dict.fromkeys(labels)

        # GET RAW DATA SORTED
        t1 = time.time()
        for i, t in enumerate(idx_taps):
            checks = []
            if i > 2:
                window_samples = int(window_time[-1] * sampling_rate)
            else:
                window_samples = int(window_time[i] * sampling_rate)

            for cols in data.keys()[2:]:
                checks.append(raw_data[cols][t: t + window_samples])
            tag_checks[list(tag_checks.keys())[i]] = checks

        t2 = time.time()
        # PLOT ----------------------------------------------------------------------------------------------------
        nr_cols = 8
        nr_rows = 6
        bar2 = progressbar.ProgressBar(maxval=len(labels)).start()
        for name_k, name in enumerate(labels):
            fig, axs = plt.subplots(nr_rows, nr_cols)
            count = 0
            for rows in range(nr_rows):
                for cols in range(nr_cols):
                    x = tag_checks[name][count]['X center']
                    y = tag_checks[name][count]['Y center']

                    # Subtract the mean to center coordinates around zero
                    x = x - np.nanmean(x)
                    y = y - np.nanmean(y)
                    x_y_limit = 10

                    axs[rows, cols].plot(x, y, color='black', linewidth=0.5)
                    axs[rows, cols].axis('equal')
                    axs[rows, cols].set_xlim([-x_y_limit, x_y_limit])
                    axs[rows, cols].set_ylim([-x_y_limit, x_y_limit])
                    axs[rows, cols].set_xticks([])
                    axs[rows, cols].set_yticks([])

                    axs[rows, cols].text(0.5, 0.75, f'{labels1[2:][count]}', transform=axs[rows, cols].transAxes, fontsize=5,
                                         horizontalalignment='center', verticalalignment='center', alpha=0.3)
                    axs[rows, cols].text(0.5, 0.25, f'{data.keys()[2:][count]}', transform=axs[rows, cols].transAxes, fontsize=5,
                                         horizontalalignment='center', verticalalignment='center', alpha=0.3)

                    if count in idx_nofish[0]:
                        # Highlight wells with no fish
                        axs[rows, cols].patch.set_facecolor('red')
                        axs[rows, cols].patch.set_alpha(0.5)
                    count += 1

            # Assign well labels to the figure:
            for i in range(8):
                axs[0, i].text(0.5, 1.25, i + 1, transform=axs[0, i].transAxes, fontsize=14,
                               horizontalalignment='center', verticalalignment='center')

            row_names = ['A', 'B', 'C', 'D', 'E', 'F']
            for k in range(6):
                axs[k, 0].text(-0.25, 0.5, row_names[k], transform=axs[k, 0].transAxes, fontsize=14,
                               horizontalalignment='center', verticalalignment='center')

            # Save Fig:
            t3 = time.time()
            CM = 1 / 2.54  # centimeters in inches
            # Width x Height
            fig.set_size_inches(8 * 2 * CM, 6 * 2 * CM)
            fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.3, hspace=0.3)
            SAVE_PATH = f'{path}/figs/'
            # fig_name = f'Tracking_{stimulus_labels[kk]}_{window_time[kk]}secs_all'
            fig_name = f'Tracking_{gg}_{name}'
            # fig.savefig(f'{SAVE_PATH}tracking/{fig_name}.pdf')
            if name == 'Start track':
                fig.savefig(f'{SAVE_PATH}tracking/{gg}/AA_{fig_name}.jpg', dpi=300)
            else:
                fig.savefig(f'{SAVE_PATH}tracking/{gg}/{fig_name}.jpg', dpi=300)
            plt.close(fig)
            t4 = time.time()
            # print(f't1: {t2-t1}')
            # print(f't2: {t3-t2}')
            # print(f't3: {t4-t3}')
            bar2.update(name_k)

    print(' --- ')
    print('Saved all tracking figures to HDD')

########################################################################################################################
# DISTANCE MOVED
all_data = dict.fromkeys(groups)
for kk, gg in enumerate(groups):
    # Load pickled data
    group = gg
    with open(f'{path}/{group}/raw_data.pkl', "rb") as input_file:
        raw_data = pickle.load(input_file)
    data = import_pickle(f'{path}/{group}/behavior_distance_moved.pkl')
    protocol = import_pickle(f'{path}/{group}/behavior_protocol.pkl')
    all_wells = pd.read_csv(f'{path}/wells.csv', sep=';', header=0, nrows=len(groups)).fillna(value='nofish')
    wells_csv = all_wells.loc[kk][1:]

    # ***********************************************
    # Remove miss tracked fish:
    if remove_fish_in_group[kk]:
        data, wells_csv = remove_miss_tracked_fish(original_data=data, miss_tracked_idx=remove_fish_in_group_well[kk],
                                                   wells=wells_csv)
    # ***********************************************

    # Rename well label to treatments
    treatment_labels = ['Time', 'Stimulus'] + list(wells_csv.to_numpy())
    # find wells with no fish in it:
    # If there is a fish it is True, if not it is False:
    idx_nofish_bool = [True, True] + list(np.array(treatment_labels[2:]) != 'nofish')
    marked_labels = data.keys()[idx_nofish_bool]
    final_treatment_labels = np.array(treatment_labels)[idx_nofish_bool]
    data_sorted = data[marked_labels]
    # Rename data with treatment labels
    data_sorted.columns = final_treatment_labels
    data_sorted.columns = final_treatment_labels
    # idx_nofish = np.where(np.array(labels1[2:]) == 'nofish')

    # Put data into one dict:
    all_data[gg] = data_sorted

# Now combine data in dict into one data frame (ignore Group1 and remove time and stimulus from the others):
data_final = pd.concat([all_data['Group2'],  all_data['Group3'][all_data['Group3'].keys()[2:]],
                        all_data['Group4'][all_data['Group4'].keys()[2:]]], axis=1)

# Cut out data corresponding to stimulation:
# find stimuli tags:
stimuli = find_stimuli(data_final)
idx_tags = stimuli['idx']
# This are the important treatments we want to look at:
hardwired_treatments = ['control', 'neo50uM', 'neo100uM', 'neo200uM', 'neo400uM',
                        'cu1uM', 'cu10uM', 'cu50uM', 'cu100uM']
data_analysed = dict.fromkeys(stimuli['labels'])
summed_over_time = dict.fromkeys(stimuli['labels'])
for k, v in enumerate(idx_tags):
    if stimuli['labels'][k].startswith('Tap'):
        window_samples = int(window_time[-1] * sampling_rate)
    else:
        window_samples = int(window_time[k] * sampling_rate)

    # Get stimuli that are important and are available as well:
    # treatments_per_stimulus = set(np.unique(data_final.loc[v:v + window_samples].keys())) & set(hardwired_treatments)
    treatments_per_stimulus0 = set(np.unique(data_final.loc[v:v + window_samples].keys())).intersection(hardwired_treatments)
    treatments_per_stimulus = np.sort(list(treatments_per_stimulus0))
    data_analysed[stimuli['labels'][k]] = dict.fromkeys(treatments_per_stimulus)
    summed_over_time[stimuli['labels'][k]] = dict.fromkeys(treatments_per_stimulus)
    for i, w in enumerate(treatments_per_stimulus):
        data_analysed[stimuli['labels'][k]][w] = data_final.loc[v:v + window_samples][w]
        summed_over_time[stimuli['labels'][k]][w] = dict.fromkeys(['mean', 'sem', 'std', 'median', 'mad', 'quantiles',
                                                                   'stimulus', 'treatment', 'n_fish', 'sums'])
        summed_over_time[stimuli['labels'][k]][w]['stimulus'] = stimuli['labels'][k]
        summed_over_time[stimuli['labels'][k]][w]['treatment'] = w
        summed_over_time[stimuli['labels'][k]][w]['n_fish'] = len(data_final.loc[v:v + window_samples][w].keys())
        summed_over_time[stimuli['labels'][k]][w]['sums'] = data_final.loc[v:v + window_samples][w].sum(axis=0)
        summed_over_time[stimuli['labels'][k]][w]['mean'] = data_final.loc[v:v + window_samples][w].sum(axis=0).mean()
        summed_over_time[stimuli['labels'][k]][w]['sem'] = data_final.loc[v:v + window_samples][w].sum(axis=0).sem()
        summed_over_time[stimuli['labels'][k]][w]['std'] = data_final.loc[v:v + window_samples][w].sum(axis=0).std()
        summed_over_time[stimuli['labels'][k]][w]['median'] = data_final.loc[v:v + window_samples][w].sum(axis=0).median()
        summed_over_time[stimuli['labels'][k]][w]['mad'] = data_final.loc[v:v + window_samples][w].sum(axis=0).mad()
        summed_over_time[stimuli['labels'][k]][w]['quantiles'] = \
            data_final.loc[v:v + window_samples][w].sum(axis=0).quantile([0.25, 0.5, 0.75])

# Plot some stuff

for stim in stimuli['labels']:
    fig = plt.figure(stim)
    trs = summed_over_time[stim].keys()
    y = []
    y_median = []
    y_median_err = []
    y_mean = []
    y_mean_err = []
    x_labels = []
    x_treatments = []
    for k, v in enumerate(trs):
        y.append(summed_over_time[stim][v]['sums'])
        y_median.append(summed_over_time[stim][v]['median'])
        y_median_err.append(summed_over_time[stim][v]['mad'])
        y_mean.append(summed_over_time[stim][v]['mean'])
        y_mean_err.append(summed_over_time[stim][v]['sem'])
        n_fish = summed_over_time[stim][v]['n_fish']
        x_labels.append(v + f'\nn={n_fish}')
        x_treatments.append(summed_over_time[stim][v]['treatment'])
        if v != summed_over_time[stim][v]['treatment']:
            print('ERROR: TREATMENTS DO NOT MATCH')

    # plt.plot(trs, y, 'ko')
    # plt.errorbar(trs, y, yerr=y_err, color='black', linestyle='None')
    # plt.errorbar(x_labels, y_median, yerr=y_median_err, color='black', linestyle='None')

    # plt.plot(trs, y, 'ko')
    # plt.plot(trs, y_mean, 'rx')
    # plt.bar(x_labels, y_median, color='gray')
    plt.boxplot(y, labels=x_labels, positions=range(len(x_labels)), showfliers=False, widths=0.25)

    if stim.startswith('Tap'):
        plt.ylim([0, 10])
    if stim.startswith('Light'):
        plt.ylim([0, 100])
    # plt.ylim([0, 12])
    # SAVE FIG
    CM = 1 / 2.54  # centimeters in inches
    # Width x Height
    fig.set_size_inches(8 * 2 * CM, 6 * 2 * CM)
    fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.3, hspace=0.3)
    SAVE_PATH = f'{path}/figs/'
    fig_name = f'Bars_{stim}'
    # fig.savefig(f'{SAVE_PATH}tracking/{fig_name}.pdf')
    fig.savefig(f'{SAVE_PATH}bars/{fig_name}.jpg', dpi=300)
    plt.close(fig)
exit()
########################################################################################################################
