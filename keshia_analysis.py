
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

'''
This script will load converted data from the Danio Vision Apparatus and creates several different figures.
For more detail please see the comments for each figure below.

Nils Brehm - 2021
'''
# ======================================================================================================================
# ======================================================================================================================
# SETTINGS:
# *********************************************
check_tracking = False

distance_moved_per_stimulus = True

boxplots = False

remove_miss_tracked_fish = False
# *********************************************

# set sample rate of the recording:
sampling_rate = 30  # in Hz
# get current working dir
dir_path = os.getcwd()
# set path for data and exporting figures
path = f'{dir_path}/keshia/20210622/analysis/'

# ======================================================================================================================
# ======================================================================================================================

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


def smooth(y_data, box_pts):
    # Running average with box window
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y_data, box, mode='same')
    return y_smooth


# START OF SCRIPT ======================================================================================================
# DATA HANDLING ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print('Starting Analysis')
# Load pickled data
data = import_pickle(f'{path}/behavior_distance_moved.pkl')
protocol = import_pickle(f'{path}/behavior_protocol.pkl')
wells = pd.read_csv(f'{path}/wells.csv', sep=';', header=0).fillna(value='nofish')

# ***********************************************
# Remove miss tracked fish:
if remove_miss_tracked_fish:
    miss_tracked_idx = ['B5']
    data_checked = data.drop(columns=miss_tracked_idx)
    wells = wells.drop(columns=miss_tracked_idx)
else:
    data_checked = data
# ***********************************************

# Rename well label to treatments
labels = ['Time', 'Stimulus'] + list(wells.to_numpy()[0])
distance_moved = data_checked.set_axis(labels, axis=1)

# Remove empty wells with ('nofish'):
distance_moved = distance_moved.drop(columns='nofish')

# find wells with no fish in it:
idx_nofish = np.where(np.array(labels[2:]) == 'nofish')

# Find Stimuli:
tags = distance_moved['Stimulus'].notnull()
stim_set = distance_moved[tags]
idx_all_tags = stim_set.index  # All tags
idx = stim_set.index[0:-1]  # All tag entries except "Stop Tracking"
idx_all_stims = stim_set.index[1:-1]  # Remove start tack and stop track tags
idx_taps = stim_set.index[3:-1]  # only taps
idx_light = stim_set.index[1:3]  # only light on and off
idx_light_on = stim_set.index[1]  # only light on
idx_light_off = stim_set.index[2]  # only light off
idx_mixed = stim_set.index[0:3]  # only start track, light on and off
idx_start_tracking = stim_set.index[0]  # only start tracking
meta_idx = [[idx_start_tracking], idx_light, idx_taps]
meta_labels = ['Start tracking', 'Lights', 'Taps']
stimulus_labels = list(stim_set['Stimulus'][tags])

# FIGURES ==============================================================================================================
# ==== Visually check tracking ====
'''
This will create three figures that can help you to check if the tracking did work successfully.
1. The Tracking of the first 60 s (no stimulus)
2. The Tracking of the first 5 s of the Light on and Light off stimuli
3. The Tracking of the first 5 s of the Tapping Stimuli

Tracked coordinates are global coordinates. This means the entire well plate is a cartesian space. The center of the
plate corresponds to the origin of that space. Up (y-axis) and right (x-axis) are positive values. Down (y-axis) and
left (x-axis) are negative values.
'''
# SET TIME WINDOWS FOR ANALYSIS:
# **********************************************************************
window_time = [60, 30, 1]  # in secs ['Start tracking', 'Lights', 'Taps']
# **********************************************************************
plot_settings()
rand_color = np.round(np.random.rand(len(stimulus_labels) - 2, 3), 1)
if check_tracking:
    bar = progressbar.ProgressBar(maxval=len(meta_idx)).start()
    # Load pickled data:
    print('Please wait while saving figures')
    with open(f'{path}/raw_data.pkl', "rb") as input_file:
        raw_data = pickle.load(input_file)

    # Find Well Labels:
    w = list(raw_data.keys())
    window_samples = np.array(np.array(window_time) * sampling_rate, dtype='int')
    cc = 1
    grays = list(np.array(np.round(np.linspace(0, 1, len(idx_taps)), 2), dtype='str'))
    color_values = [['black'], ['red', 'blue'], rand_color]
    for meta_k, meta_v in enumerate(meta_idx):  # Loop through ['Start tracking', 'Lights', 'Taps']
        fig, axs = plt.subplots(6, 8)
        fig2, axs2 = plt.subplots(6, 8)

        for kk, start in enumerate(meta_v):  # Loop through all tags in stimulus set (Tap01, Tap02, ....)
            count = 0
            colors = color_values[meta_k][kk]
            for row in range(6):
                for col in range(8):
                    x = raw_data[w[count]]['X center'][start:start + window_samples[meta_k]]
                    y = raw_data[w[count]]['Y center'][start:start + window_samples[meta_k]]
                    axs[row, col].plot(x, y, color=colors)
                    axs[row, col].axis('equal')
                    axs[row, col].set_xticks([])
                    axs[row, col].set_yticks([])

                    axs2[row, col].plot(x, y, color=colors)
                    axs2[row, col].axis('equal')

                    # Axis limits (centered around median position values)
                    m_x = np.nanmedian(x)
                    m_y = np.nanmedian(y)
                    axs[row, col].set_xlim([m_x - 10, m_x + 10])
                    axs[row, col].set_ylim([m_y - 10, m_y + 10])
                    axs[row, col].set_xticks([])
                    axs[row, col].set_yticks([])

                    axs2[row, col].set_xlim([m_x - 10, m_x + 10])
                    axs2[row, col].set_ylim([m_y - 10, m_y + 10])
                    axs2[row, col].set_xticks([])
                    axs2[row, col].set_yticks([])

                    if count in idx_nofish[0]:
                        axs[row, col].patch.set_facecolor('red')
                        axs2[row, col].patch.set_facecolor('red')
                        axs[row, col].patch.set_alpha(0.5)
                        axs2[row, col].patch.set_alpha(0.5)

                    count += 1
            # Assign well labels to the figure:
            for i in range(8):
                axs[0, i].text(0.5, 1.25, i+1, transform=axs[0, i].transAxes, fontsize=14,
                               horizontalalignment='center', verticalalignment='center')
                axs2[0, i].text(0.5, 1.25, i + 1, transform=axs2[0, i].transAxes, fontsize=14,
                               horizontalalignment='center', verticalalignment='center')

            row_names = ['A', 'B', 'C', 'D', 'E', 'F']
            for k in range(6):
                axs[k, 0].text(-0.25, 0.5, row_names[k], transform=axs[k, 0].transAxes, fontsize=14,
                               horizontalalignment='center', verticalalignment='center')
                axs2[k, 0].text(-0.25, 0.5, row_names[k], transform=axs2[k, 0].transAxes, fontsize=14,
                               horizontalalignment='center', verticalalignment='center')

            if meta_labels[meta_k] == 'Taps':
                # Save Fig:
                CM = 1 / 2.54  # centimeters in inches
                # Width x Height
                fig2.set_size_inches(8 * 2 * CM, 6 * 2 * CM)
                fig2.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.3, hspace=0.3)
                SAVE_PATH = f'{path}/figs/'
                # fig_name = f'Tracking_{stimulus_labels[kk]}_{window_time[kk]}secs'
                fig_name2 = f'Tracking_{meta_labels[meta_k]}_{window_time[meta_k]}secs'
                # fig2.savefig(f'{SAVE_PATH}tracking/{fig_name2}_{kk}.pdf')
                fig2.savefig(f'{SAVE_PATH}tracking/{fig_name2}_{kk}.jpg', dpi=300)
                plt.close(fig2)
                fig2, axs2 = plt.subplots(6, 8)

        # axs[0, 0].legend(np.arange(1, 11, 1))

        # Save Fig:
        CM = 1 / 2.54  # centimeters in inches
        # Width x Height
        fig.set_size_inches(8*2 * CM, 6*2 * CM)
        fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.3, hspace=0.3)
        SAVE_PATH = f'{path}/figs/'
        # fig_name = f'Tracking_{stimulus_labels[kk]}_{window_time[kk]}secs_all'
        fig_name = f'Tracking_{meta_labels[meta_k]}_{window_time[meta_k]}secs_all'
        # fig.savefig(f'{SAVE_PATH}tracking/{fig_name}.pdf')
        fig.savefig(f'{SAVE_PATH}tracking/{fig_name}.jpg', dpi=300)
        plt.close(fig)
        cc += 1
        bar.update(meta_k)

    print('---------- All Tracking Figures saved! ----------')

# ==== DISTANCE MOVED PER STIMULUS =====================================================================================
'''
This will create two figures.
1. Distance Moved of all Tapping Stimuli per Treatment Group
2. Distance Moved of the first x secs and of Light on and Light off stimuli
'''

if distance_moved_per_stimulus:
    # Stimulus protocol:
    # Start track
    # Light on
    # Light off
    # Tap01
    # Tap02
    # ...
    # Stop track

    # Set Stimulus Selection:
    # ***********************************
    stimulus_name = 'Taps'
    idx_select = idx_taps
    smooth_data = False
    smooth_window = 30  # window size in samples
    smooth_window_time = smooth_window / sampling_rate
    time_window = 1  # in secs
    max_y = 40
    max_y_automated = False
    number_of_x_ticks = 3
    number_of_y_ticks = 5
    # ***********************************

    if smooth_data:
        stimulus_name = stimulus_name + f'SMOOTHED_{smooth_window_time}secs_'
    sampling_rate = 30  # in Hz
    window = int(sampling_rate * time_window)  # in samples

    t = np.linspace(0, time_window, window+1)

    treatments_labels = ['Control', 'Anterior', 'Posterior', 'Full']
    tr = ['C', 'A', 'P', 'F']
    animals = distance_moved.keys()[2:]
    groups = dict.fromkeys(tr)
    groups['C'] = []
    groups['A'] = []
    groups['P'] = []
    groups['F'] = []

    for k in animals:
        groups['C'].append(k.startswith('C'))
        groups['A'].append(k.startswith('A'))
        groups['P'].append(k.startswith('P'))
        groups['F'].append(k.startswith('F'))

    a = []
    if idx_select.size > 1:
        for k, v in enumerate(idx_select):
            a.append(distance_moved.loc[v:v + window])
    else:
        a.append(distance_moved.loc[idx_select:idx_select + window])

    time_means = dict.fromkeys(animals)
    time_sem = dict.fromkeys(animals)
    for i in animals:
        c = []
        for k in a:
            # Convert to mm/sec
            c.append(k[i] * sampling_rate)
        time_means[i] = np.mean(c, axis=0)
        time_sem[i] = np.std(c, axis=0) / np.sqrt(len(c))

    sampling_rate = 30  # in Hz
    # PLOT IT: =========================================================================================================
    fig, axs = plt.subplots(2, 2)
    ax_nr = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # Get max value of all measurements:
    all_vals = []
    for ii, kk in enumerate(ax_nr):
        for k in animals[groups[tr[ii]]]:
            y = time_means[k]
            if smooth_data:
                y = smooth(y, smooth_window)
            all_vals.append(y)
            y_err = time_sem[k]
            axs[kk].plot(t, y)
            axs[kk].errorbar(t, y, yerr=y_err)
            axs[kk].set_title(treatments_labels[ii])
            axs[kk].set_xticks(np.linspace(0, time_window, number_of_x_ticks))
    # Axis settings
    if max_y_automated:
        max_y = np.round(np.nanmax(all_vals), 1)
    axs[0, 0].set_ylim([0, max_y])
    axs[0, 1].set_ylim([0, max_y])
    axs[1, 0].set_ylim([0, max_y])
    axs[1, 1].set_ylim([0, max_y])

    axs[0, 0].set_yticks(np.linspace(0, max_y, number_of_y_ticks))
    axs[0, 1].set_yticks(np.linspace(0, max_y, number_of_y_ticks))
    axs[1, 0].set_yticks(np.linspace(0, max_y, number_of_y_ticks))
    axs[1, 1].set_yticks(np.linspace(0, max_y, number_of_y_ticks))

    axs[0, 0].set_xticklabels([])
    axs[0, 1].set_xticklabels([])

    axs[0, 1].set_yticklabels([])
    axs[1, 1].set_yticklabels([])

    axs[1, 1].text(-0.35, -0.2, 'Geschwindigkeit [mm/s]', transform=axs[0, 0].transAxes, fontsize=8,
                   horizontalalignment='center', verticalalignment='center', rotation=90)
    axs[1, 1].text(-0.1, -0.3, 'Zeit [s]', transform=axs[1, 1].transAxes, fontsize=8,
                   horizontalalignment='center', verticalalignment='center')

    # Save Fig ====
    CM = 1 / 2.54  # centimeters in inches
    fig.set_size_inches(10 * CM, 10 * CM)
    fig.subplots_adjust(left=0.2, top=0.9, bottom=0.2, right=0.9, wspace=0.1, hspace=0.5)
    SAVE_PATH = f'{path}/figs/'
    fig.savefig(f'{SAVE_PATH}distancemoved/DistanceMoved_{stimulus_name}_{time_window}_secs.pdf')
    fig.savefig(f'{SAVE_PATH}distancemoved/DistanceMoved_{stimulus_name}_{time_window}_secs.jpg', dpi=300)
    plt.close(fig)
    print('Distance Moved Plots saved!')

# ======================================================================================================================
# Statistics:
# BOX PLOTS
if boxplots:
    sampling_rate = 30  # in Hz
    time_window = [60, 30, 30] + [1] * 10  # in secs
    animals = distance_moved.keys()[2:]
    dn = dict.fromkeys(animals)
    for i_a, v_a in enumerate(animals):
        dn[v_a] = {}
        for i_idx, v_idx in enumerate(idx):
            time_samples = int(time_window[i_idx] * sampling_rate)
            # Velocity (distance / time window)
            dn[v_a][stimulus_labels[i_idx]] = distance_moved[v_a].loc[v_idx:v_idx + time_samples].sum() / time_window[i_idx]

    treatments_labels = ['Control', 'Anterior', 'Posterior', 'Full']
    treats = ['C', 'A', 'P', 'F']

    # BOXPLOT
    medianprops = dict(linestyle='-', linewidth=1.5, color='black')
    meanlineprops = dict(linestyle='--', linewidth=1, color='red')

    aa = ['S', 'T', 'On', 'Off']
    title_names = [f'Spontan ({time_window[0]} s)', f'Taps ({time_window[-1]} s)', f'Licht an ({time_window[1]} s)',
                   f'Licht aus ({time_window[1]} s)']

    treatment_groups = []
    for aa_i, aa_v in enumerate(aa):
        if aa_i > 1:
            pos = 'end'
        else:
            pos = 'start'
        # Get summed distance moved per Stimuli:
        treatment_groups.append(sort_stimuli_data(n=dn, fish=animals, stimulus_name=aa_v, position=pos, t=treats))

    # Number of animals per treatment:
    n_animals = []
    for k in treatment_groups[0]:
        n_animals.append(len(k))

    treatments_labels_n = [f'Control (n={n_animals[0]})', f'Anterior (n={n_animals[1]})',
                           f'Posterior (n={n_animals[2]})', f'Full (n={n_animals[3]})']

    fig, axs = plt.subplots(2, 2)
    cc = 0
    medians = []
    quartiles = []
    for rows in range(2):
        for cols in range(2):
            a = axs[rows, cols].boxplot(treatment_groups[cc], labels=treatments_labels, meanprops=meanlineprops, meanline=True,
                                        showmeans=False, showfliers=False, medianprops=medianprops)
            # Get Median Values from box plot
            for u in a['medians']:
                medians.append(np.round(u.get_ydata()[0], 2))
            for u in a['boxes']:
                quartiles.append(np.round(u.get_ydata()[0], 2))
                quartiles.append(np.round(u.get_ydata()[2], 2))

            axs[rows, cols].violinplot(treatment_groups[cc], showmeans=False, showmedians=False, showextrema=False)
            axs[rows, cols].text(0.5, 0.9, title_names[cc], transform=axs[rows, cols].transAxes, fontsize=8,
                                 horizontalalignment='center', verticalalignment='center')

            axs[rows, cols].set_ylim([0, 15])
            axs[rows, cols].set_yticks([0, 5, 10, 15])
            axs[rows, cols].spines['top'].set_visible(False)
            axs[rows, cols].spines['right'].set_visible(False)
            axs[rows, cols].spines['bottom'].set_visible(True)
            axs[rows, cols].spines['left'].set_visible(True)
            cc += 1

    # Axis Settings
    # X Axis settings
    axs[0, 0].set_xticklabels([])
    axs[0, 1].set_xticklabels([])
    # axs[0, 0].set_xticks([])
    # axs[0, 1].set_xticks([])
    axs[0, 0].spines['bottom'].set_visible(True)
    axs[0, 1].spines['bottom'].set_visible(True)
    axs[1, 0].set_xticklabels(treatments_labels, rotation=45)
    axs[1, 1].set_xticklabels(treatments_labels, rotation=45)

    # Y Axis settings
    axs[0, 1].set_yticklabels([])
    axs[1, 1].set_yticklabels([])
    axs[1, 1].text(-0.35, 0, 'Geschwindigkeit [mm/s]', transform=axs[0, 0].transAxes, fontsize=8,
                   horizontalalignment='center', verticalalignment='center', rotation=90)

    # Save Fig ====
    CM = 1 / 2.54  # centimeters in inches
    fig.set_size_inches(10 * CM, 10 * CM)
    fig.subplots_adjust(left=0.2, top=0.9, bottom=0.2, right=0.9, wspace=0.05, hspace=0.25)
    SAVE_PATH = f'{path}/figs/'
    fig.savefig(f'{SAVE_PATH}BoxPlots.pdf')
    fig.savefig(f'{SAVE_PATH}BoxPlots.jpg', dpi=300)
    plt.close(fig)
    with open(f'{SAVE_PATH}BOXPLOT_n_animals.txt', 'w') as f:
        f.write('Number of animals in each treatment group:\n')
        for k in treatments_labels_n:
            f.write(f'{k}\n')
    with open(f'{SAVE_PATH}BOXPLOT_medians.txt', 'w') as f:
        f.write('Median values for each treatment group:\n')
        for k in medians:
            f.write(f'{k}\n')
    with open(f'{SAVE_PATH}BOXPLOT_quartiles.txt', 'w') as f:
        f.write('Quartile values for each treatment group:\n')
        for k in quartiles:
            f.write(f'{k}\n')
    print('Figure saved!')
