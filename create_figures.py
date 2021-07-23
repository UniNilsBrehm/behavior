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

# LOAD DATA
sampling_rate = 30
dir_path = f'C:/Uni Freiburg/Behavior/kurs2021/'  # get current working dir
path_summed_distances = f'{dir_path}/analysis/data/summed_distances.pkl'
path_time_courses = f'{dir_path}/analysis/data/time_courses.pkl'

with open(path_time_courses, "rb") as input_file:
    time_course_all = pickle.load(input_file)
with open(path_summed_distances, "rb") as input_file:
    summed_distances = pickle.load(input_file)

# # Just some testing
# p = 'C:/Users/Nils/Desktop/test.csv'
# df = dt.fread(p)
#
# # WORKING WITH LONG FORMAT AND DATATABLE PACKAGE
# # Find idx of all "tap4_x" stimuli:
# idx = [x for x, v in enumerate(df['stimulus'].to_list()[0]) if v.startswith('tap4')]
# tap4 = df[idx, :]
#
# # Get mean distance for each stimuli and treatment group
# means = df[:, dt.mean(f[:]), by('stimulus', 'treatment')]

# Look at some statistics
# Get mean distance for each stimuli and treatment group
# means = data_all_groups[:, [dt.mean(f['Distance(summed)']), dt.sd(f['Distance(summed)'])], by('Stimulus', 'Treatment')]
# data_all_groups[:, :, by(f['Stimulus'] == 'Tap5_1', f['Treatment'])]

# # Pandas Pivot Table
# mean = pd.pivot_table(summed_distances.to_pandas(), values='Distance(summed)', index=['Stimulus'],
#                     columns=['Treatment'], aggfunc=np.mean)
#
# sd = pd.pivot_table(summed_distances.to_pandas(), values='Distance(summed)', index=['Stimulus'],
#                     columns=['Treatment'], aggfunc=np.std)
# embed()
# exit()
# PLOTTING
row_names = ['A', 'B', 'C', 'D', 'E', 'F']
# Loop through Groups
for k in tqdm(time_course_all):
    # Loop through stimuli:
    for i in time_course_all[k]:
        max_time = time_course_all[k][i].shape[0] / sampling_rate  # in secs; has to be set manually
        fig, axs = plt.subplots(6, 8)
        plot_data = time_course_all[k][i]
        t = np.linspace(0, max_time, plot_data.shape[0])

        # y_max = np.nanmax(plot_data)
        cc = 0
        for rows in range(axs.shape[0]):
            axs[rows, 0].text(-0.25, 0.5, row_names[rows], transform=axs[rows, 0].transAxes, fontsize=14,
                              horizontalalignment='center', verticalalignment='center')

            for cols in range(axs.shape[1]):
                d = np.array(plot_data[:, cc].to_list()[0], dtype='float')
                # d_min_max = (d-np.nanmin(d)) / (np.nanmax(d)-np.nanmin(d))
                d_z = stats.zscore(d, axis=0, ddof=0, nan_policy='omit')  # propagate or omit
                axs[rows, cols].plot(t, d_z, color='black', linewidth=0.5)
                axs[rows, cols].set_yticks([])
                axs[rows, cols].set_xticks([])
                sd_limit = 5
                axs[rows, cols].set_ylim([-sd_limit, sd_limit])  # upper limit is in SD away from mean at zero
                axs[0, cols].text(0.5, 1.25, cols + 1, transform=axs[0, cols].transAxes, fontsize=14,
                                  horizontalalignment='center', verticalalignment='center')

                cc += 1
        # SAVE FIG
        CM = 1 / 2.54  # centimeters in inches
        # Width x Height
        fig.set_size_inches(8 * 2 * CM, 6 * 2 * CM)
        fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.3, hspace=0.3)
        SAVE_PATH = f'{dir_path}/analysis/figs/DistanceMoved/'
        fig_name = f'{k}_DistanceMoved_{i}'
        # fig.savefig(f'{SAVE_PATH}tracking/{fig_name}.pdf')
        fig.savefig(f'{SAVE_PATH}{fig_name}.jpg', dpi=300)
        plt.close(fig)
print('Figures saved to HDD')