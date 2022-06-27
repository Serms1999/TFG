#!/usr/bin/env python3

import os
import sys
import re
import platform
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from analysis import generate_offset_dict, generate_delta_offset_dict, generate_stats_csv
from globals import *

os.environ["PATH"] += os.pathsep + latexPaths[platform.system()]

mpl.use("pgf")
mpl.rcParams.update(pgf_with_latex)

# Data
data = 0
data_type = ''
offset_diffs = 0
delta_offsets = 0
delta_offsets_no_out = 0
state = 27


def get_limits(data: dict):
    limits = {}
    for key in data:
        Q1 = data[key]['delta_offset'].quantile(0.25)
        Q3 = data[key]['delta_offset'].quantile(0.75)
        IQR = Q3 - Q1

        limits[key] = [IQR, data[key]['delta_offset'].quantile(0.5)]

    return limits

def generate_delta_offset_plot(data: dict):
    path = f'plots/{data_type}'
    plt.figure()

    for key in data:
        plt.scatter(delta_offsets[key]['time'], delta_offsets[key]['delta_offset'] / np.power(10, 3),
                    color=colors[key], label=key, s=3)

        plt.xlabel('Tiempo (\\si{\\second})')
        plt.ylabel('$\\Delta$Offset (\\si{\\micro\\second})')
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

        plt.savefig(f'{path}/delta_offset_plot_{key}.pdf', format='pdf', bbox_inches='tight')
        plt.clf()

def find_anomalies(data, plot=False):
    path = f'plots/{data_type}'
    plt.figure()

    no_outlier_data = {}

    for key in data:
        X = data[key]
        iforest = IsolationForest(n_estimators=100, contamination=0.05, n_jobs=-1, random_state=state, warm_start=True)

        # Returns 1 of inliers, -1 for outliers
        pred = iforest.fit_predict(X.values)

        # Extract outliers
        outlier_index = np.where(pred == -1)
        outliers = X.iloc[outlier_index]
        non_outliers = X.drop(X.index[outlier_index], inplace=False)
        outlier_values = outliers['delta_offset']
        outlier_times = outliers['time']
        no_outlier_values = non_outliers['delta_offset']
        no_outlier_times = non_outliers['time']

        no_outlier_data[key] = pd.DataFrame({'time': no_outlier_times, 'delta_offset': no_outlier_values})

        if plot:
            # Plot the data
            plt.scatter(no_outlier_times, no_outlier_values / np.power(10, 3),
                        color=colors[key], label='normal values', s=3)

            plt.scatter(outlier_times, outlier_values / np.power(10, 3),
                        color='red', label='outliers', s=6, marker='x')

            plt.xlabel('Tiempo (\\si{\\second})')
            plt.ylabel('$\\Delta$Offset (\\si{\\micro\\second})')
            plt.ylim(-150, 150)
            #plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

            plt.savefig(f'{path}/delta_offset_plot_{key}.pdf', format='pdf', bbox_inches='tight')
            plt.clf()

    if plot:
        plt.figure()
        for key in data:
            no_outlier_times, no_outlier_values = no_outlier_data[key]
            plt.plot(no_outlier_times, np.cumsum(no_outlier_values) / np.power(10, 6), color=colors[key], label=key)

        plt.xlabel('Tiempo (\\si{\\second})')
        plt.ylabel('Offset (\\si{\\milli\\second})')
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

        plt.savefig(f'{path}/no_outliers_offset_plot.pdf', format='pdf', bbox_inches='tight')
        plt.clf()

    return no_outlier_data


def plot_anomalies_legend():
    path = f'plots/{data_type}'
    fig = plt.figure()
    for key in ['Disp. 1', 'Disp. 2', 'Disp. 3', 'Disp. 4', 'Disp. 5']:
        plt.scatter(0, 0, color=colors[key], label=f'{key} normal', s=3)

    plt.scatter(0, 0, color='red', label='outliers', s=6, marker='x')
    plt.legend()

    fig = plt.figure("Line plot")
    legendFig = plt.figure("Legend plot")
    ax = fig.add_subplot(111)
    line1 = ax.scatter(0, 0, c=colors['Disp. 1'], s=3)
    line2 = ax.scatter(0, 0, c=colors['Disp. 2'], s=3)
    line3 = ax.scatter(0, 0, c=colors['Disp. 3'], s=3)
    line4 = ax.scatter(0, 0, c=colors['Disp. 4'], s=3)
    line5 = ax.scatter(0, 0, c=colors['Disp. 5'], s=3)
    line6 = ax.scatter(0, 0, c='red', marker='x', s=6)
    legendFig.legend([line1, line2, line3, line4, line5, line6],
                     ['Disp. 1', 'Disp. 2', 'Disp. 3', 'Disp. 4', 'Disp. 5', 'outliers'], loc='center')
    legendFig.savefig(f'{path}/anomaly_legend.pdf', format='pdf', bbox_inches='tight')
    plt.clf()


def main():
    global data, data_type, offset_diffs, delta_offsets, delta_offsets_no_out

    data_type = 'parallel'

    # Leer datos
    data = read_data(data_type)

    # Calculamos los offsets
    offset_diffs = generate_offset_dict(data)

    # Calculamos los incrementos de los offsets sin eliminar outliers
    delta_offsets = generate_delta_offset_dict(data, offset_diffs, remove_outliers=False)

    # Calculamos los incrementos de los offsets, eliminando outliers
    delta_offsets_no_out = generate_delta_offset_dict(data, offset_diffs, remove_outliers=True)

    """
    training_sets = {}
    test_sets = {}
    for key in delta_offsets:
        training_sets[key] = pd.DataFrame(delta_offsets[key][:30000])
        training_sets[key]['delta_offset'] /= np.power(10, 3)
        test_sets[key] = pd.DataFrame(delta_offsets[key][30000:50000])
        test_sets[key]['delta_offset'] /= np.power(10, 3)

    fig = px.line(training_sets['Disp. 1'].reset_index(), x='time', y='delta_offset', title=r"$\text{TAXI RIDES}$")
    fig.update_xaxes(
        rangeslider_visible=True,
    )
    fig.show()

    find_anomalies(training_sets, test_sets)
    """


if __name__ == '__main__':
    main()
