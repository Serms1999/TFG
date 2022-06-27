#!/usr/bin/env python3

import os
import sys
import platform

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import norm

from globals import *

os.environ["PATH"] += os.pathsep + latexPaths[platform.system()]

mpl.use("pgf")
mpl.rcParams.update(pgf_with_latex)

# Data
data = 0
data_type = ''
offset_diffs = 0
delta_offsets = 0


def generate_stats_csv(data: DataFrame, generate=False, delta=False):

    path = f'stats/{data_type}'
    fname_dev = 'statsDev'
    fname = 'stats'
    if delta:
        fname = 'delta_' + fname
        fname_dev = 'delta_' + fname_dev

    names = ['Sum', 'Mean', 'Median', 'Mode', 'Std', 'IQR', 'Kurtosis', 'Skew', 'Max', 'Min']
    df = pd.DataFrame(columns=names)

    list_df = []
    num_dev = len(data.keys())

    if generate:
        for i, dev in enumerate(data):
            if delta:
                dfDev = statistics(slice_window(data[dev]['delta_offset'], 60), names)
            else:
                dfDev = statistics(slice_window(data[dev], 60), names)
            devCol = pd.DataFrame({'Device': np.repeat(dev, len(dfDev))})

            dfDev = dfDev.join(devCol)
            dfDev.index = np.array(dfDev.index) + 1
            list_df += [dfDev]

            dfDev.to_csv(f'{path}/{fname_dev}{i + 1}.csv', index=True, header=True)
            print(f'{fname_dev}{i + 1}.csv generado')
    else:
        for i, _ in enumerate(data):
            list_df += [pd.read_csv(f'{path}/{fname_dev}{i + 1}.csv', index_col=0)]

    num_rows = np.max([x.index[-1] for x in list_df])

    for row in range(1, num_rows + 1):
        for dev in range(num_dev):
            try:
                df = df.append(list_df[dev].loc[row], ignore_index=True)
            except KeyError:
                pass

    df.index = np.array(df.index) + 1
    df.to_csv(f'{path}/{fname}.csv', index=True, header=True)
    print(f'{fname_dev}.csv generado')


def generate_boxplots(data: DataFrame):
    path = f'plots/{data_type}'

    data_copy = data.copy()

    def save_boxplot(fname: str, outliers: bool, unit: str):
        plt.figure()
        data_copy.boxplot(showfliers=outliers)
        plt.grid(outliers)
        plt.xlabel('Dispositivo')
        plt.ylabel(f'$\\Delta$Offset (\\si{{\\{unit}\\second}})')

        plt.savefig(f'{path}/{fname}', format='pdf')
        plt.clf()

    for key in data:
        data_copy[key] /= np.power(10, 3)

    save_boxplot('boxplot_no_out.pdf', outliers=False, unit='micro')

    for key in data:
        data_copy[key] /= np.power(10, 3)

    save_boxplot('boxplot.pdf', outliers=True, unit='milli')


def generate_offset_dict(data: DataFrame):
    d = {}

    for key in data:
        d[key] = np.array([t - s for s, t in zip(data[key]['offset'], data[key]['offset'][1:])])  # / pow(10, 3)

    return pd.DataFrame(d)


def generate_delta_offset_dict(data: DataFrame, offset_diffs: DataFrame, remove_outliers=True):
    delta = dict()

    for key in data:
        time_series = pd.Series(data[key]['time'][1:])
        offset_series = offset_diffs[key]
        delta[key] = pd.DataFrame({'time': time_series, 'delta_offset': offset_series})
        outlier_list = [0, len(data[key]) - 1]

        if remove_outliers:
            Q1 = delta[key]['delta_offset'].quantile(0.25)
            Q3 = delta[key]['delta_offset'].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_list = delta[key].index[(delta[key]['delta_offset'] < lower_bound) |
                                                    (delta[key]['delta_offset'] > upper_bound)]

        delta[key] = delta[key].drop(sorted(set(outlier_list)))

    return delta


def generate_histograms(data: DataFrame):
    mu_111, std_111 = norm.fit(data['Disp. 1'][1:])
    mu_154, std_154 = norm.fit(data['Disp. 2'][1:])
    mu_159, std_159 = norm.fit(data['Disp. 3'][1:])
    mu_175, std_175 = norm.fit(data['Disp. 4'][1:])
    mu_179, std_179 = norm.fit(data['Disp. 5'][1:])

    dict_delta = {'111': data['Disp. 1'][1:], '154': data['Disp. 2'][1:], '159': data['Disp. 3'][1:],
                  '175': data['Disp. 4'][1:], '179': data['Disp. 5'][1:]}
    dict_mu = {'111': mu_111, '154': mu_154, '159': mu_159, '175': mu_175, '179': mu_179}
    dict_std = {'111': std_111, '154': std_154, '159': std_159, '175': std_175, '179': std_179}

    for ip in dict_mu.keys():
        plt.figure()

        plt.hist(dict_delta[ip], density=True, bins=25, alpha=0.6, color='green')

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, dict_mu[ip], dict_std[ip])
        plt.plot(x, p, 'k', linewidth=2)
        plt.savefig(f'hist_{ip}.pdf', format='pdf')
        plt.clf()


def generate_correlation_plot(data: DataFrame):
    path = f'plots/{data_type}'
    corr = data.corr()

    # d2 = {'Disp. 1': data_111['offset'], 'Disp. 2': data_154['offset'], 'Disp. 3': data_159['offset'],
    #      'Disp. 4': data_175['offset'], 'Disp. 5': data_179['offset']}
    # data2 = pd.DataFrame(d2)
    # corr2 = data2.corr()

    plt.figure()
    plt.matshow(corr)
    plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns)
    plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns)
    plt.colorbar()

    plt.savefig(f'{path}/correlation_plot.pdf', format='pdf')
    plt.clf()


def generate_offset_plot(data: DataFrame):
    path = f'plots/{data_type}'

    plt.figure()
    for disp in data:
        plt.plot(data[disp]['time'], data[disp]['offset'] / np.power(10, 6), color=colors[disp], label=disp)

    """
    vertical_lines = [4000,  16500, 29000, 41000]
    for line in vertical_lines:
        plt.axvline(x=line, color='black', linestyle=':')
    """

    plt.xlabel('Tiempo (\\si{\\second})')
    plt.ylabel('Offset (\\si{\\milli\\second})')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.savefig(f'{path}/offset_plot_45.pdf', format='pdf', bbox_inches='tight')
    plt.clf()


def generate_delta_offset_plot(data: dict):
    path = f'plots/{data_type}'
    plt.figure()

    for key in data:
        plt.plot(delta_offsets[key]['time'], delta_offsets[key]['delta_offset'] / np.power(10, 3), color=colors[key],
                    label=key)

    plt.xlabel('Tiempo (\\si{\\second})')
    plt.ylabel('$\\Delta$Offset (\\si{\\micro\\second})')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    plt.savefig(f'{path}/delta_offset_plot.pdf', format='pdf', bbox_inches='tight')
    plt.clf()


def main():
    global data, data_type, offset_diffs, delta_offsets

    data_type = 'parallel'

    # Leer datos
    data = read_data(data_type)

    # Calculamos los offsets
    offset_diffs = generate_offset_dict(data)

    # Calculamos los incrementos de los offsets, eliminando outliers
    delta_offsets = generate_delta_offset_dict(data, offset_diffs)


if __name__ == "__main__":
    main()

