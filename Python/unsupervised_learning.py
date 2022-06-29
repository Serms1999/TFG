#!/usr/bin/env python3

import argparse
import itertools
import os
import platform
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from globals import *

os.environ["PATH"] += os.pathsep + latexPaths[platform.system()]

mpl.use("pgf")
mpl.rcParams.update(pgf_with_latex)

data_type = ''
path = ''
state = 27
gen = ''
gen_final = False
device_stats = {}
models_to_train_list = []
metrics_results = None
confusion_matrices = {}
data_dev = {}

parser = argparse.ArgumentParser()
parser.add_argument('--delta', dest='delta', action='store_true', help='Usar desviación en cada instante')
parser.add_argument('--no-delta', dest='delta', action='store_false', help='Usar desviación acumulada')
parser.set_defaults(delta=True)
parser.add_argument('-t', '--data_type', type=str, choices=['individual', 'parallel'], default='parallel',
                    required=False, help='Elegir captura de datos secuencial o paralela')
parser.add_argument('-p', '--path', type=str, required=False)
parser.add_argument('-g', '--generate', type=str, choices=['all', 'final', 'adjust', 'none', 'custom'],
                    help='Modelos que entrenar', default='none', required=False)
parser.add_argument('-m', '--models',
                    type=lambda s: re.split(' |, ', s),
                    required=False,
                    help='Lista de modelos a entrenar separados por | o coma',
                    default=[])
args = parser.parse_args()
delta = args.delta
data_type = args.data_type
if args.path:
    path = args.path
else:
    path = f'models/{data_type}'
gen = args.generate
if gen == 'custom':
    models_to_train_list = args.models

models = {}

def get_models():
    global gen_final
    d = {}
    if gen == 'all':
        for m in models:
            d[m] = True
        gen_final = True
    elif gen == 'adjust':
        for m in models:
            d[m] = True
    elif gen == 'final':
        gen_final = True
        for m in models:
            d[m] = False
    elif gen == 'none':
        for m in models:
            d[m] = False
    else:
        for m in models:
            d[m] = m in models_to_train_list
    return d


def read_stats(data_type: str, fname='stats'):
    if delta:
        fname = 'delta_' + fname

    df = pd.read_csv(f'stats/{data_type}/{fname}.csv', sep=',', index_col=0)
    df.index = np.array(df.index) - 1
    return df


def generate_device_dict(data: pd.DataFrame):
    pattern = [-1] * len(data['Device'].unique())
    device_stats = {}

    for i, dev in enumerate(data['Device'].unique()):
        pattern_dev = pattern.copy()
        pattern_dev[i] = 1

        column = np.array([1 if x == dev else -1 for x in data['Device']])
        device_stats[dev] = data.drop(columns='Device').join(pd.DataFrame({dev: column}))

    return device_stats


def IsolationForestTrain(data: pd.DataFrame, device_stats: dict):
    global metrics_results, confusion_matrices
    for i, dev in enumerate(['Disp. 1', 'Disp. 2', 'Disp. 3', 'Disp. 4', 'Disp. 5']):
        data_dev = read_stats(data_type, fname=f'statsDev{i+1}')

        data_var_out = 'Device'
        data_vars_in = list(data.columns)
        data_vars_in.remove(data_var_out)

        """
        X = device_stats[dev][data_vars_in]
        y = (device_stats[dev][device_stats[dev][dev] == -1][dev]
        """
        aux = device_stats[dev][device_stats[dev][dev] == -1]
        X = aux[data_vars_in]
        # y = aux[dev]
        y = pd.DataFrame({'Device': np.repeat(-1, len(aux))})

        X_dev = data_dev[data_vars_in]
        y_dev = pd.DataFrame({'Device': np.repeat(1, len(data_dev))})

        # Dividimos el dataframe en una particion de train y otra de test
        X_train_dev, X_test_dev, _, y_test_dev = train_split_reordered(X_dev, y_dev, _test_size=0.2,
                                                                 _random_state=state,
                                                                 _shuffle=True)

        _, X_test, _, y_test = train_split_reordered(X, y, _test_size=0.2,
                                                      _random_state=state,
                                                      _shuffle=True)

        model = IsolationForest(n_estimators=100, contamination=0.05, n_jobs=-1, random_state=state, warm_start=True)

        # Returns 1 of inliers, -1 for outliers
        model.fit(X_train_dev.values)
        y_pred_dev = model.predict(X_test_dev.values)
        y_pred = model.predict(X_test.values)

        results = pd.DataFrame.from_dict({
            0: [dev,
                'Isolation Forest',
                metrics.recall_score(y_test_dev, y_pred_dev),
                metrics.recall_score(y_test, y_pred, pos_label=-1)
                ]
        }, columns=['Device', 'Algorithm', 'Recall', 'TNR'], orient='index')

        if metrics_results is None:
            metrics_results = results
        else:
            metrics_results = pd.concat([metrics_results, results], ignore_index=True)

        unique_label = np.unique([y_test.Device.values, y_pred])
        confusion_matrices[dev] = (metrics.confusion_matrix(y_test, y_pred, labels=unique_label),
                                   metrics.confusion_matrix(y_test_dev, y_pred_dev, labels=unique_label), unique_label)


def LOFTrain(data: pd.DataFrame, device_stats: dict):
    global metrics_results
    for i, dev in enumerate(['Disp. 1', 'Disp. 2', 'Disp. 3', 'Disp. 4', 'Disp. 5']):
        data_dev = read_stats(data_type, fname=f'statsDev{i+1}')

        data_var_out = 'Device'
        data_vars_in = list(data.columns)
        data_vars_in.remove(data_var_out)

        """
        X = device_stats[dev][data_vars_in]
        y = device_stats[dev][device_stats[dev][dev] == -1][dev]
        """
        aux = device_stats[dev][device_stats[dev][dev] == -1]
        X = aux[data_vars_in]
        # y = aux[dev]
        y = pd.DataFrame({'Device': np.repeat(-1, len(aux))})

        X_dev = data_dev[data_vars_in]
        y_dev = pd.DataFrame({'Device': np.repeat(1, len(data_dev))})

        # Dividimos el dataframe en una particion de train y otra de test
        X_train_dev, X_test_dev, _, y_test_dev = train_split_reordered(X_dev, y_dev, _test_size=0.2,
                                                                 _random_state=state,
                                                                 _shuffle=True)

        _, X_test, _, y_test = train_split_reordered(X, y, _test_size=0.2,
                                                      _random_state=state,
                                                      _shuffle=True)

        #iforest = IsolationForest(n_estimators=100, contamination=0.05, n_jobs=-1, random_state=state, warm_start=True)
        model = LocalOutlierFactor(n_neighbors=100, novelty=True)

        # Returns 1 of inliers, -1 for outliers
        model.fit(X_train_dev.values)
        y_pred_dev = model.predict(X_test_dev.values)
        y_pred = model.predict(X_test.values)

        results = pd.DataFrame.from_dict({
            0: [dev,
                'LOF',
                metrics.recall_score(y_test_dev, y_pred_dev),
                metrics.recall_score(y_test, y_pred, pos_label=-1)
                ]
        }, columns=['Device', 'Algorithm', 'Recall', 'TNR'], orient='index')

        metrics_results = pd.concat([metrics_results, results], ignore_index=True)


def OC_SVMTrain(data: pd.DataFrame, device_stats: dict):
    global metrics_results
    for i, dev in enumerate(['Disp. 1', 'Disp. 2', 'Disp. 3', 'Disp. 4', 'Disp. 5']):
        data_dev = read_stats(data_type, fname=f'statsDev{i+1}')

        data_var_out = 'Device'
        data_vars_in = list(data.columns)
        data_vars_in.remove(data_var_out)

        aux = device_stats[dev][device_stats[dev][dev] == -1]
        X = aux[data_vars_in]
        #y = aux[dev]
        y = pd.DataFrame({'Device': np.repeat(-1, len(aux))})

        X_dev = data_dev[data_vars_in]
        y_dev = pd.DataFrame({'Device': np.repeat(1, len(data_dev))})

        # Dividimos el dataframe en una particion de train y otra de test
        X_train_dev, X_test_dev, _, y_test_dev = train_split_reordered(X_dev, y_dev, _test_size=0.2,
                                                                 _random_state=state,
                                                                 _shuffle=True)

        _, X_test, _, y_test = train_split_reordered(X, y, _test_size=0.2,
                                                      _random_state=state,
                                                      _shuffle=True)

        model = OneClassSVM()

        # Returns 1 of inliers, -1 for outliers
        model.fit(X_train_dev.values)
        y_pred_dev = model.predict(X_test_dev.values)
        y_pred = model.predict(X_test.values)

        results = pd.DataFrame.from_dict({
            0: [dev,
                'OC-SVM',
                metrics.recall_score(y_test_dev, y_pred_dev),
                metrics.recall_score(y_test, y_pred, pos_label=-1)
                ]
        }, columns=['Device', 'Algorithm', 'Recall', 'TNR'], orient='index')

        metrics_results = pd.concat([metrics_results, results], ignore_index=True)


def metrics_to_latex():
    global metrics_results
    path = '../Proyecto_Latex'

    lines = []
    lines += [r'\begin{tabular}{lcccccc}']
    lines += [r'    \toprule']
    lines += [r'     & \multicolumn{2}{c}{\texttt{Isolation Forest}} &'
                    r' \multicolumn{2}{c}{\texttt{Local Outlier Factor}} &'
                    r' \multicolumn{2}{c}{\texttt{OneClass-SVM}} \\']
    lines += [r'    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}']
    lines += [r'    & $Recall$ & $TNR$ & $Recall$ & $TNR$ & $Recall$ & $TNR$ \\']
    lines += [r'    \midrule']

    for dev in metrics_results['Device'].unique():
        aux = metrics_results[metrics_results['Device'] == dev].drop(columns='Device')

        line_aux = '     ' + dev

        for alg in aux['Algorithm'].unique():
            aux2 = aux[aux['Algorithm'] == alg].drop(columns='Algorithm')
            line_aux += fr' & {(float(aux2["Recall"]) * 100):.2f}\% & {(float(aux2["TNR"]) * 100):.2f}\%'

        line_aux += r' \\'
        lines += [line_aux]

    lines += [r'    \bottomrule']
    lines += [r'\end{tabular}']

    with open(f'{path}/unsupervised_table.tex', 'w') as file:
        file.writelines([f'{line}\n' for line in lines])


def plot_confusion_matrix(model_name, cm, classes=None, figsize=(10, 10), text_size=10, plot_colorbar=True):
    path_plots = f'plots/{data_type}'
    #classes --> number of classes / labels in your dataset (10 classes for this example)
    #figsize --> (10 , 10) has been set as a default figsize. Can be adjusted for our needs.
    #text_size --> size of the text

    # Setting the default figsize
    figsize = figsize
    # Create the confusion matrix from sklearn
    #cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize our confusion matrix

    # Number of clases
    n_classes = cm.shape[0]

    # Making our plot pretty
    fig, ax = plt.subplots(figsize=figsize)
    # Drawing the matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues, vmin=0, vmax=1)

    # Set labels to be classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label axes
    ax.set(title=f'Matríz de confusión {model_name}',
           xlabel='Valor predicho',
           ylabel='Valor real',
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)
    # Set the xaxis labels to bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Adjust the label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    # Set threshold for different colors
    threshold = cm.min() + 1

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]}',
                 horizontalalignment='center',
                 color='white' if cm[i, j] > threshold else 'black',
                 size=10
                 )

    if plot_colorbar:
        # Plot colorbar
        #ax.set_visible(False)
        cbar = plt.colorbar(cax)
        cbar.set_ticks(np.linspace(0, 1, 6))
        cbar.set_ticklabels([f'{int(x)}%' for x in np.linspace(0, 1, 6) * 100])
        #plt.savefig(f'{path_plots}/colorbar_matrices.pdf', format='pdf', bbox_inches='tight')
        #ax.set_visible(True)

    plt.savefig(f'{path_plots}/{model_name}_matrix.pdf', format='pdf', bbox_inches='tight')


def main():
    global data_dev, delta, data_type, path, metrics_results, device_stats

    _ = get_models()

    data = read_stats(data_type)
    device_stats = generate_device_dict(data)

    if gen_final:
        IsolationForestTrain(data, device_stats)
        #LOFTrain(data, device_stats)
        #OC_SVMTrain(data, device_stats)

        # Guardamos los resultados
        metrics_results.to_csv(f'{path}/unsupervised_results.csv', index=False)

    else:
        metrics_results = pd.read_csv(f'{path}/unsupervised_results.csv', index_col=None)


if __name__ == "__main__":
    main()
