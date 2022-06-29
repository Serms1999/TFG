#!/usr/bin/env python3

import argparse
import itertools
import os
import platform
import re

from joblib import load
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from globals import latexPaths, pgf_with_latex, colors

os.environ["PATH"] += os.pathsep + latexPaths[platform.system()]

mpl.use("pgf")
mpl.rcParams.update(pgf_with_latex)

path_models = ''
path_plots = ''
models = {}
metrics_results = {}
confusion_matrices = {}

parser = argparse.ArgumentParser()
parser.add_argument('--delta', dest='delta', action='store_true', help='Usar desviación en cada instante')
parser.add_argument('--no-delta', dest='delta', action='store_false', help='Usar desviación acumulada')
parser.set_defaults(delta=True)
parser.add_argument('-t', '--data_type', type=str, choices=['individual', 'parallel'], default='parallel',
                    required=False, help='Elegir captura de datos secuencial o paralela')
args = parser.parse_args()
delta = args.delta
data_type = args.data_type

names = {
    'random_forest': 'Random Forest',
    'mlp': 'MLP',
    'naive_bayes': 'Naive Bayes',
    'knn': 'KNN',
    'decision_tree': 'Árboles de Decisión',
    'svm_linear': 'SVM',
    'final_model': 'Modelo Final'
}


def read_models():
    global models

    model_name = ''
    if delta:
        model_name = 'delta_'

    for model in names:
        models[model] = load(f'{path_models}/{model_name + model}.joblib')

def read_results():
    global metrics_results, confusion_matrices
    """
    try:
        with open(f'{path_models}/model_results.csv', 'r') as file:
            for line in file.readlines():
                model, value = re.split(',', line.rstrip('\n'))
                metrics_results[model] = float(value)
    except IOError:
        raise IOError
    """

    metrics_results = pd.read_csv(f'{path_models}/model_results.csv', index_col=0)
    confusion_matrices = load(f'{path_models}/matrices.joblib')


def plot_accuracy_barplot():
    global metrics_results
    data = metrics_results.copy()
    del data['final_model']

    plt.figure()
    for key, value in data.items():
        plt.bar(names[key], value, color=colors['Disp. 1'])

    plt.title('Resultados obtenidos')
    plt.xlabel('Modelos')
    plt.ylabel('Accuracy')

    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(data.values()):
        plt.text(xlocs[i] - 0.25, v + 0.01, '{:.2f}%'.format(v * 100))

    plt.yticks(ticks=np.linspace(0, 1, 6), labels=[f'{int(y)}%' for y in np.linspace(0, 1, 6) * 100])

    plt.savefig(f'{path_plots}/accur_results.pdf', format='pdf')
    plt.clf()


def plot_all_matrices():
    for model, cm in confusion_matrices.items():
        if model == 'final_model':
            plot_confusion_matrix(model, cm, classes=colors.keys(), text_size=25, plot_colorbar=True)
        else:
            plot_confusion_matrix(model, cm, classes=colors.keys(), text_size=25, plot_colorbar=False)


def plot_confusion_matrix(model_name, cm, classes=None, figsize=(10, 10), text_size=10, plot_colorbar=True):
    global names
    """
    classes --> number of classes / labels in your dataset (10 classes for this example)
    figsize --> (10 , 10) has been set as a default figsize. Can be adjusted for our needs.
    text_size --> size of the text
    """

    # Setting the default figsize
    figsize = figsize
    # Create the confusion matrix from sklearn
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize our confusion matrix

    # Number of clases
    n_classes = cm.shape[0]

    # Making our plot pretty
    fig, ax = plt.subplots(figsize=figsize)
    # Drawing the matrix plot
    cax = ax.matshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)

    # Set labels to be classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label axes
    ax.set(title=f'Matríz de confusión {names[model_name]}',
           xlabel='Etiqueta predicha',
           ylabel='Etiqueta real',
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
    threshold = (cm.max() + cm.min()) / 2

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)',
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


def random_forest_analysis():
    if delta:
        data = pd.read_csv(f'{path_models}/delta_random_forest_results.csv')
    else:
        data = pd.read_csv(f'{path_models}/random_forest_results.csv')
    data['max_features'] = data['max_features'].fillna('none')

    accuracy_means = pd.DataFrame(columns=['criterion', 'max_features', 'mean_accuracy'])

    x = [(criterion, max_features) for criterion in data['criterion'].unique()
         for max_features in data['max_features'].unique()]

    for criterion, max_features in x:
        mean_accuracy = data[(data['criterion'] == criterion) & (data['max_features'] == max_features)]['Accuracy'].mean()
        row = {'criterion': criterion, 'max_features': max_features, 'mean_accuracy': mean_accuracy}
        accuracy_means = accuracy_means.append(row, ignore_index=True)

    labels = accuracy_means['max_features'].unique()
    d = {}
    for criterion in accuracy_means['criterion'].unique():
        d[criterion] = np.array(accuracy_means.loc[accuracy_means['criterion'] == criterion, 'mean_accuracy']).tolist()

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, d['gini'], width, label='gini', color=colors['Disp. 1'])
    ax.bar(x + width / 2, d['entropy'], width, label='entropy', color=colors['Disp. 5'])

    plt.ylim(0.4, 0.45)
    ax.set_xlabel('\\texttt{max\_features}')
    ax.set_ylabel('Accuracy promedio')
    ax.set_title('Comparativa hiperparámetros')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title='\\texttt{criterion}', bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    fname = 'random_forest_results.pdf'
    if delta:
        fname = 'delta_' + fname

    plt.savefig(f'{path_plots}/{fname}', format='pdf', bbox_inches='tight')

    # Fijamos el hiperparametro
    criterion = 'gini'

    data_crit = data[data['criterion'] == criterion]
    cols = list(data.columns)
    cols.remove('criterion')
    data_crit = data_crit[cols]
    data_crit = data_crit[data_crit['max_features'] != 'none']

    labels = data_crit['n_estimators'].unique()
    d = {}
    for max_features in data_crit['max_features'].unique():
        d[max_features] = np.array(data_crit.loc[data_crit['max_features'] == max_features, 'Accuracy']).tolist()

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, d['sqrt'], width, label='sqrt', color=colors['Disp. 1'])
    ax.bar(x + width / 2, d['log2'], width, label='log2', color=colors['Disp. 5'])

    plt.ylim(0.42, 0.45)
    ax.set_xlabel('\\texttt{n\_estimators}')
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparativa hiperparámetros\n\\texttt{criterion = gini}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title='\\texttt{max_features}', bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

    fname = 'random_forest_results2.pdf'
    if delta:
        fname = 'delta_' + fname

    plt.savefig(f'{path_plots}/{fname}', format='pdf', bbox_inches='tight')


def main():
    global path_models, path_plots
    path_models = f'models/{data_type}'
    path_plots = f'plots/{data_type}'

    # Leemos los resultados
    read_results()

    # Cargamos los modelos
    read_models()


if __name__ == '__main__':
    main()
