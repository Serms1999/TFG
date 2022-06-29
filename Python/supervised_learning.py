#!/usr/bin/env python3

import argparse
import os
import platform
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from globals import *

os.environ["PATH"] += os.pathsep + latexPaths[platform.system()]

mpl.use("pgf")
mpl.rcParams.update(pgf_with_latex)

data_type = ''
path = ''
state = 27
gen = ''
gen_final = False
models_to_train_list = []
metrics_results = None
confusion_matrices = {}

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

models = {
    'random_forest': {
        'model': RandomForestClassifier(n_jobs=10, max_depth=1000, random_state=state),
        'param_grid': {
            'n_estimators': np.arange(100, 1500, 100),
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None]
        }
    },
    'mlp': {
        'model': MLPClassifier(max_iter=2000, random_state=state),
        'param_grid': {
            'hidden_layer_sizes': [(5,), (10,), (15,), (50,), (100,), (500,),
                                   (5, 2), (10, 2), (15, 2), (50, 2), (100, 2), (500, 2),
                                   (5, 3), (10, 3), (15, 3), (50, 3), (100, 3), (500, 5)],
            'activation': ['relu', 'logistic', 'tanh'],
            'solver': ['lbfgs', 'sgd', 'adam']
        }
    },
    'naive_bayes': {
        'model': GaussianNB(),
        'param_grid': {}
    },
    'knn': {
        'model': KNeighborsClassifier(n_jobs=10),
        'param_grid': {
            'n_neighbors': np.arange(1, 6, 1),
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute']
        }
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(random_state=state),
        'param_grid': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [5000, 10000, 20000, 30000]
        }
    },
    'svm_linear': {
        'model': LinearSVC(multi_class='ovr', max_iter=5000, random_state=state, loss='squared_hinge', dual=False),
        'param_grid': {
            # 'loss': ['hinge', 'squared_hinge'],
            # 'C': np.linspace(0, 2, 20),
            # 'dual': [True,  ]
            'C': np.linspace(0, 2, 100)
        }
    }
}


def read_stats(data_type: str):
    fname = 'stats'
    if delta:
        fname = 'delta_' + fname

    df = pd.read_csv(f'stats/{data_type}/{fname}.csv', sep=',', index_col=0)
    df.index = np.array(df.index) - 1
    return df


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


def get_model_args(model_name: str) -> tuple:
    model = models[model_name]

    return model['model'], model['param_grid']


def print_results(model_name, y_test, y_pred, best_params=None):
    global metrics_results, confusion_matrices

    results = pd.DataFrame.from_dict({
        0: [model_name,
            metrics.accuracy_score(y_test, y_pred),
            metrics.recall_score(y_test, y_pred, average='macro'),
            metrics.f1_score(y_test, y_pred, average='macro')
        ]
    }, columns=['model_name', 'accuracy', 'recall', 'f-score'], orient='index')

    print(f'Accuracy: {results.iloc[0]["accuracy"]}')
    print(f'Matriz de confusion')
    unique_label = np.unique([y_test, y_pred])
    cm = metrics.confusion_matrix(y_test, y_pred, labels=unique_label)
    cmtx = pd.DataFrame(
        cm, index=['true:{:}'.format(x) for x in unique_label],
        columns=['pred:{:}'.format(x) for x in unique_label]
    )
    print(cmtx)
    if best_params is not None:
        print('Mejores hiperparametros')
        print(best_params)

    if metrics_results is None:
        metrics_results = results
    else:
        metrics_results = pd.concat([metrics_results, results], ignore_index=True)
    confusion_matrices[model_name] = cm


def find_correlation(X, cutoff, plot=False):
    def plot_corr(fname, plot_figure=plot):
        if plot_figure:
            plot_path = f'plots/{data_type}'
            plt.figure()
            plt.matshow(corr_matrix, cmap='Blues')
            plt.xticks(range(X.select_dtypes(['number']).shape[1]), X.select_dtypes(['number']).columns,
                       rotation=45)
            plt.yticks(range(X.select_dtypes(['number']).shape[1]), X.select_dtypes(['number']).columns)
            plt.colorbar()
            plt.clim(-1, 1)

            plt.savefig(f'{plot_path}/{fname}.pdf', format='pdf')
            plt.clf()

    # Generamos la matriz de correlacion
    corr_matrix = X.corr()

    plot_corr(fname='correlacion_stats')

    # Nos quedamos con el valor absoluto de la correlacion
    corr_matrix = corr_matrix.abs()

    # Seleccionamos solo el triangulo superior
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Buscamos las correlaciones por encima de cutoff
    to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]

    # Quitamos las variables correladas
    X.drop(to_drop, axis=1, inplace=True)

    # Volvemos a generar la matriz de correlacion
    corr_matrix = X.corr()

    plot_corr(fname='correlacion_stats_ftred')

    return X


def train_model(model_name: str, x_train, x_test, y_train, y_test, generate=True):
    print('-' * 10 + f' {model_name} ' + '-' * 10)

    filename = model_name
    if delta:
        filename = 'delta_' + filename

    if not generate:
        CV_model = load(f'{path}/{filename}.joblib')

    else:
        model, grid = get_model_args(model_name)
        CV_model = GridSearchCV(param_grid=grid, estimator=model, cv=5, scoring='accuracy', n_jobs=-1)
        # CV_model = RandomizedSearchCV(param_distributions=grid, estimator=model, cv=5, scoring='accuracy', n_jobs=10,
        # n_iter=10, random_state=state)
        CV_model.fit(x_train, y_train)

        dump(CV_model, f'{path}/{filename}.joblib')

    model = CV_model.best_estimator_

    results = pd.concat([pd.DataFrame(CV_model.cv_results_["params"]),
                         pd.DataFrame(CV_model.cv_results_["mean_test_score"],
                                      columns=["Accuracy"])], axis=1)
    results.to_csv(f'{path}/{filename}_results.csv', index=False, header=True)

    y_pred = model.predict(x_test)
    print_results(model_name, y_pred=y_pred, y_test=np.array(y_test.array), best_params=CV_model.best_params_)


def final_model_train(x_train, x_test, y_train, y_test, generate=True):
    print('-' * 10 + ' Final Model ' + '-' * 10)

    filename = 'final_model'
    if delta:
        filename = 'delta_' + filename

    if generate:
        if data_type == 'parallel':
            rfc = RandomForestClassifier(n_jobs=-1,
                                         max_depth=1000,
                                         n_estimators=100,
                                         random_state=state,
                                         criterion='gini',
                                         max_features='sqrt')
        elif data_type == 'individual':
            rfc = RandomForestClassifier(n_jobs=-1,
                                         max_depth=1000,
                                         n_estimators=100,
                                         random_state=state,
                                         criterion='entropy',
                                         max_features=None)

        rfc.fit(x_train, y_train)
        dump(rfc, f'{path}/{filename}.joblib')

    else:
        rfc = load(f'{path}/{filename}.joblib')

    y_pred = rfc.predict(x_test)
    print_results('final_model', y_pred=y_pred, y_test=np.array(y_test.array))


def main():
    global delta, data_type, path, metrics_results

    models_to_train = get_models()

    data = read_stats(data_type)
    data_var_out = 'Device'
    data_vars_in = list(data.columns)
    data_vars_in.remove(data_var_out)

    X = data[data_vars_in]
    y = data[data_var_out]

    # Eliminamos las variables correladas
    X_filtered = find_correlation(X.copy(), cutoff=0.9, plot=True)

    # Dividimos el dataframe en una particion de train y otra de test
    X_train, X_test, y_train, y_test = train_split_reordered(X_filtered, y, _test_size=0.3,
                                                             _random_state=state,
                                                             _shuffle=True)

    # Para encontrar los parametros adecuados trabajamos con un conjunto menor del conjunto de entrenamiento
    X_train_reduced, y_train_reduced = reduce_train_set(X_train, y_train, size=0.35)

    # Generamos las particiones de entrenamiento y validacion
    X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_split_reordered(X_train_reduced,
                                                                                         y_train_reduced,
                                                                                         _test_size=0.3,
                                                                                         _random_state=state,
                                                                                         _shuffle=True)

    # Realizamos los entrenamientos

    for model_name, gen in models_to_train.items():
        train_model(model_name, X_train_sample, X_test_sample, y_train_sample, y_test_sample, generate=gen)

    final_model_train(X_train, X_test, y_train, y_test, generate=gen_final)

    # Guardamos los resultados
    metrics_results.to_csv(f'{path}/supervised_results.csv', index=False)
    dump(confusion_matrices, f'{path}/unsupervised_matrices.joblib')


if __name__ == "__main__":
    main()
