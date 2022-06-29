import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Latex configuration

latexPaths = {'Linux': '/usr/local/texlive/2021/bin/x86_64-linux',  # Linux
              'Darwin': '/usr/local/texlive/2022/bin/universal-darwin'}  # Mac

pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "lualatex",  # change this if using xelatex or lualatex
    "text.usetex": True,  # use LaTeX to write all text
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots
    "font.sans-serif": [],  # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts
    "xtick.labelsize": 8,  # a little smaller
    "ytick.labelsize": 8,
    "pgf.preamble": r"\usepackage{siunitx} \usepackage[T1]{fontenc}"
                    r" \usepackage[utf8x]{inputenc}"  # using this preamble
}

# Graphs colors

colors = {'Disp. 1': '#005780', 'Disp. 2': '#6e60a7',
          'Disp. 3': '#d0599d', 'Disp. 4': '#ff6964', 'Disp. 5': '#ffa600'}


def stats_array(x, names):
    serie = pd.Series(x)
    stats = []
    stats += [serie.sum()]
    stats += [serie.mean()]
    stats += [serie.median()]
    stats += [serie.mode()[0]]
    stats += [serie.std()]
    stats += [serie.quantile(0.75) - serie.quantile(0.25)]
    stats += [serie.kurt()]
    stats += [serie.skew()]
    stats += [serie.max()]
    stats += [serie.min()]

    return pd.DataFrame(dict(zip(names, stats)), index=[0])


def statistics(data, names):
    df = pd.DataFrame(columns=names)
    for array in data:
        df = pd.concat([df, stats_array(array, names)], ignore_index=True)

    return df


def slice_window(A, k):
    array = np.array(A)
    return [array[i:i + k] for i in range(len(array) - k + 1)]


def split_array(A, k):
    array = np.array(A)
    return [array[i * k:(i + 1) * k] for i in range(len(array) // k)]


def reduce_train_set(X_train, y_train, size):
    num_sample = int(np.floor(len(X_train.index) * size))
    return X_train.head(num_sample), y_train.head(num_sample)


def reorder_dataframes_by_index(*args):
    return [df.sort_index() for df in args]


def train_split_reordered(X, y, _test_size=None, _train_size=None, _random_state=None, _shuffle=True, _stratify=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_test_size, train_size=_train_size,
                                                        random_state=_random_state, shuffle=_shuffle,
                                                        stratify=_stratify)

    X_train, X_test, y_train, y_test = reorder_dataframes_by_index(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test


def read_data(path):
    data = dict()
    devices_ip = [111, 154, 159, 175, 179]

    for i, ip in enumerate(devices_ip):
        key = f'Disp. {i + 1}'
        data[key] = pd.read_csv(f'traces/{path}/captura_{ip}.csv', sep=';')

        data[key]['offset'] -= data[key]['offset'][0]
        data[key]['time'] /= np.power(10, 9)

    return data
