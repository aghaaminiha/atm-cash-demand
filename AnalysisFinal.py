# ATM cash demand prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.api import graphics
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, SARIMAX
import warnings
import os

matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['axes.linewidth'] = 1.5
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------------------------
time_span = dict(Before=('1/20/2020', '1/21/2020', '2/19/2020'), During=('2/19/2020', '2/20/2020', '3/19/2020'))
grid_search = dict(status=False, param=['ARIMA', False], non_param_ds=['MLP_DS', False], non_param_rf=['MLP', False])
target_atm = 'ATM (mean)'
replicas = 5

# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------
ATMs = ['ATM 1', 'ATM 2', 'ATM 3', 'ATM (mean)']
iteration_method = ['Approximate', 'Updated']
pandemic_status = ['Before', 'During']
eval_metrics = ['MSE', 'POCID']


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------------------
def add_new_features(df):
    wnd_sd_hd = []
    num_hds_ahead = []
    for row in range(len(df)):
        wnd_sd_hd.append('No')
        num_hds_ahead.append(0)
        # ----------------------
        i, tomorrow = 0, df.loc[row, 'Tomorrow SD HD Wnd']
        while tomorrow == 'Yes':
            num_hds_ahead[row] += 1
            i += 1
            if row + i == len(df):
                tomorrow = 'No'
            else:
                tomorrow = df.loc[row + i, 'Tomorrow SD HD Wnd']
        if (row == len(df) - 2) or (row == len(df) - 1):
            num_hds_ahead[row] += 3
        # ----------------------
        today = df.loc[row, 'Weekday']
        wnd1 = row + 5 - today
        wnd2 = row + 6 - today
        if today == 7:
            wnd1, wnd2 = row + 5, row + 6
        if (wnd1 > len(df) - 1) or (wnd2 > len(df) - 1):
            wnd_sd_hd[row] = 'Yes'
            continue
        if (df.loc[wnd1, 'Special day'] == 'Yes') or (df.loc[wnd1, 'Holiday'] == 'Yes'):
            wnd_sd_hd[row] = 'Yes'
        elif (df.loc[wnd2, 'Special day'] == 'Yes') or (df.loc[wnd2, 'Holiday'] == 'Yes'):
            wnd_sd_hd[row] = 'Yes'
        # ----------------------
    df['n HDs ahead'] = num_hds_ahead
    df['Weekend SD HD'] = wnd_sd_hd
    return df


def clean_data(df):
    df = add_new_features(df)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    # ---------------------------------
    _scales = []
    for _atm in ATMs:
        sc = [df[_atm].min(), df[_atm].max()]
        _scales.append(sc)
    # ---------------------------------
    # df = df.reset_index(drop=True)
    return df, _scales


def read_data(file_name):
    sheet_names = pd.ExcelFile('{}.xlsx'.format(file_name)).sheet_names
    df = pd.DataFrame()
    for sheet_name in sheet_names:
        if sheet_name == sheet_names[0]:
            df = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name)
        else:
            df2 = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name)
            df = pd.concat([df, df2], ignore_index=True, sort=False)
    df, scale = clean_data(df)
    # ---------------------------------
    excel_output(df, root='', file_name='dataInput')
    return df, scale


def view_data(df):
    root = 'regression/rawData/features'
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    columns = ['Season', 'Month', 'Day of month', 'Weekday', 'n HDs ahead']
    for column in columns:
        df2, df3 = grouped_by_col(df, column=column)
        fig, ax = plt.subplots(1, figsize=(27, 9))
        _X = df2[column]
        _y1 = [i / j for i, j in zip(df2['ATM 1'], df3['ATM 1'])]
        _y2 = [i / j for i, j in zip(df2['ATM 2'], df3['ATM 2'])]
        _y3 = [i / j for i, j in zip(df2['ATM 3'], df3['ATM 3'])]
        _y4 = [i / j for i, j in zip(df2['ATM (mean)'], df3['ATM (mean)'])]
        if column == 'Weekday':
            for i in range(6):
                _y1.append(_y1[i])
                _y2.append(_y2[i])
                _y3.append(_y3[i])
                _y4.append(_y4[i])
            _y1, _y2, _y3, _y4 = _y1[6:], _y2[6:], _y3[6:], _y4[6:]
        plt.plot(_X, _y1, linestyle='-', linewidth=5.0, c='black', label='ATM 1')
        plt.plot(_X, _y2, linestyle='--', linewidth=5.0, c='black', label='ATM 2')
        plt.plot(_X, _y3, linestyle='-.', linewidth=5.0, c='black', label='ATM 3')
        plt.plot(_X, _y4, linestyle=':', linewidth=5.0, c='black', label='ATMs (mean)')
        # ---------------------------------
        plt.grid(axis='both', linewidth=0.5, zorder=0)
        if column == 'Season':
            _x_axis_labels = ['Spring', 'Summer', 'Fall', 'Winter']
            m = 5
        elif column == 'Month':
            _x_axis_labels = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']
            m = 6
        elif column == 'Day of month':
            _x_axis_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                              '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                              '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
            m = 7
        elif column == 'Weekday':
            _x_axis_labels = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            m = 6
        else:
            _x_axis_labels = [0, 1, 2, 3, 4, 5]
            m = 7
        x_axis_labels = _x_axis_labels
        x_axis_index = [i + 1 for i in np.arange(len(x_axis_labels))]
        if column == 'n HDs ahead':
            x_axis_index = [i for i in np.arange(len(x_axis_labels))]
        ax.set_xticks(x_axis_index)
        ax.set_xticklabels(x_axis_labels, fontsize=40)
        if column == 'n HDs ahead':
            plt.xlabel('Number of consecutive holidays ahead', fontsize=40, labelpad=17)
        # ------------
        ax.ticklabel_format(axis='y', style='sci', scilimits=(m, m))
        ax.yaxis.get_offset_text().set_fontsize(30)
        plt.yticks(fontsize=35)
        plt.ylabel('Cash withdrawal (IRI)', fontsize=35, labelpad=17)
        # ------------
        ax.tick_params(axis='both', pad=20)
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1.02), ncol=4, fontsize=32)
        plt.tight_layout()
        plt.savefig('{}/{}.png'.format(root, column))
        plt.close()


def grouped_by_col(df, column):
    df1 = df.groupby([column]).mean().reset_index(drop=False)
    df2 = df.groupby([column]).count().reset_index(drop=False)
    return df1, df2


def excel_output(_object, root, file_name):
    if root != '':
        _object.to_excel('{}/{}.xls'.format(root, file_name))
    else:
        _object.to_excel('{}.xls'.format(file_name))


# ---------------------------------------------------------------------------------------------------------------------
def encode_data(df):
    cat_index = df.columns[(df.dtypes == 'object').values]
    num_index = df.columns[(df.dtypes != 'object').values]
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    sc = MinMaxScaler()
    ct = make_column_transformer((ohe, cat_index), (sc, num_index), remainder='passthrough')
    ct.fit_transform(df)
    df2 = ct.transform(df)
    # ---------------------------------
    names = []
    for cat in cat_index:
        unique = df[cat].value_counts().sort_index()
        for name in unique.index:
            names.append('{}_{}'.format(cat, name))
    for num in num_index:
        names.append(num)
    # ---------------------------------
    df2 = pd.DataFrame(df2)
    df2.columns = names
    df2.index = df.index
    return df2


def collect_data(df, mode):
    df2 = pd.DataFrame()
    if mode == 'non_param_data_sequence':
        df2['weekAgo'] = df.loc[:, target_atm].shift(7)
        df2['6dayAgo'] = df.loc[:, target_atm].shift(6)
        df2['5dayAgo'] = df.loc[:, target_atm].shift(5)
        df2['4dayAgo'] = df.loc[:, target_atm].shift(4)
        df2['3dayAgo'] = df.loc[:, target_atm].shift(3)
        df2['2dayAgo'] = df.loc[:, target_atm].shift(2)
        df2['1dayAgo'] = df.loc[:, target_atm].shift(1)
        df2[target_atm] = df.loc[:, target_atm]
        df2 = df2.dropna()
    else:
        aux = df[target_atm]
        df2 = df.drop(['ID', 'ATM 1', 'ATM 2', 'ATM 3', 'ATM 4', 'ATM (mean)'], axis=1)
        df2[target_atm] = aux
        df2 = df2.dropna()
    return df2


def grid_search_param(_model):
    models = []
    if _model == 'ARIMA':
        for p in [1, 6, 7, 8, 9]:
            for d in [0, 1]:
                for q in [0, 1]:
                    models.append(('ARIMA', 'ARIMA({}_{}_{})'.format(p, d, q), (p, d, q)))
    else:
        for P in [0, 1]:
            for D in [0, 1]:
                for Q in [0, 1]:
                    models.append(('SARIMA', 'SARIMA({}_{}_{})'.format(P, D, Q), (P, D, Q, 7)))
    return models


def grid_search_non_param(_model):
    models = []
    _hp1 = {'MLP_DS': [(2,), (4,), (6,), (8,), (10,),
                       (2, 2), (4, 4), (6, 6), (8, 8), (10, 10),
                       (2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8), (10, 10, 10)],
            'SVM_DS': [1, 0.1, 0.01, 0.001, 0.0001],
            'RF_DS': [10, 50, 100, 200, 500],
            'KNN_DS': [3, 4, 5, 6, 7],
            'MLP': [(2,), (4,), (6,), (8,), (10,),
                    (2, 2), (4, 4), (6, 6), (8, 8), (10, 10),
                    (2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8), (10, 10, 10)],
            'SVM': [1, 0.1, 0.01, 0.001, 0.0001],
            'RF': [10, 50, 100, 200, 500],
            'KNN': [3, 4, 5, 6, 7]}
    _hp2 = {'MLP_DS': ['constant'],
            'SVM_DS': [1, 5, 10, 100, 1000],
            'RF_DS': [0.6, 0.7, 0.8, 0.9, 1.0],
            'KNN_DS': ['uniform', 'distance'],
            'MLP': ['constant'],
            'SVM': [1, 5, 10, 100, 1000],
            'RF': [0.6, 0.7, 0.8, 0.9, 1.0],
            'KNN': ['uniform', 'distance']}
    for n in _hp1[_model]:
        for m in _hp2[_model]:
            if _model == 'MLP_DS':
                models.append(('MLP_DS_{}_{}'.format(n, m), MLPRegressor(hidden_layer_sizes=n, learning_rate=m,
                                                                         max_iter=5000)))
            elif _model == 'SVM_DS':
                models.append(('SVM_DS_{}_{}'.format(n, m), SVR(gamma=n, C=m)))
            elif _model == 'RF_DS':
                models.append(('RF_DS_{}_{}'.format(n, m), RandomForestRegressor(n_estimators=n, max_features=m)))
            elif _model == 'KNN_DS':
                models.append(('KNN_DS_{}_{}'.format(n, m), KNeighborsRegressor(n_neighbors=n, weights=m)))
            elif _model == 'MLP':
                models.append(('MLP_{}_{}'.format(n, m), MLPRegressor(hidden_layer_sizes=n, learning_rate=m,
                                                                      max_iter=5000)))
            elif _model == 'SVM':
                models.append(('SVM_{}_{}'.format(n, m), SVR(gamma=n, C=m)))
            elif _model == 'RF':
                models.append(('RF_{}_{}'.format(n, m), RandomForestRegressor(n_estimators=n, max_features=m)))
            elif _model == 'KNN':
                models.append(('KNN_{}_{}'.format(n, m), KNeighborsRegressor(n_neighbors=n, weights=m)))
    return models


def split_y(_y, _pandemic):
    y_training = np.asarray(_y[:time_span[_pandemic][0]])
    y_testing = np.asarray(_y[time_span[_pandemic][1]:time_span[_pandemic][2]])
    return y_training, y_testing


def split_data(df, _pandemic):
    training = df[:time_span[_pandemic][0]]
    testing = df[time_span[_pandemic][1]:time_span[_pandemic][2]]
    x_training = training.iloc[:, 0:-1]
    x_testing = testing.iloc[:, 0:-1]
    y_training = np.asarray(training.iloc[:, -1])
    y_testing = np.asarray(testing.iloc[:, -1])
    return x_training, y_training, x_testing, y_testing


def moving_average(z, window):
    n, _L = len(z) - 1, 0
    for i in range(window):
        _L += z[n - i] / window
    return _L


def evaluation(_y_test, _y_pred, scoring):
    if scoring == 'MSE':
        result = ('MSE', mean_squared_error(_y_test, _y_pred))
    else:
        _sum = 0
        for t in range(1, len(_y_test)):
            _direction = (_y_test[t] - _y_test[t - 1]) * (_y_pred[t] - _y_pred[t - 1])
            if _direction > 0:
                _sum += 1
        result = ('POCID', 100 * _sum / (len(_y_test) - 1))
    return result


def tuned_model(_model, _hp1, _hp2):
    if _model == 'MLP_DS' or _model == 'MLP':
        _selected_model = MLPRegressor(hidden_layer_sizes=_hp1, max_iter=5000, random_state=5)
    elif _model == 'SVM_DS' or _model == 'SVM':
        _selected_model = SVR(gamma=_hp1, C=_hp2)
    elif _model == 'RF_DS' or _model == 'RF':
        _selected_model = RandomForestRegressor(n_estimators=_hp1, max_features=_hp2, random_state=5)
    else:
        _selected_model = KNeighborsRegressor(n_neighbors=_hp1, weights=_hp2)
    return _selected_model


# ---------------------------------------------------------------------------------------------------------------------
def raw_data_plot(_y, _atm):
    root = 'regression/rawData'
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    df = pd.DataFrame(_y)
    fig, ax = plt.subplots(1, figsize=(27, 9))
    ax.plot(df.index, df[_atm], c='gray', linewidth=2.0, label='Raw Data')
    # ---------------------------------
    plt.text(0.02, 0.9, '{}'.format(_atm),
             ha='left', va='center', transform=ax.transAxes, fontdict={'color': 'k', 'weight': 'bold', 'size': 32})
    # ---------------------------------
    plt.grid(axis='x', linewidth=0.7, zorder=0)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_xlim(pd.to_datetime('3/21/2017'), pd.to_datetime('3/19/2020'))
    ax.tick_params(axis='x', pad=20)
    # ax.set_xlim(df.index[0], df.index[len(df) - 1])
    ax.set_ylim(0, 1)
    plt.ylabel('Cash withdrawal (norm)', fontsize=35, labelpad=17)
    ax.tick_params(axis='both', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.02), ncol=2, fontsize=32)
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(root, _atm))
    plt.close()


def split_data_plot(_y, _atm, _start_date):
    root = 'regression/splitData'
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    df = pd.DataFrame(_y)
    fig, ax = plt.subplots(1, figsize=(27, 9))
    ax.plot(df.loc[:time_span['Before'][1], _atm], c='gray', linewidth=3.0, label='Train')
    ax.plot(df.loc[time_span['Before'][1]:time_span['Before'][2], _atm],
            c='indianred', linewidth=3.0, label='Test (Before COVID-19)')
    ax.plot(df.loc[time_span['Before'][2]:time_span['During'][2], _atm],
            c='steelblue', linewidth=3.0, label='Test (During COVID-19)')
    # ---------------------------------
    if _start_date == '12/15/2019':
        _figName = 'magnified'
    else:
        _figName = 'normal'
        plt.text(0.02, 0.9, '{}'.format(_atm),
                 ha='left', va='center', transform=ax.transAxes, fontdict={'color': 'k', 'weight': 'bold', 'size': 32})
    # ---------------------------------
    plt.grid(axis='x', linewidth=0.7, zorder=0)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_xlim(pd.to_datetime(_start_date), pd.to_datetime('3/19/2020'))
    ax.tick_params(axis='x', pad=20)
    ax.set_ylim(0, 1)
    plt.ylabel('Cash withdrawal (norm)', fontsize=35, labelpad=17)
    ax.tick_params(axis='both', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.02), ncol=3, fontsize=32)
    plt.tight_layout()
    plt.savefig('{}/{} ({}).png'.format(root, _atm, _figName))
    plt.close()


def correlation_plots(df, _atm, _diff_order, _fig_name):
    root = 'regression/rawData/statistics'
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    if _diff_order == 0:
        _y = df.loc[:, _atm]
    else:
        _y = df.loc[:, _atm].shift(_diff_order)
        _y = _y.dropna()
    fig, ax = plt.subplots(2, figsize=(27, 12))
    graphics.plot_acf(_y, lags=70, ax=ax[0])
    graphics.plot_pacf(_y, lags=70, ax=ax[1])
    # ---------------------------------
    ax[0].xaxis.set_tick_params(labelsize=30)
    y_axis_index = np.arange(-0.4, 1.4, 0.4)
    ax[0].set_yticks(y_axis_index)
    ax[0].set_yticklabels(['{:.1f}'.format(i) for i in y_axis_index], fontsize=25)
    ax[0].set_ylabel('Autocorrelation', fontsize=35, labelpad=17)
    ax[0].title.set_text('')
    ax[0].tick_params(axis='both', pad=10)
    ax[1].xaxis.set_tick_params(labelsize=30)
    ax[1].set_yticks(y_axis_index)
    ax[1].set_yticklabels(['{:.1f}'.format(i) for i in y_axis_index], fontsize=25)
    ax[1].set_ylabel('Partial Autocorrelation', fontsize=35, labelpad=17)
    ax[1].set_xlabel('Lag', fontsize=35, labelpad=7)
    ax[1].title.set_text('')
    ax[1].tick_params(axis='both', pad=10)
    # ---------------------------------
    plt.suptitle(t='({})  Differencing order = {}'.format(_fig_name, _diff_order),
                 x=0.98, y=0.92, ha='right', va='center', fontsize=50)
    plt.tight_layout()
    plt.savefig('{}/{}_{}.png'.format(root, _diff_order, _atm))
    plt.close()


def features_importance_plot(df, estimator, _atm, _iteration, _pandemic):
    root = 'regression/features/{}/{}'.format(_iteration, _pandemic)
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    names = df.columns
    imp = estimator.feature_importances_
    indices = np.argsort(imp)
    fig, ax = plt.subplots(1, figsize=(27, 9))
    plt.barh(range(len(indices)), imp[indices], color='black', align='center')
    x_axis_index = np.arange(0, 0.55, 0.05)
    ax.set_xticks(x_axis_index)
    ax.set_xticklabels(['{:.2f}'.format(i) for i in x_axis_index], fontsize=30)
    ax.set_xlabel('Relative Importance', fontsize=35, labelpad=17)
    plt.yticks(range(len(indices)), [names[i] for i in indices], fontsize=21)
    ax.tick_params(axis='both', pad=10)
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(root, _atm))
    plt.close()


def scores_plot(_iteration, _pandemic):
    root = 'regression/comparison/{}/{}'.format(_iteration, _pandemic)
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    n, m, _color, _figName = 0, 1, 'maroon', 'A'
    if _pandemic == pandemic_status[1]:
        n, _color, _figName = 1, 'steelblue', 'B'
    if _iteration == iteration_method[1]:
        m = 2
    # ---------------------------------
    mse_av, mse_sd = [i[m][1] for i in eval_scores[0][1][n][1]], [i[m][2] for i in eval_scores[0][1][n][1]]
    pocid_av, pocid_sd = [i[m][1] for i in eval_scores[1][1][n][1]], [i[m][2] for i in eval_scores[1][1][n][1]]
    fitness_av = [float(i) / (1 + 10 * float(j)) for i, j in zip(pocid_av, mse_av)]
    fitness_sd = [0.5 * (float(i) + float(j)) for i, j in zip(mse_sd, pocid_sd)]
    # ---------------------------------
    models = [i[0] for i in eval_scores[0][1][0][1]]
    x_axis_index = np.arange(len(models))
    bar_width = 0.25
    fig, ax1 = plt.subplots(1, figsize=(27, 9))
    mse = ax1.bar(x_axis_index - bar_width, mse_av, width=bar_width, color='lightgray', edgecolor='black', hatch='.',
                  zorder=3, yerr=mse_sd, capsize=5, align='center', ecolor='black', alpha=0.5)
    ax2 = ax1.twinx()
    fitness = ax2.bar(x_axis_index, fitness_av, width=bar_width, color='black', edgecolor='black', zorder=3,
                      yerr=fitness_sd, capsize=5, align='center', ecolor='black', alpha=0.5)
    pocid = ax2.bar(x_axis_index + bar_width, pocid_av, width=bar_width, color='white', edgecolor='black', hatch='/',
                    zorder=3, yerr=pocid_sd, capsize=5, align='center', ecolor='black', alpha=0.5)
    # ---------------------------------
    plt.text(0.02, 0.9, '({})'.format(_figName),
             ha='left', va='center', transform=ax1.transAxes, fontdict={'color': 'k', 'size': 57})
    # ---------------------------------
    plt.text(0.10, 0.9, '{} COVID-19'.format(_pandemic),
             ha='left', va='center', transform=ax1.transAxes, fontdict={'color': 'k', 'size': 35})
    # ---------------------------------
    ax1.grid(axis='y', linewidth=0.5, zorder=0)
    ax1.set_xticks(x_axis_index)
    ax1.set_xticklabels(models, fontsize=30, rotation=45)
    # ---------------------------------
    # y_axis_ 1 (MSE)
    y_axis_index = np.arange(0, 0.3, 0.05)
    ax1.set_yticks(y_axis_index)
    ax1.set_yticklabels(['{:.2f}'.format(i) for i in y_axis_index], fontsize=30)
    ax1.set_ylabel('MSE', fontsize=35, labelpad=17)
    # ---------------------------------
    # y_axis_2 (POCID) & (Fitness)
    y_axis_index = np.arange(0, 120, 20)
    ax2.set_yticks(y_axis_index)
    ax2.set_yticklabels(y_axis_index, fontsize=30)
    ax2.set_ylabel('POCID | Fitness', fontsize=35, labelpad=17)
    # ---------------------------------
    ax1.tick_params(axis='both', pad=20)
    ax2.tick_params(axis='y', pad=20)
    plt.legend([mse, fitness, pocid], ['MSE (Accuracy)', 'Fitness (higher is better)', 'POCID (Trend)'],
               loc='lower right', bbox_to_anchor=(1, 1.02), ncol=3, fontsize=32)
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(root, target_atm))
    plt.close()
    return mse_av, pocid_av, fitness_av


def summary_of_models(df, _iteration, _pandemic):
    root = 'regression/comparison'
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    df['{} {} mse_av'.format(_iteration, _pandemic)] = mse_average
    df['{} {} pocid_av'.format(_iteration, _pandemic)] = pocid_average
    df['{} {} fitness_av'.format(_iteration, _pandemic)] = fitness_average
    return df


def prediction_plot(_prediction, _pandemic, _iteration, _algorithm):
    root = 'regression/prediction/{}/{}/{}'.format(_iteration, _pandemic, target_atm)
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    column = '{} {} MSE'.format(_iteration, _algorithm)
    if _pandemic == pandemic_status[0]:
        _color = 'maroon'
    else:
        _color = 'steelblue'
    # ---------------------------------
    df = pd.DataFrame(y)
    fig, ax = plt.subplots(1, figsize=(27, 9))
    ax.plot(df.loc[:time_span[_pandemic][1], target_atm], c='gray', linewidth=3.0, label='Train')
    ax.plot(df.loc[time_span[_pandemic][1]:time_span[_pandemic][2], target_atm],
            c='black', linewidth=3.0, label='Test')
    ax.plot(_prediction.loc[time_span[_pandemic][1]:time_span[_pandemic][2], column],
            c=_color, linewidth=3.0, label='Prediction ({})'.format(_algorithm))
    # ---------------------------------
    plt.text(0.02, 0.9, '{} COVID-19'.format(_pandemic),
             ha='left', va='center', transform=ax.transAxes, fontdict={'color': 'k', 'size': 35})
    # ---------------------------------
    plt.grid(axis='y', linewidth=0.5)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_xlim(pd.to_datetime('12/15/2019'), pd.to_datetime('3/19/2020'))
    ax.set_ylim(0, 1)
    plt.ylabel('Cash withdrawal (norm)', fontsize=35, labelpad=17)
    ax.tick_params(axis='both', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.02), ncol=3, fontsize=30)
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(root, _algorithm))
    plt.close()


def prediction_top3_plot(_prediction, _pandemic, _iteration, _top3):
    root = 'regression/prediction/{}/{}'.format(_iteration, _pandemic)
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    if _pandemic == pandemic_status[0]:
        _figName = 'A'
    else:
        _figName = 'B'
    # ---------------------------------
    df = pd.DataFrame(y)
    fig, ax = plt.subplots(1, figsize=(27, 9))
    # ---------------------------------
    ax.plot(df.loc[:time_span[_pandemic][1], target_atm], c='gray', linewidth=3.0, label='Train')
    ax.plot(df.loc[time_span[_pandemic][1]:time_span[_pandemic][2], target_atm],
            c='black', linewidth=3.0, label='Test')
    ax.plot(_prediction.loc[time_span[_pandemic][1]:time_span[_pandemic][2], '{} {} MSE'.format(_iteration, _top3[0])],
            c='green', linewidth=3.0, label='{}'.format(_top3[0]))
    ax.plot(_prediction.loc[time_span[_pandemic][1]:time_span[_pandemic][2], '{} {} MSE'.format(_iteration, _top3[1])],
            c='purple', linewidth=3.0, label='{}'.format(_top3[1]))
    ax.plot(_prediction.loc[time_span[_pandemic][1]:time_span[_pandemic][2], '{} {} MSE'.format(_iteration, _top3[2])],
            c='orange', linewidth=3.0, label='{}'.format(_top3[2]))
    # ---------------------------------
    plt.text(0.02, 0.9, '({})'.format(_figName),
             ha='left', va='center', transform=ax.transAxes, fontdict={'color': 'k', 'size': 57})
    plt.text(0.10, 0.9, '{} COVID-19'.format(_pandemic),
             ha='left', va='center', transform=ax.transAxes, fontdict={'color': 'k', 'size': 35})
    # ---------------------------------
    plt.grid(axis='y', linewidth=0.5)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_xlim(pd.to_datetime('12/15/2019'), pd.to_datetime('3/19/2020'))
    ax.set_ylim(0, 1)
    ax.set_ylabel('Cash withdrawal (norm)', fontsize=35, labelpad=17)
    ax.tick_params(axis='both', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.02), ncol=5, fontsize=32)
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(root, target_atm))
    plt.close()


# --------------------------------------------------------------------------------------------------------------------
# BEGIN
# --------------------------------------------------------------------------------------------------------------------

# reading, encoding, and viewing data
dataInput, scales = read_data('dataATM')
atm_encoded = encode_data(dataInput)
view_data(dataInput)

# data summary
for atm in ATMs:
    raw_data_plot(atm_encoded[atm], atm)
    split_data_plot(atm_encoded[atm], atm, '3/21/2017')   # regular x-axis
    split_data_plot(atm_encoded[atm], atm, '12/15/2019')  # magnified x-axis
    correlation_plots(atm_encoded, atm, 0, 'A')  # differential order = 0
    correlation_plots(atm_encoded, atm, 1, 'B')  # differential order = 1

# --------------------------------------------------------------------------------------------------------------------
# HEART
# --------------------------------------------------------------------------------------------------------------------

print('target ATM : {}'.format(target_atm))

# pre-processing (feature matrix + target vector)
y = atm_encoded[target_atm]
atm2 = collect_data(atm_encoded, mode='non_param_data_sequence')
atm3 = collect_data(atm_encoded, mode='non_param_regular_features')
algorithms1 = [('MA', 'MovingAverage'),
               ('SES', 'SingleExponentialSmoothing'),
               ('HES', 'HoltExponentialSmoothing'),
               ('ARIMA', 'AutoRegressionIntegratingMovingAverage'),
               ('SARIMA', 'SeasonalARIMA')]
algorithms2 = [('MLP_DS', 'MultiLayerPerceptronDataSequence'),
               ('SVM_DS', 'SupportVectorMachinesDataSequence'),
               ('RF_DS', 'RandomForestDataSequence'),
               ('KNN_DS', 'KNearestNeighborsDataSequence')]
algorithms3 = [('MLP', 'MultiLayerPerceptronRegularFeatures'),
               ('SVM', 'SupportVectorMachinesRegularFeatures'),
               ('RF', 'RandomForestRegularFeatures'),
               ('KNN', 'KNearestNeighborsRegularFeatures')]

# grid search: tuning hyper-parameters (one-time process => one model at a time)
algorithms4 = grid_search_param(grid_search['param'][0])
algorithms5 = grid_search_non_param(grid_search['non_param_ds'][0])
algorithms6 = grid_search_non_param(grid_search['non_param_rf'][0])

# results of the grid search: best algorithms parameter
best_algorithms_param = {
    'Approximate': {'Before': {'ARIMA': {'ATM 1': [(6, 1, 0)],
                                         'ATM 2': [(9, 1, 1)],
                                         'ATM 3': [(6, 1, 1)],
                                         'ATM (mean)': [(8, 1, 1)]},
                               'SARIMA': {'ATM 1': [(1, 1, 0), (0, 1, 1, 7)],
                                          'ATM 2': [(1, 1, 1), (1, 1, 1, 7)],
                                          'ATM 3': [(1, 1, 1), (1, 0, 0, 7)],
                                          'ATM (mean)': [(1, 1, 1), (1, 1, 1, 7)]},
                               'MLP_DS': {'ATM 1': [(10, 10), 'temp'],
                                          'ATM 2': [(8, 8, 8), 'temp'],
                                          'ATM 3': [(10,), 'temp'],
                                          'ATM (mean)': [(8, 8, 8), 'temp']},
                               'SVM_DS': {'ATM 1': [0.1, 10],
                                          'ATM 2': [1.0, 1000],
                                          'ATM 3': [0.0001, 1000],
                                          'ATM (mean)': [1.0, 10]},
                               'RF_DS': {'ATM 1': [100, 1.0],
                                         'ATM 2': [100, 1.0],
                                         'ATM 3': [100, 0.6],
                                         'ATM (mean)': [100, 0.7]},
                               'KNN_DS': {'ATM 1': [3, 'distance'],
                                          'ATM 2': [4, 'uniform'],
                                          'ATM 3': [7, 'uniform'],
                                          'ATM (mean)': [6, 'uniform']},
                               'MLP': {'ATM 1': [(6, 6), 'temp'],
                                       'ATM 2': [(10, 10, 10), 'temp'],
                                       'ATM 3': [(10, 10), 'temp'],
                                       'ATM (mean)': [(10, 10), 'temp']},
                               'SVM': {'ATM 1': [0.001, 100],
                                       'ATM 2': [0.1, 1],
                                       'ATM 3': [1.0, 5],
                                       'ATM (mean)': [0.001, 1000]},
                               'RF': {'ATM 1': [500, 0.6],
                                      'ATM 2': [500, 1.0],
                                      'ATM 3': [10, 0.6],
                                      'ATM (mean)': [500, 1.0]},
                               'KNN': {'ATM 1': [3, 'distance'],
                                       'ATM 2': [3, 'distance'],
                                       'ATM 3': [4, 'uniform'],
                                       'ATM (mean)': [7, 'distance']},
                               },
                    'During': {'ARIMA': {'ATM 1': [(9, 0, 0)],
                                         'ATM 2': [(9, 1, 0)],
                                         'ATM 3': [(9, 1, 1)],
                                         'ATM (mean)': [(9, 1, 1)]},
                               'SARIMA': {'ATM 1': [(1, 0, 0), (0, 0, 1, 7)],
                                          'ATM 2': [(1, 1, 0), (0, 1, 1, 7)],
                                          'ATM 3': [(1, 1, 1), (1, 0, 1, 7)],
                                          'ATM (mean)': [(1, 1, 1), (1, 0, 0, 7)]},
                               'MLP_DS': {'ATM 1': [(4, 4), 'temp'],
                                          'ATM 2': [8, 8, 8],
                                          'ATM 3': [10, 10, 10],
                                          'ATM (mean)': [6, 6, 6]},
                               'SVM_DS': {'ATM 1': [1.0, 100],
                                          'ATM 2': [0.001, 5],
                                          'ATM 3': [0.0001, 1000],
                                          'ATM (mean)': [1.0, 100]},
                               'RF_DS': {'ATM 1': [200, 0.8],
                                         'ATM 2': [10, 0.9],
                                         'ATM 3': [50, 0.8],
                                         'ATM (mean)': [100, 0.7]},
                               'KNN_DS': {'ATM 1': [7, 'distance'],
                                          'ATM 2': [7, 'uniform'],
                                          'ATM 3': [4, 'uniform'],
                                          'ATM (mean)': [4, 'uniform']},
                               'MLP': {'ATM 1': [(8, 8), 'temp'],
                                       'ATM 2': [(8, 8), 'temp'],
                                       'ATM 3': [(6, 6), 'temp'],
                                       'ATM (mean)': [(10, 10, 10), 'temp']},
                               'SVM': {'ATM 1': [0.0001, 5],
                                       'ATM 2': [0.001, 1000],
                                       'ATM 3': [0.01, 100],
                                       'ATM (mean)': [0.1, 5]},
                               'RF': {'ATM 1': [500, 0.6],
                                      'ATM 2': [200, 0.6],
                                      'ATM 3': [10, 0.7],
                                      'ATM (mean)': [200, 0.9]},
                               'KNN': {'ATM 1': [5, 'distance'],
                                       'ATM 2': [3, 'uniform'],
                                       'ATM 3': [4, 'distance'],
                                       'ATM (mean)': [5, 'distance']}}},
    'Updated': {'Before': {'ARIMA': {'ATM 1': [(9, 0, 0)],
                                     'ATM 2': [(7, 0, 0)],
                                     'ATM 3': [(7, 1, 1)],
                                     'ATM (mean)': [(1, 1, 1)]},
                           'SARIMA': {'ATM 1': [(1, 0, 0), (0, 1, 1, 7)],
                                      'ATM 2': [(1, 0, 0), (1, 1, 0, 7)],
                                      'ATM 3': [(1, 1, 1), (0, 1, 1, 7)],
                                      'ATM (mean)': [(1, 1, 1), (1, 1, 1, 7)]},
                           'MLP_DS': {'ATM 1': [(4, 4), 'temp'],
                                      'ATM 2': [(10,), 'temp'],
                                      'ATM 3': [(10, 10), 'temp'],
                                      'ATM (mean)': [(2,), 'temp']},
                           'SVM_DS': {'ATM 1': [1.0, 1000],
                                      'ATM 2': [0.1, 10],
                                      'ATM 3': [0.1, 10],
                                      'ATM (mean)': [1.0, 1000]},
                           'RF_DS': {'ATM 1': [10, 0.7],
                                     'ATM 2': [10, 0.8],
                                     'ATM 3': [10, 0.6],
                                     'ATM (mean)': [10, 1.0]},
                           'KNN_DS': {'ATM 1': [6, 'uniform'],
                                      'ATM 2': [5, 'distance'],
                                      'ATM 3': [4, 'distance'],
                                      'ATM (mean)': [4, 'distance']},
                           'MLP': {'ATM 1': [(10,), 'temp'],
                                   'ATM 2': [(10, 10, 10), 'temp'],
                                   'ATM 3': [(10, 10, 10), 'temp'],
                                   'ATM (mean)': [(6, 6), 'temp']},
                           'SVM': {'ATM 1': [0.001, 100],
                                   'ATM 2': [0.1, 1],
                                   'ATM 3': [1.0, 5],
                                   'ATM (mean)': [0.001, 1000]},
                           'RF': {'ATM 1': [500, 0.6],
                                  'ATM 2': [500, 1.0],
                                  'ATM 3': [10, 0.6],
                                  'ATM (mean)': [500, 1.0]},
                           'KNN': {'ATM 1': [3, 'distance'],
                                   'ATM 2': [3, 'distance'],
                                   'ATM 3': [4, 'uniform'],
                                   'ATM (mean)': [7, 'distance']},
                           },
                'During': {'ARIMA': {'ATM 1': [(8, 1, 1)],
                                     'ATM 2': [(9, 0, 1)],
                                     'ATM 3': [(9, 1, 1)],
                                     'ATM (mean)': [(9, 1, 1)]},
                           'SARIMA': {'ATM 1': [(1, 1, 1), (0, 1, 0, 7)],
                                      'ATM 2': [(1, 0, 1), (0, 1, 1, 7)],
                                      'ATM 3': [(1, 1, 1), (0, 1, 0, 7)],
                                      'ATM (mean)': [(1, 1, 1), (1, 1, 0, 7)]},
                           'MLP_DS': {'ATM 1': [(10, 10, 10), 'temp'],
                                      'ATM 2': [(8, 8), 'temp'],
                                      'ATM 3': [(10, 10), 'temp'],
                                      'ATM (mean)': [(8, 8), 'temp']},
                           'SVM_DS': {'ATM 1': [0.001, 1000],
                                      'ATM 2': [1.0, 1],
                                      'ATM 3': [0.01, 1000],
                                      'ATM (mean)': [0.001, 1000]},
                           'RF_DS': {'ATM 1': [100, 0.7],
                                     'ATM 2': [10, 0.6],
                                     'ATM 3': [200, 1.0],
                                     'ATM (mean)': [10, 0.7]},
                           'KNN_DS': {'ATM 1': [6, 'uniform'],
                                      'ATM 2': [6, 'distance'],
                                      'ATM 3': [3, 'uniform'],
                                      'ATM (mean)': [4, 'uniform']},
                           'MLP': {'ATM 1': [(8, 8), 'temp'],
                                   'ATM 2': [(8, 8), 'temp'],
                                   'ATM 3': [(6, 6), 'temp'],
                                   'ATM (mean)': [(10, 10, 10), 'temp']},
                           'SVM': {'ATM 1': [0.0001, 5],
                                   'ATM 2': [0.001, 1000],
                                   'ATM 3': [0.01, 100],
                                   'ATM (mean)': [0.1, 5]},
                           'RF': {'ATM 1': [500, 0.6],
                                  'ATM 2': [200, 0.6],
                                  'ATM 3': [10, 0.7],
                                  'ATM (mean)': [200, 0.9]},
                           'KNN': {'ATM 1': [5, 'distance'],
                                   'ATM 2': [3, 'uniform'],
                                   'ATM 3': [4, 'distance'],
                                   'ATM (mean)': [5, 'distance']}
                           }
                }
}

# features importance
for atm in ATMs:
    for iteration in iteration_method:
        for pandemic in pandemic_status:
            X_train, y_train, X_test, y_test = split_data(atm3, pandemic)
            hp1 = best_algorithms_param[iteration][pandemic]['RF'][atm][0]
            hp2 = best_algorithms_param[iteration][pandemic]['RF'][atm][1]
            model = tuned_model('RF', hp1, hp2)
            model.fit(X_train, y_train)
            features_importance_plot(atm3, model, atm, iteration, pandemic)

# models performance evaluation
eval_scores = []
best_models = []
prediction_before = pd.DataFrame()
prediction_during = pd.DataFrame()
for eval_metric in eval_metrics:
    pand_scores = []
    for pandemic in pandemic_status:
        print('{} COVID-19 => Metric: {}'.format(pandemic, eval_metric))
        algm_scores = []
        # ---------------------------------
        # parametric methods
        y_train, y_test = split_y(y, pandemic)
        if grid_search['status']:
            algorithms = algorithms4
        else:
            algorithms = algorithms1
        for algorithm in algorithms:
            if grid_search['status'] and not grid_search['param'][1]:
                break
            print(algorithm[1])
            iter_scores = []
            for iteration in iteration_method:
                repl_scores = []
                y_predict = []
                replicas = 1
                for replica in range(replicas):
                    y_hist = [i for i in y_train]
                    y_pred = []
                    for sample in range(len(y_test)):
                        if algorithm[0] == 'MA':
                            y_hat = moving_average(y_hist, window=7)  # window=len(y_hist)
                            y_pred.append(y_hat)
                        else:
                            if algorithm[0] == 'SES':
                                model = SimpleExpSmoothing(y_hist, )
                                model_fit = model.fit()
                            elif algorithm[0] == 'HES':
                                model = Holt(y_hist)
                                model_fit = model.fit()
                            elif algorithm[0] == 'ARIMA':
                                if grid_search['status']:
                                    _order = algorithm[2]
                                else:
                                    _order = best_algorithms_param[iteration][pandemic][algorithm[0]][target_atm][0]
                                model = ARIMA(y_hist, order=_order)
                                model_fit = model.fit(disp=0)
                            else:
                                _order = best_algorithms_param[iteration][pandemic][algorithm[0]][target_atm][0]
                                if grid_search['status']:
                                    _seasonal = algorithm[2]
                                else:
                                    _seasonal = \
                                        best_algorithms_param[iteration][pandemic][algorithm[0]][target_atm][1]
                                model = SARIMAX(y_hist, order=_order, seasonal_order=_seasonal,
                                                enforce_stationarity=False)
                                model_fit = model.fit(disp=0)
                            y_hat = model_fit.forecast()
                            y_pred.append(y_hat[0])
                        if iteration == iteration_method[0]:
                            y_hist.append(y_pred[sample])
                        else:
                            y_hist.append(y_test[sample])
                    score = evaluation(y_test, y_pred, scoring=eval_metric)
                    repl_scores.append(score[1])
                    if replica == 0:
                        y_predict = y_pred
                    else:
                        y_predict = [x + y for x, y in zip(y_predict, y_pred)]
                iter_scores.append((iteration, np.mean(repl_scores), np.std(repl_scores)))
                temp = [x / replicas for x in y_predict]
                if pandemic == pandemic_status[0]:
                    prediction_before['{} {} {}'.format(iteration, algorithm[0], eval_metric)] = temp
                else:
                    prediction_during['{} {} {}'.format(iteration, algorithm[0], eval_metric)] = temp
            if grid_search['status']:
                algm_scores.append((algorithm[1], iter_scores[0], iter_scores[1]))
            else:
                algm_scores.append((algorithm[0], iter_scores[0], iter_scores[1]))
        print('Time series parametric methods => DONE!')
        # ---------------------------------
        # non-parametric methods, data sequence algorithm
        X_train, y_train, X_test, y_test = split_data(atm2, pandemic)
        if grid_search['status']:
            algorithms = algorithms5
        else:
            algorithms = algorithms2
        for algorithm in algorithms:
            if grid_search['status'] and not grid_search['non_param_ds'][1]:
                break
            print(algorithm[0])
            iter_scores = []
            for iteration in iteration_method:
                repl_scores = []
                y_predict = []
                if algorithm[0] == 'SVM_DS' or algorithm[0] == 'KNN_DS':
                    replicas = 1
                for replica in range(replicas):
                    X_hist, y_hist = X_train, [i for i in y_train]
                    y_pred = []
                    for sample in range(len(y_test)):
                        last_week = []
                        for i in range(7):
                            last_week.append(y_hist[len(y_hist) - 7 + i])
                        X_temp = pd.DataFrame(last_week).transpose()
                        X_temp.columns = X_test.columns
                        if grid_search['status']:
                            model = algorithm[1]
                        else:
                            hp1 = best_algorithms_param[iteration][pandemic][algorithm[0]][target_atm][0]
                            hp2 = best_algorithms_param[iteration][pandemic][algorithm[0]][target_atm][1]
                            model = tuned_model(algorithm[0], hp1, hp2)
                        model.fit(X_train, y_train)
                        y_hat = model.predict(X_temp)
                        y_pred.append(y_hat[0])
                        X_hist = pd.concat([X_hist, X_temp], ignore_index=True)
                        if iteration == iteration_method[0]:
                            y_hist.append(y_pred[sample])
                        else:
                            y_hist.append(y_test[sample])
                    score = evaluation(y_test, y_pred, scoring=eval_metric)
                    repl_scores.append(score[1])
                    if replica == 0:
                        y_predict = y_pred
                    else:
                        y_predict = [x + y for x, y in zip(y_predict, y_pred)]
                iter_scores.append((iteration, np.mean(repl_scores), np.std(repl_scores)))
                temp = [x / replicas for x in y_predict]
                if pandemic == pandemic_status[0]:
                    prediction_before['{} {} {}'.format(iteration, algorithm[0], eval_metric)] = temp
                else:
                    prediction_during['{} {} {}'.format(iteration, algorithm[0], eval_metric)] = temp
            algm_scores.append((algorithm[0], iter_scores[0], iter_scores[1]))
        print('non-parametric methods => data sequence algorithm => DONE!')
        # ---------------------------------
        # non-parametric methods, regular features algorithm
        X_train, y_train, X_test, y_test = split_data(atm3, pandemic)
        if grid_search['status']:
            algorithms = algorithms6
        else:
            algorithms = algorithms3
        for algorithm in algorithms:
            if grid_search['status'] and not grid_search['non_param_rf'][1]:
                break
            print(algorithm[0])
            iter_scores = []
            for iteration in iteration_method:
                repl_scores = []
                y_predict = []
                if algorithm[0] == 'SVM' or algorithm[0] == 'KNN':
                    replicas = 1
                for replica in range(replicas):
                    X_hist, y_hist = X_train, [i for i in y_train]
                    y_pred = []
                    for sample in range(len(y_test)):
                        X_temp = pd.DataFrame(X_test.iloc[sample, :]).transpose()
                        if grid_search['status']:
                            model = algorithm[1]
                        else:
                            hp1 = best_algorithms_param[iteration][pandemic][algorithm[0]][target_atm][0]
                            hp2 = best_algorithms_param[iteration][pandemic][algorithm[0]][target_atm][1]
                            model = tuned_model(algorithm[0], hp1, hp2)
                        model.fit(X_train, y_train)

                        y_hat = model.predict(X_temp)
                        y_pred.append(y_hat[0])
                        X_hist = pd.concat([X_hist, X_temp], ignore_index=True)
                        if iteration == iteration_method[0]:
                            y_hist.append(y_pred[sample])
                        else:
                            y_hist.append(y_test[sample])
                    score = evaluation(y_test, y_pred, scoring=eval_metric)
                    repl_scores.append(score[1])
                    if replica == 0:
                        y_predict = y_pred
                    else:
                        y_predict = [x + y for x, y in zip(y_predict, y_pred)]
                iter_scores.append((iteration, np.mean(repl_scores), np.std(repl_scores)))
                temp = [x / replicas for x in y_predict]
                if pandemic == pandemic_status[0]:
                    prediction_before['{} {} {}'.format(iteration, algorithm[0], eval_metric)] = temp
                else:
                    prediction_during['{} {} {}'.format(iteration, algorithm[0], eval_metric)] = temp
            algm_scores.append((algorithm[0], iter_scores[0], iter_scores[1]))
        print('non-parametric methods - regular features algorithm => DONE!')
        # ---------------------------------
        pand_scores.append((pandemic, algm_scores))
    eval_scores.append((eval_metric, pand_scores))

# ------------------------------------------------------------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------------------------------------------------------------

# comparing models
models_summary = pd.DataFrame()
for iteration in iteration_method:
    for pandemic in pandemic_status:
        mse_average, pocid_average, fitness_average = scores_plot(iteration, pandemic)
        models_summary = summary_of_models(models_summary, iteration, pandemic)
models_summary.index = [i[0] for i in eval_scores[0][1][0][1]]
excel_output(models_summary, root='regression/comparison', file_name=target_atm)

# prediction for unseen (testing) data
if not grid_search['status']:
    prediction_before.index = y[time_span['Before'][1]:time_span['Before'][2]].index
    prediction_during.index = y[time_span['During'][1]:time_span['During'][2]].index
    algorithms_name = [i[0] for i in algorithms1] + [i[0] for i in algorithms2] + [i[0] for i in algorithms3]
    for iteration in iteration_method:
        for algorithm in algorithms_name:
            prediction_plot(prediction_before, 'Before', iteration, algorithm)
            prediction_plot(prediction_during, 'During', iteration, algorithm)
        # -----------------------
        id1_before = models_summary.loc['MA': 'SARIMA', '{} {} fitness_av'.format(iteration, 'Before')].idxmax()
        id2_before = models_summary.loc['MLP_DS':'KNN_DS', '{} {} fitness_av'.format(iteration, 'Before')].idxmax()
        id3_before = models_summary.loc['MLP':'KNN', '{} {} fitness_av'.format(iteration, 'Before')].idxmax()
        top3_algorithms = [id1_before, id2_before, id3_before]
        prediction_top3_plot(prediction_before, 'Before', iteration, top3_algorithms)
        # -----------------------
        id1_before = models_summary.loc['MA':'SARIMA', '{} {} fitness_av'.format(iteration, 'During')].idxmax()
        id2_before = models_summary.loc['MLP_DS':'KNN_DS', '{} {} fitness_av'.format(iteration, 'During')].idxmax()
        id3_before = models_summary.loc['MLP':'KNN', '{} {} fitness_av'.format(iteration, 'During')].idxmax()
        top3_algorithms = [id1_before, id2_before, id3_before]
        prediction_top3_plot(prediction_during, 'During', iteration, top3_algorithms)

# ------------------------------------------------------------------------------------------------------------------
# END
# ------------------------------------------------------------------------------------------------------------------
print('Done!')
