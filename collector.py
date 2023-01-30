# Copyright 2023 Egor Sedov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# import signal_generation as sg
# import channel_model as ch
import hpcom
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
# from sklearn.externals import joblib
from datetime import datetime
import xgboost as xgb


from composer import *
from estimator import *


def train_xgboost(paths_to_data, p_ave_dbm, z_km, train_runs, test_runs, p_list, data_type='wnwon_1'):

    # last element of paths_to_data (paths_to_data[-1]) will be used for validation on different power levels
    # so be careful about order!

    job_name = 'p' + str(p_ave_dbm) + '_z' + str(z_km) + data_type
    
    df_list = []
    for path in paths_to_data:
        df_list.append(pd.read_pickle(path))

    n_neighbours = 5
    n_gauss = 0
    n_channels = 1

    df = pd.concat(df_list, ignore_index=True, sort=False)
    df_tree = get_data_with_m_neighbor(df, z_km, p_ave_dbm, n_channels, train_runs, n_neighbours, r=0.1,
                                       n_gauss=n_gauss)
    # df_tree = get_data_with_m_neighbor(df_noise, z_km, p_ave_dbm, n_channels, train_runs, n_neighbours,
    #                                    r=0.1, n_gauss=n_gauss, scale_type='constellation')
    df_tree_for_test = get_data_with_m_neighbor(df_list[-1], z_km, p_ave_dbm, n_channels, test_runs, n_neighbours,
                                                r=0.1, n_gauss=n_gauss, scale_type='constellation')

    # df_tree = get_data_gauss_parallel(df, z_km, p_ave_dbm, n_channels, train_runs, n_neighbours, r=0.1, n_gauss=n_gauss, sigma=sigma)
    # df_tree_for_test = get_data_gauss_parallel(df, z_km, p_ave_dbm, n_channels, test_runs, n_neighbours, r=0.1, n_gauss=1, sigma=sigma)

    df = pd.DataFrame([])
    df_noise = pd.DataFrame([])
    del df, df_noise

    print("P_ave [dBm] = ", p_ave_dbm)
    p_ave = (10 ** (p_ave_dbm / 10)) / 1000
    mod_type = hpcom.modulation.get_modulation_type_from_order(16)
    scale_constellation = hpcom.modulation.get_scale_coef_constellation(mod_type) / np.sqrt(p_ave / 2)
    constellation = hpcom.modulation.get_constellation('16qam')

    # drop labels
    labels_to_drop = ['point_orig_abs', 'point_orig_angle',
                      'point_orig_real', 'point_orig_imag',
                      'point_label',
                      'diff_real', 'diff_imag',
                      'diff_sq_real', 'diff_sq_imag',
                      'diff_circle_real', 'diff_circle_imag',
                      'diff_p3_real', 'diff_p3_imag']

    # for get_data_with_m_neighbor
    labels_to_drop = labels_to_drop + [f'diff_gauss_{m}_real' for m in range(n_gauss)] + [f'diff_gauss_{m}_imag' for m
                                                                                          in range(n_gauss)]

    # for get_data_gauss_parallel
    # labels_to_drop = labels_to_drop + [f'diff_gauss_real'] + [f'diff_gauss_imag']

    # chose what to predict
    # label_predict = ['diff_circle_real', 'diff_circle_imag']
    label_predict = ['diff_real', 'diff_imag']
    m = 5
    # label_predict = [f'diff_gauss_{m}_real', f'diff_gauss_{m}_imag']
    # label_predict = [f'diff_gauss_real', f'diff_gauss_imag']
    # label_predict = ['diff_p3_real', 'diff_p3_imag']
    # label_predict = ['point_label']

    # form validation dataframe
    X_for_test = df_tree_for_test.drop(labels=labels_to_drop, axis=1)
    points_init_for_test = df_tree_for_test['point_orig_abs'].values * np.exp(
        1.0j * df_tree_for_test['point_orig_angle'])
    labels_for_test = df_tree_for_test['point_label']

    # form train dataframe
    y = df_tree[[label_predict[0], label_predict[1]]].values  # predict 2 values simultaneously
    X = df_tree.drop(labels=labels_to_drop, axis=1)
    split_ratio = 0.01
    random_state = 42

    # additionally split for small test dataframe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_state)
    po_abs_train, po_abs_test, po_angle_train, po_angle_test = train_test_split(df_tree['point_orig_abs'].values,
                                                                                df_tree['point_orig_angle'].values,
                                                                                test_size=split_ratio,
                                                                                random_state=random_state)

    print('Colecting data for different power levels')

    # for test data for different average powers
    X_test_list = []
    points_init_list = []
    # p_list = [1, 2, 3, 4, 5, 6, 7, 8]  # list of available average powers
    for p in p_list:
        X_test_cur, points_init_test_cur = get_data_for_test(paths_to_data[-1],
                                                             p_ave_dbm=p, z_km=z_km,
                                                             test_runs=test_runs,
                                                             n_channels=n_channels,
                                                             n_neighbours=n_neighbours,
                                                             n_gauss=0)

        X_test_list.append(X_test_cur)
        points_init_list.append(points_init_test_cur)

    # XGBoost

    random_state_for_tree = 0
    learning_rate = 0.2
    reg_lambda = 0.0  # 1.0 - L2 regularization term on weights. Increasing this value will make model more conservative.
    reg_alpha = 1.0  # 0.0 - L1 regularization term on weights. Increasing this value will make model more conservative.
    n_estimators = 1000
    max_depth = 200

    start_time = datetime.now()
    # Use "gpu_hist" for training the model.
    xgb_reg = xgb.XGBRegressor(tree_method="gpu_hist",
                               n_estimators=n_estimators,
                               learning_rate=learning_rate,
                               reg_lambda=reg_lambda,
                               reg_alpha=reg_alpha,
                               seed=random_state_for_tree)
    # Fit the model using predictor X and response y.
    xgb_reg.fit(X_train, y_train)

    calculation_time = (datetime.now() - start_time).total_seconds()
    print("model took: \n",
          calculation_time * 1000, "ms \n",
          calculation_time, "s \n",
          calculation_time / 60., "min \n")

    # Save model into JSON format.
    xgb_reg.save_model("data/" + job_name + "_xgb_regressor.json")

    columns = ['z_km', 'p_ave_dbm_train', 'p_ave_dbm_test',
               'ber_orig', 'ber_pred',
               'q_db_orig', 'q_db_pred',
               'evm_orig', 'evm_pred']
    df_result_for_model = pd.DataFrame(columns=columns)

    # print('Dataframe')
    # print(df_result_for_model)

    for k in range(len(p_list)):
        # return and collect values for different power levels
        result_for_current_power = predict_and_eval((xgb_reg,), X_test_list[k], points_init_list[k],
                                                    p_ave_dbm=p_list[k], scale_type='constellation',
                                                    name='RF ' + str(p_list[k]) + ' dbm test')

        data = [z_km, p_ave_dbm, p_list[k],
                result_for_current_power['ber_orig'], result_for_current_power['ber_predict'],
                result_for_current_power['q_orig_db'], result_for_current_power['q_pred_db'],
                result_for_current_power['evm_orig'], result_for_current_power['evm_pred']]

        # print('Data shape:', np.shape(data))

        df_result_for_model.loc[len(df_result_for_model)] = data

    return df_result_for_model
    
    