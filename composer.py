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


def get_grid(min, max, n):
    x = np.linspace(min, max, n)
    y = np.linspace(min, max, n)
    # full coordinate arrays
    return np.meshgrid(x, y)


def transform(points, scale):
    amplitudes = np.absolute(points)
    angles = np.angle(points)

    # amplitudes = amplitudes * (0.2 * amplitudes + np.absolute(angles) * 0.05)
    amplitudes = amplitudes * (scale * amplitudes)

    return amplitudes * np.exp(1.0j * angles)


def inverse_transform(points, scale):
    amplitudes = np.absolute(points)
    angles = np.angle(points)

    # amplitudes = amplitudes * (0.2 * amplitudes + np.absolute(angles) * 0.05)
    amplitudes = np.sqrt(amplitudes / scale)

    return amplitudes * np.exp(1.0j * angles)


# def get_m_neighbours(points, k, m):
#
#     point_minus_m = points[k - m]
#     point_plus_m = points[(k + m) % len(points)]
#
#     return point_minus_m, point_plus_m

def get_m_neighbours(points, m):
    n = len(points)
    points = np.concatenate((points[-m:], points, points[:m]))
    result = np.array([points[k - m:k + m + 1] for k in range(m, n + m)], dtype=complex)
    return result


def get_shift_to_square(point, point_to_shift, r):
    x_center, y_center = point.real, point.imag
    x_point, y_point = point_to_shift.real, point_to_shift.imag

    x_left, x_right = x_center - r, x_center + r
    y_left, y_right = y_center - r, y_center + r

    if x_point < x_left:
        x_shift = x_left - x_point
    elif x_point > x_right:
        x_shift = x_right - x_point
    else:
        x_shift = 0

    if y_point < y_left:
        y_shift = y_left - y_point
    elif y_point > y_right:
        y_shift = y_right - y_point
    else:
        y_shift = 0

    return complex(x_shift, y_shift)


def get_shift_to_circle(points, points_to_shift, r):
    ind = np.where(np.absolute(points_to_shift - points) > r)
    shifts = np.zeros(len(points), dtype=complex)
    points_on_circle = r * np.exp(1.0j * np.angle(points_to_shift[ind] - points[ind])) + points[ind]
    shifts[ind] = points_on_circle - points_to_shift[ind]
    return shifts


def get_data_with_m_neighbor_old(df, z_km, p_ave_dbm, n_channels, run_list, m_neighbours):
    # Create dataframe (table) with following columns:
    # neighbor symbols from -M to M, current symbol
    # optional: z_km, p_ave_dbm, n_channels
    # predict absolute value and angle for correction shift (in 2 dimensional plane -> real and imag part of symbol)

    p_ave = (10 ** (p_ave_dbm / 10)) / 1000
    mod_type = sg.get_modulation_type_from_order(16)
    scale_constellation = sg.get_scale_coef_constellation(mod_type) / np.sqrt(p_ave / 2)

    # columns = ['point_abs', 'point_angle', 'diff_abs', 'diff_angle']
    columns = ['point_orig_abs', 'point_orig_angle',
               'point_abs', 'point_angle',
               'diff_real', 'diff_imag',
               'diff_sq_real', 'diff_sq_imag',
               'diff_circle_real', 'diff_circle_imag']

    for m in range(1, m_neighbours + 1):
        columns = columns + [f'minus_m_{m}_abs', f'plus_m_{m}_abs', f'minus_m_{m}_angle', f'plus_m_{m}_angle']

    df_result = pd.DataFrame(columns=columns)
    # print(df_result)

    for run in run_list:
        df_current = df[(df['z_km'] == z_km) & (df['p_ave_dbm'] == p_ave_dbm) & (df['n_channels'] == n_channels) & (
                df['run'] == run)]

        points_x_orig = df_current['points_x_orig'].iloc[0][0]
        # points_x_orig = transform(points_x_orig, scale=400)
        points_x_shifted = df_current['points_x_shifted'].iloc[0]
        # points_x_shifted = transform(points_x_shifted, scale=400)
        points_diff = points_x_orig - points_x_shifted

        n_points = len(points_x_orig)
        points_diff_sq = np.zeros(n_points, dtype=complex)
        for k in range(n_points):
            points_diff_sq[k] = get_shift_to_square(points_x_orig[k], points_x_shifted[k], 0.5 / scale_constellation)

        points_diff_circle = get_shift_to_circle(points_x_orig, points_x_shifted, 0.5 / scale_constellation)

        for k in tqdm(range(len(points_x_shifted))):
            # for k in tqdm(range(100)):

            # row = np.array([np.absolute(points_x_shifted[k]), np.angle(points_x_shifted[k]),
            #                 np.absolute(points_diff[k]), np.angle(points_diff[k])])
            row = np.array([np.absolute(points_x_orig[k]), np.angle(points_x_orig[k]),
                            np.absolute(points_x_shifted[k]), np.angle(points_x_shifted[k]),
                            np.real(points_diff[k]), np.imag(points_diff[k]),
                            np.real(points_diff_sq[k]), np.imag(points_diff_sq[k]),
                            np.real(points_diff_circle[k]), np.imag(points_diff_circle[k])])
            for m in range(1, m_neighbours + 1):
                m_neighbours_points = get_m_neighbours(points_x_shifted, k, m)
                row = np.concatenate((row, np.absolute(m_neighbours_points), np.angle(m_neighbours_points)))

            # print(np.shape(row), row)
            # print(len(columns))

            df_result = pd.concat([df_result, pd.DataFrame([row], columns=columns)], ignore_index=True)
            # df_result = pd.DataFrame([row])

            # print(df_result)

    return df_result


def form_columns_names_base(m_neighbours, general_type='x', value_type='real'):
    columns = []

    for m in range(m_neighbours, 0, -1):
        columns = columns + [f'minus_m_{m}_' + general_type + '_' + value_type]
    columns = columns + ['point_' + general_type + '_' + value_type]
    for m in range(1, m_neighbours + 1):
        columns = columns + [f'plus_m_{m}_' + general_type + '_' + value_type]

    return columns


def form_columns_names(m_neighbours, general_type='x'):
    columns = []

    columns = columns + form_columns_names_base(m_neighbours, general_type, value_type='abs')
    columns = columns + form_columns_names_base(m_neighbours, general_type, value_type='angle')
    columns = columns + form_columns_names_base(m_neighbours, general_type, value_type='real')
    columns = columns + form_columns_names_base(m_neighbours, general_type, value_type='imag')

    return columns


def get_label_dict_for_constellation(constellation):
    dict = {}
    k = 0
    for point in constellation:
        dict[str(point)] = k
        k += 1

    dict_inv = {v: k for k, v in dict.items()}

    return dict, dict_inv


def get_labels_for_points(points, constellation):
    labels = np.zeros(len(points))
    dict, _ = get_label_dict_for_constellation(constellation)
    k = 0
    for p in points:
        labels[k] = dict[str(p)]
        k += 1

    return labels


def get_data_wo_neighbor(df, z_km, p_ave_dbm, n_channels, run_list):
    # Create dataframe (table) with following columns:
    # neighbor symbols from -M to M, current symbol
    # optional: z_km, p_ave_dbm, n_channels
    # predict absolute value and angle for correction shift (in 2 dimensional plane -> real and imag part of symbol)

    # seed = datetime.now()
    # np.random.seed(seed)

    p_ave = (10 ** (p_ave_dbm / 10)) / 1000
    mod_type = hpcom.modulation.get_modulation_type_from_order(16)
    scale_constellation = hpcom.modulation.get_scale_coef_constellation(mod_type) / np.sqrt(p_ave / 2)
    constellation = hpcom.modulation.get_constellation(mod_type)

    columns = ['point_orig_real', 'point_orig_imag',
               'point_received_real', 'point_received_imag',
               'point_label',
               'diff_real', 'diff_imag']

    print(columns)

    df_result = pd.DataFrame(columns=columns)
    # print(df_result)

    for run in run_list:
        print('run number', run)
        df_current = df[(df['z_km'] == z_km) & (df['p_ave_dbm'] == p_ave_dbm) & (df['n_channels'] == n_channels) & (
                df['run'] == run)]

        points_x_orig = df_current['points_x_orig'].iloc[0][0]
        points_x_scaled = hpcom.modulation.get_nearest_constellation_points_new(points_x_orig * scale_constellation,
                                                                                constellation)
        # points_x_labels_str = list(map(str, points_x_scaled))
        points_x_labels = get_labels_for_points(points_x_scaled, constellation)

        points_x_shifted = df_current['points_x_shifted'].iloc[0]
        points_diff = (points_x_orig - points_x_shifted) * scale_constellation
        points_x_shifted_scaled = points_x_shifted * scale_constellation

        # points_y_orig = df_current['points_y_orig'].iloc[0][0]
        # points_y_shifted = df_current['points_y_shifted'].iloc[0]
        # points_y_scaled = sg.get_nearest_constellation_points_new(points_y_orig * scale_constellation, constellation)

        data = np.column_stack((np.real(points_x_scaled), np.imag(points_x_scaled),
                                np.real(points_x_shifted_scaled), np.imag(points_x_shifted_scaled),
                                points_x_labels,
                                np.real(points_diff), np.imag(points_diff)))

        # print(np.shape(data))

        df_result = pd.concat([df_result, pd.DataFrame(data, columns=columns)], ignore_index=True)

        # k = 1
        # one_row = [np.absolute(points_x_orig[k]), np.angle(points_x_orig[k]),
        #                         np.real(points_diff[k]), np.imag(points_diff[k]),
        #                         np.real(points_diff_sq[k]), np.imag(points_diff_sq[k]),
        #                         np.real(points_diff_circle[k]), np.imag(points_diff_circle[k])]
        # print(one_row, np.absolute(points_with_neighbours[k]), np.angle(points_with_neighbours[k]))
        # print(np.array(df_result.iloc[k]))
        # if np.isclose(np.array(df_result.iloc[0]), one_row):
        #     print('Error')

    return df_result


def get_data_with_m_neighbor(df, z_km, p_ave_dbm, n_channels, run_list, m_neighbours,
                             r=0.1, sigma=1.0, n_gauss=10,
                             scale_type='constellation'):
    # Create dataframe (table) with following columns:
    # neighbor symbols from -M to M, current symbol
    # optional: z_km, p_ave_dbm, n_channels
    # predict absolute value and angle for correction shift (in 2 dimensional plane -> real and imag part of symbol)

    # seed = datetime.now()
    # np.random.seed(seed)

    p_ave = (10 ** (p_ave_dbm / 10)) / 1000
    mod_type = hpcom.modulation.get_modulation_type_from_order(16)
    scale_constellation = hpcom.modulation.get_scale_coef_constellation(mod_type) / np.sqrt(p_ave / 2)
    constellation = hpcom.modulation.get_constellation(mod_type)

    scale = 1
    if scale_type == 'constellation':
        print('Scale data to correspond to initial constellation')
        scale = scale_constellation
    else:
        print('No such type of scale_type. Set scale to 1')

    columns = ['point_orig_abs', 'point_orig_angle',
               'point_orig_real', 'point_orig_imag',
               'point_label',
               'diff_real', 'diff_imag',
               'diff_sq_real', 'diff_sq_imag',
               'diff_circle_real', 'diff_circle_imag',
               'diff_p3_real', 'diff_p3_imag']

    columns = columns + [f'diff_gauss_{m}_real' for m in range(n_gauss)]
    columns = columns + [f'diff_gauss_{m}_imag' for m in range(n_gauss)]

    columns = columns + form_columns_names(m_neighbours, general_type='x')
    columns = columns + form_columns_names(m_neighbours, general_type='y')

    print(columns)

    df_result = pd.DataFrame(columns=columns)
    # print(df_result)

    for run in run_list:
        print('run number', run)
        df_current = df[(df['z_km'] == z_km) & (df['p_ave_dbm'] == p_ave_dbm) & (df['n_channels'] == n_channels) & (
                df['run'] == run)]

        points_x_orig = df_current['points_x_orig'].iloc[0][0] * scale
        points_x_scaled = hpcom.modulation.get_nearest_constellation_points_new(
            points_x_orig * scale_constellation / scale,
            constellation)
        # points_x_labels_str = list(map(str, points_x_scaled))
        points_x_labels = get_labels_for_points(points_x_scaled, constellation)

        points_x_shifted = df_current['points_x_shifted'].iloc[0] * scale
        points_x_with_neighbours = get_m_neighbours(points_x_shifted, m_neighbours)
        points_diff = points_x_orig - points_x_shifted

        print(np.shape(points_x_with_neighbours))

        # points_y_orig = df_current['points_y_orig'].iloc[0][0]
        points_y_shifted = df_current['points_y_shifted'].iloc[0] * scale
        points_y_with_neighbours = get_m_neighbours(points_y_shifted, m_neighbours)
        # points_y_scaled = sg.get_nearest_constellation_points_new(points_y_orig * scale_constellation, constellation)

        n_points = len(points_x_orig)
        mu, sigma = 0, sigma / (3.0 * scale_constellation / scale)  # TODO: check if it correct
        # points_diff_gauss = points_diff + (np.random.normal(mu, sigma, n_points) + 1.0j * np.random.normal(mu, sigma, n_points))
        points_diff_gauss = np.tile(points_diff.reshape((n_points, 1)), (1, n_gauss)) + (
                np.random.normal(mu, sigma, (n_points, n_gauss)) +
                1.0j * np.random.normal(mu, sigma, (n_points, n_gauss)))

        points_diff_sq = np.zeros(n_points, dtype=complex)
        for k in range(n_points):
            points_diff_sq[k] = get_shift_to_square(points_x_orig[k], points_x_shifted[k],
                                                    r / scale_constellation * scale)

        points_diff_circle = get_shift_to_circle(points_x_orig, points_x_shifted, r / scale_constellation * scale)
        diff_p_real = np.power(np.real(points_diff) * 1.0 / scale_constellation * scale, 3)
        diff_p_imag = np.power(np.imag(points_diff) * 1.0 / scale_constellation * scale, 3)

        data = np.column_stack((np.absolute(points_x_orig), np.angle(points_x_orig),
                                np.real(points_x_orig), np.imag(points_x_orig),
                                points_x_labels,
                                np.real(points_diff), np.imag(points_diff),
                                np.real(points_diff_sq), np.imag(points_diff_sq),
                                np.real(points_diff_circle), np.imag(points_diff_circle),
                                diff_p_real, diff_p_imag,
                                np.real(points_diff_gauss), np.imag(points_diff_gauss),
                                np.absolute(points_x_with_neighbours), np.angle(points_x_with_neighbours),
                                np.absolute(points_y_with_neighbours), np.angle(points_y_with_neighbours),
                                np.real(points_x_with_neighbours), np.imag(points_x_with_neighbours),
                                np.real(points_y_with_neighbours), np.imag(points_y_with_neighbours)))

        # print(np.shape(data))

        df_result = pd.concat([df_result, pd.DataFrame(data, columns=columns)], ignore_index=True)

        # k = 1
        # one_row = [np.absolute(points_x_orig[k]), np.angle(points_x_orig[k]),
        #                         np.real(points_diff[k]), np.imag(points_diff[k]),
        #                         np.real(points_diff_sq[k]), np.imag(points_diff_sq[k]),
        #                         np.real(points_diff_circle[k]), np.imag(points_diff_circle[k])]
        # print(one_row, np.absolute(points_with_neighbours[k]), np.angle(points_with_neighbours[k]))
        # print(np.array(df_result.iloc[k]))
        # if np.isclose(np.array(df_result.iloc[0]), one_row):
        #     print('Error')

    return df_result


# almost the same as previous function. I am sorry. I really am.
def get_data_gauss_parallel(df, z_km, p_ave_dbm, n_channels, run_list, m_neighbours,
                            r=0.1, sigma=1.0, n_gauss=10,
                            scale_type='constellation'):
    # Create dataframe (table) with following columns:
    # neighbor symbols from -M to M, current symbol
    # optional: z_km, p_ave_dbm, n_channels
    # predict absolute value and angle for correction shift (in 2 dimensional plane -> real and imag part of symbol)

    # seed = datetime.now()
    # np.random.seed(seed)

    p_ave = (10 ** (p_ave_dbm / 10)) / 1000
    mod_type = hpcom.modulation.get_modulation_type_from_order(16)
    scale_constellation = hpcom.modulation.get_scale_coef_constellation(mod_type) / np.sqrt(p_ave / 2)
    constellation = hpcom.modulation.get_constellation(mod_type)

    scale = 1
    if scale_type == 'constellation':
        print('Scale data to correspond to initial constellation')
        scale = scale_constellation
    else:
        print('No such type of scale_type. Set scale to 1')

    columns = ['point_orig_abs', 'point_orig_angle',
               'point_orig_real', 'point_orig_imag',
               'point_label',
               'diff_real', 'diff_imag',
               'diff_sq_real', 'diff_sq_imag',
               'diff_circle_real', 'diff_circle_imag',
               'diff_p3_real', 'diff_p3_imag']

    columns = columns + [f'diff_gauss_real']
    columns = columns + [f'diff_gauss_imag']

    columns = columns + form_columns_names(m_neighbours, general_type='x')
    columns = columns + form_columns_names(m_neighbours, general_type='y')

    print(columns)

    df_result = pd.DataFrame(columns=columns)
    # print(df_result)

    for run in run_list:
        print('run number', run)
        df_current = df[(df['z_km'] == z_km) & (df['p_ave_dbm'] == p_ave_dbm) & (df['n_channels'] == n_channels) & (
                df['run'] == run)]

        points_x_orig = df_current['points_x_orig'].iloc[0][0] * scale
        points_x_scaled = hpcom.modulation.get_nearest_constellation_points_new(
            points_x_orig * scale_constellation / scale,
            constellation)
        # points_x_labels_str = list(map(str, points_x_scaled))
        points_x_labels = get_labels_for_points(points_x_scaled, constellation)

        points_x_shifted = df_current['points_x_shifted'].iloc[0] * scale
        points_x_with_neighbours = get_m_neighbours(points_x_shifted, m_neighbours)
        points_diff = points_x_orig - points_x_shifted

        print(np.shape(points_x_with_neighbours))

        # points_y_orig = df_current['points_y_orig'].iloc[0][0]
        points_y_shifted = df_current['points_y_shifted'].iloc[0] * scale
        points_y_with_neighbours = get_m_neighbours(points_y_shifted, m_neighbours)
        # points_y_scaled = sg.get_nearest_constellation_points_new(points_y_orig * scale_constellation, constellation)

        n_points = len(points_x_orig)
        mu, sigma = 0, sigma / (3.0 * scale_constellation / scale)
        # points_diff_gauss = np.tile(points_diff.reshape((n_points, 1)), (1, 1)) + (np.random.normal(mu, sigma, (n_points, 1)) +
        #                                                        1.0j * np.random.normal(mu, sigma, (n_points, 1)))

        points_diff_sq = np.zeros(n_points, dtype=complex)
        for k in range(n_points):
            points_diff_sq[k] = get_shift_to_square(points_x_orig[k], points_x_shifted[k],
                                                    r / scale_constellation * scale)

        points_diff_circle = get_shift_to_circle(points_x_orig, points_x_shifted, r / scale_constellation * scale)
        diff_p_real = np.power(np.real(points_diff) * 1.0 / scale_constellation * scale, 3)
        diff_p_imag = np.power(np.imag(points_diff) * 1.0 / scale_constellation * scale, 3)

        for k in range(n_gauss):
            points_diff_gauss = points_diff + (
                    np.random.normal(mu, sigma, n_points) + 1.0j * np.random.normal(mu, sigma, n_points))

            data = np.column_stack((np.absolute(points_x_orig), np.angle(points_x_orig),
                                    np.real(points_x_orig), np.imag(points_x_orig),
                                    points_x_labels,
                                    np.real(points_diff), np.imag(points_diff),
                                    np.real(points_diff_sq), np.imag(points_diff_sq),
                                    np.real(points_diff_circle), np.imag(points_diff_circle),
                                    diff_p_real, diff_p_imag,
                                    np.real(points_diff_gauss), np.imag(points_diff_gauss),
                                    np.absolute(points_x_with_neighbours), np.angle(points_x_with_neighbours),
                                    np.absolute(points_y_with_neighbours), np.angle(points_y_with_neighbours),
                                    np.real(points_x_with_neighbours), np.imag(points_x_with_neighbours),
                                    np.real(points_y_with_neighbours), np.imag(points_y_with_neighbours),))

            # print(np.shape(data))

            df_result = pd.concat([df_result, pd.DataFrame(data, columns=columns)], ignore_index=True)

        # k = 1
        # one_row = [np.absolute(points_x_orig[k]), np.angle(points_x_orig[k]),
        #                         np.real(points_diff[k]), np.imag(points_diff[k]),
        #                         np.real(points_diff_sq[k]), np.imag(points_diff_sq[k]),
        #                         np.real(points_diff_circle[k]), np.imag(points_diff_circle[k])]
        # print(one_row, np.absolute(points_with_neighbours[k]), np.angle(points_with_neighbours[k]))
        # print(np.array(df_result.iloc[k]))
        # if np.isclose(np.array(df_result.iloc[0]), one_row):
        #     print('Error')

    return df_result


def get_data_for_test(path_to_data, p_ave_dbm, z_km,
                      test_runs,
                      n_channels=1,
                      n_neighbours=4,
                      n_gauss=0):
    df = pd.read_pickle(path_to_data)

    sigma = 1.5

    df_tree_for_test = get_data_with_m_neighbor(df, z_km, p_ave_dbm, n_channels, test_runs, n_neighbours,
                                                r=0.1, n_gauss=n_gauss, scale_type='constellation')
    # df_tree_for_test = get_data_gauss_parallel(df, z_km, p_ave_dbm, n_channels, test_runs, n_neighbours, r=0.1, n_gauss=1, sigma=sigma)

    df = pd.DataFrame([])
    del df

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

    X_for_test = df_tree_for_test.drop(labels=labels_to_drop, axis=1)
    points_init_for_test = df_tree_for_test['point_orig_abs'].values * np.exp(
        1.0j * df_tree_for_test['point_orig_angle'])
    labels_for_test = df_tree_for_test['point_label']

    return X_for_test, points_init_for_test
