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

import seaborn as sns
import numpy as np
import scipy as sp
import hpcom

import matplotlib.pyplot as plt

from composer import get_label_dict_for_constellation


def plot_feature_importance(importance, names, model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    # fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True))

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


def transform(points, scale):
    return points


def inverse_transform(points, scale):
    return points


def predict(model_real, model_imag, X_test, inverse_function=None, parameters=None):
    points_orig = X_test['point_x_abs'].values * np.exp(1.0j * X_test['point_x_angle'].values)
    if inverse_function == None:
        y_real_pred = model_real.predict(X_test)
        y_imag_pred = model_imag.predict(X_test)
    else:
        y_real_pred = inverse_function(model_real.predict(X_test), parameters[0])
        y_imag_pred = inverse_function(model_imag.predict(X_test), parameters[0])
    points_predict = np.real(points_orig) + y_real_pred + 1.0j * (np.imag(points_orig) + y_imag_pred)

    return points_predict


def predict_single(model, X_test, inverse_function=None, parameters=None):
    points_orig = X_test['point_x_abs'].values * np.exp(1.0j * X_test['point_x_angle'].values)
    if inverse_function == None:
        y_pred = model.predict(X_test)
    else:
        y_pred = inverse_function(model.predict(X_test), parameters[0])
    points_predict = np.real(points_orig) + y_pred[:,0] + 1.0j * (np.imag(points_orig) + y_pred[:,1])

    return points_predict


def inverse_p5(values, scale):
    return np.power(np.absolute(values), 1./5.) * np.sign(values) * scale


def inverse_p3(values, scale):
    return np.power(np.absolute(values), 1./3.) * np.sign(values) * scale


def predict_and_eval(model, X_test, points_init, p_ave_dbm, name='test'):

    # p_ave_dbm_expl = 6
    p_ave_dbm_expl = p_ave_dbm
    print("P_ave [dBm] = ", p_ave_dbm_expl)
    p_ave_expl = (10 ** (p_ave_dbm_expl / 10)) / 1000
    mod_type = hpcom.modulation.get_modulation_type_from_order(16)
    scale_constellation = hpcom.modulation.get_scale_coef_constellation(mod_type) / np.sqrt(p_ave_expl / 2)
    constellation = hpcom.modulation.get_constellation('16qam')

    points_orig = X_test['point_x_abs'].values * np.exp(1.0j * X_test['point_x_angle'].values)
    # points_predict = predict(model_real, model_imag, X_test, inverse_function=inverse_p3, parameters=[scale_constellation])
    if len(model) == 2:
        points_predict = predict(model[0], model[1], X_test, inverse_function=None, parameters=None)
    else:
        points_predict = predict_single(model, X_test, inverse_function=None, parameters=None)



    p_found_orig_for_test = hpcom.modulation.get_nearest_constellation_points_new(points_orig * scale_constellation, constellation)
    p_found_pred_for_test = hpcom.modulation.get_nearest_constellation_points_new(points_predict * scale_constellation, constellation)

    ber_orig = hpcom.metrics.get_ber_by_points(points_init * scale_constellation, p_found_orig_for_test, '16qam')
    ber_predict = hpcom.metrics.get_ber_by_points(points_init * scale_constellation, p_found_pred_for_test, '16qam')

    q_orig = np.sqrt(2) * sp.special.erfcinv(2 * ber_orig[0])
    q_pred = np.sqrt(2) * sp.special.erfcinv(2 * ber_predict[0])
    q_orig_db = 20 * np.log10(q_orig)
    q_pred_db = 20 * np.log10(q_pred)
    evm_orig = hpcom.metrics.get_evm_rms_new(points_init, points_orig)
    evm_pred = hpcom.metrics.get_evm_rms_new(points_init, points_predict)
    ber_evm_orig = hpcom.metrics.get_ber_from_evm(points_init, points_orig, 16)
    ber_evm_pred = hpcom.metrics.get_ber_from_evm(points_init, points_predict, 16)

    name = name.center(len(name) + 10, '-')
    print_out = [name, f'Number of points {np.shape(points_init)}',
                 f'ber (orig / pred / delta) ({ber_orig} / {ber_predict} / {ber_orig[0] - ber_predict[0]})',
                 f'ber from EVM_rms (orig / pred / delta) ({ber_evm_orig} / {ber_evm_pred} / {ber_evm_orig - ber_evm_pred})',
                 f'q-factor [dB] (orig / pred / delta) ({q_orig_db} / {q_pred_db} / {q_pred_db - q_orig_db})',
                 f'EVM [%] (orig / pred / delta) ({evm_orig * 100.} / {evm_pred * 100.} / {(evm_orig - evm_pred) * 100}']

    f = open('log.txt', 'a')
    for line in print_out:
        print(line)
        f.write(line + '\n')

    f.close()

    return points_orig, points_predict


def plot_result(p_ave_dbm, points_predict_for_test, points_orig_for_test, points_predict_train, points_orig_train):
    p_ave_dbm_expl = p_ave_dbm
    p_ave_expl = (10 ** (p_ave_dbm_expl / 10)) / 1000
    mod_type = hpcom.modulation.get_modulation_type_from_order(16)
    scale_constellation = hpcom.modulation.get_scale_coef_constellation(mod_type) / np.sqrt(p_ave_expl / 2)
    constellation = hpcom.modulation.get_constellation('16qam')

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs[0][0].scatter(points_predict_for_test.real * scale_constellation,
                      points_predict_for_test.imag * scale_constellation,
                      s=10, c='blue', marker='*')
    axs[0][0].scatter(constellation.real,
                      constellation.imag,
                      s=60, color='xkcd:bright red', marker='.')
    # axs[0][0].scatter(points_predict_iter.real * scale_constellation,
    #                points_predict_iter.imag * scale_constellation,
    #                s=10, c='xkcd:purple', marker='*')
    axs[0][0].grid(True)

    axs[0][1].scatter(points_orig_for_test.real * scale_constellation,
                      points_orig_for_test.imag * scale_constellation,
                      s=20, color='xkcd:bright green', marker='.')
    axs[0][1].scatter(points_predict_for_test.real * scale_constellation,
                      points_predict_for_test.imag * scale_constellation,
                      s=10, c='blue', marker='*')
    axs[0][1].grid(True)

    axs[1][0].scatter(points_predict_train.real * scale_constellation,
                      points_predict_train.imag * scale_constellation,
                      s=10, c='blue', marker='*')
    axs[1][0].scatter(constellation.real,
                      constellation.imag,
                      s=60, color='xkcd:bright red', marker='.')
    axs[1][0].grid(True)

    axs[1][1].scatter(points_orig_train.real * scale_constellation,
                      points_orig_train.imag * scale_constellation,
                      s=20, color='xkcd:bright green', marker='.')
    axs[1][1].scatter(points_predict_train.real * scale_constellation,
                      points_predict_train.imag * scale_constellation,
                      s=10, c='blue', marker='*')
    axs[1][1].grid(True)


def labels_to_points(labels, constellation):
    _, dict = get_label_dict_for_constellation(constellation)

    points = np.zeros(len(labels), dtype=complex)
    k = 0
    for l in labels:
        points[k] = complex(dict[l])
        k += 1

    return points


def predict_clf(model_clf, X_test, constellation):

    y_label_pred = model_clf.predict(X_test)
    return labels_to_points(y_label_pred, constellation)


def predict_clf_and_eval(model_clf, X_test, points_init, name='test', filename='log.txt'):

    p_ave_dbm_expl = 6
    p_ave_expl = (10 ** (p_ave_dbm_expl / 10)) / 1000
    mod_type = sg.get_modulation_type_from_order(16)
    scale_constellation = sg.get_scale_coef_constellation(mod_type) / np.sqrt(p_ave_expl / 2)
    constellation = sg.get_constellation('16qam')

    points_orig = X_test['point_x_abs'].values * np.exp(1.0j * X_test['point_x_angle'].values)
    points_predict = predict_clf(model_clf, X_test, constellation)
    print(np.shape(points_orig), np.shape(points_predict))


    p_found_orig_for_test = sg.get_nearest_constellation_points_new(points_orig * scale_constellation, constellation)
    p_found_pred_for_test = sg.get_nearest_constellation_points_new(points_predict, constellation)

    ber_orig = sg.get_ber_by_points(points_init * scale_constellation, p_found_orig_for_test, '16qam')
    ber_predict = sg.get_ber_by_points(points_init * scale_constellation, p_found_pred_for_test, '16qam')

    q_orig = np.sqrt(2) * sp.special.erfcinv(2 * ber_orig[0])
    q_pred = np.sqrt(2) * sp.special.erfcinv(2 * ber_predict[0])
    q_orig_db = 20 * np.log10(q_orig)
    q_pred_db = 20 * np.log10(q_pred)
    evm_orig = sg.get_evm_rms_new(points_init, points_orig)
    evm_pred = sg.get_evm_rms_new(points_init, points_predict / scale_constellation)
    ber_evm_orig = sg.get_ber_from_evm(points_init, points_orig, 16)
    ber_evm_pred = sg.get_ber_from_evm(points_init, points_predict / scale_constellation, 16)

    name = name.center(len(name) + 10, '-')
    print_out = [name, f'Number of points {np.shape(points_init)}',
                 f'ber (orig / pred / delta) ({ber_orig} / {ber_predict} / {ber_orig[0] - ber_predict[0]})',
                 f'ber from EVM_rms (orig / pred / delta) ({ber_evm_orig} / {ber_evm_pred} / {ber_evm_orig - ber_evm_pred})',
                 f'q-factor [dB] (orig / pred / delta) ({q_orig_db} / {q_pred_db} / {q_pred_db - q_orig_db})',
                 f'EVM [%] (orig / pred / delta) ({evm_orig * 100.} / {evm_pred * 100.} / {(evm_orig - evm_pred) * 100}']

    f = open(filename, 'a')
    for line in print_out:
        print(line)
        f.write(line + '\n')

    f.close()

    return points_orig, points_predict