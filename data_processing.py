import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.fftpack import fft, fftfreq 
from scipy.signal import argrelextrema
import operator
from keras.utils import to_categorical

def process_data(raw_data: pd.DataFrame):
    num_columns = raw_data.shape[1]
    if num_columns == 7:
        return process_seven_columns(raw_data)
    elif num_columns == 4:
        return process_four_columns(raw_data)
    else:
        raise ValueError(f"Неподдерживаемый формат данных с {num_columns} столбцами")

def process_seven_columns(csv_data: pd.DataFrame):
    X = csv_data.iloc[:, :-1].values  # Первые 6 столбцов
    Y = csv_data.iloc[:, -1].values  # Последний столбец

    time_series_length = 45

    X_series = [X[i:i + time_series_length] for i in range(0, len(X), time_series_length) if len(X[i:i + time_series_length]) == time_series_length]
    Y_series = [Y[i:i + time_series_length] for i in range(0, len(Y), time_series_length) if len(Y[i:i + time_series_length]) == time_series_length]

    print(f"Количество временных рядов (7 столбцов): {len(X_series)}")  # Диагностика
    if len(X_series) == 0:
        raise ValueError("Нет данных для временных рядов (7 столбцов)")

    X_series_np = np.array(X_series)
    Y_series_np = np.array(Y_series)
    Y_compressed = compress_labels(Y_series_np)

    return X_series_np, Y_compressed

def process_four_columns(csv_data: pd.DataFrame):
    X = csv_data.iloc[:, :-1].values  # Первые 3 столбца
    Y = csv_data.iloc[:, -1].values  # Последний столбец

    X_train_expanded = np.expand_dims(X, axis=-1)

    print(f"Количество временных рядов (4 столбца): {len(X_train_expanded)}")  # Диагностика
    if len(X_train_expanded) == 0:
        raise ValueError("Нет данных для временных рядов (4 столбца)")
    X_series_np = np.array(X_train_expanded)

    print('X_series_np.: ',X_series_np.shape)
    Y_train = to_categorical(Y, 6)
    Y_compressed = compress_labels(Y_train)

    return X_series_np, Y_compressed

def compress_labels(Y_series_np):
    Y_compressed = np.array([y[0] for y in Y_series_np])
    for i in range(len(Y_compressed)):
        if Y_compressed[i] == 13:
            Y_compressed[i] = 9
        if Y_compressed[i] == 14:
            Y_compressed[i] = 10
        if Y_compressed[i] == 130:
            Y_compressed[i] = 11
        if Y_compressed[i] == 140:
            Y_compressed[i] = 12
    return Y_compressed

def generate_features(processed_data):
    num_features = processed_data.shape[2]

    if num_features == 6:
        return generate_features_six_columns(processed_data)
    elif num_features == 3:
        return generate_features_three_columns(processed_data)
    else:
        raise ValueError(f"Неподдерживаемый формат данных с {num_features} измерениями")

def generate_features_six_columns(processed_data):
    X_train1 = processed_data[:, :, 0]
    X_train2 = processed_data[:, :, 1]
    X_train3 = processed_data[:, :, 2]
    X_train4 = processed_data[:, :, 3]
    X_train5 = processed_data[:, :, 4]
    X_train6 = processed_data[:, :, 5]

    features = make_feature_vector(X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, Te=1/50)
    if features.size == 0:
        raise ValueError("Пустой массив признаков (6 измерений)")
    return features

def generate_features_three_columns(processed_data):
    X_train1 = processed_data[:, :, 0]
    X_train2 = processed_data[:, :, 1]
    X_train3 = processed_data[:, :, 2]

    features = make_feature_vector(X_train1, X_train2, X_train3, Te=1/50)
    if features.size == 0:
        raise ValueError("Пустой массив признаков (3 измерения)")
    return features

def stat_area_features(x, Te=1.0):
    if x.size == 0:
        raise ValueError("Пустой массив данных для статических функций")
    mean_ts = np.mean(x, axis=1).reshape(-1,1) 
    max_ts = np.amax(x, axis=1).reshape(-1,1)
    min_ts = np.amin(x, axis=1).reshape(-1,1)
    std_ts = np.std(x, axis=1).reshape(-1,1)
    skew_ts = st.skew(x, axis=1).reshape(-1,1)
    kurtosis_ts = st.kurtosis(x, axis=1).reshape(-1,1)
    iqr_ts = st.iqr(x, axis=1).reshape(-1,1)
    mad_ts = np.median(np.sort(abs(x - np.median(x, axis=1).reshape(-1,1)), axis=1), axis=1).reshape(-1,1)
    area_ts = np.trapz(x, axis=1, dx=Te).reshape(-1,1)
    sq_area_ts = np.trapz(x ** 2, axis=1, dx=Te).reshape(-1,1)

    return np.concatenate((mean_ts, max_ts, min_ts, std_ts, skew_ts, kurtosis_ts, iqr_ts, mad_ts, area_ts, sq_area_ts), axis=1)

def frequency_domain_features(x, Te=1.0):
    if x.shape[1] % 2 == 0:
        N = int(x.shape[1] / 2)
    else:
        N = int(x.shape[1] / 2) + 1
    xf = np.repeat(fftfreq(x.shape[1], d=Te)[:N].reshape(1, -1), x.shape[0], axis=0)
    dft = np.abs(fft(x, axis=1))[:, :N]
    dft_features = stat_area_features(dft, Te=1.0)
    dft_weighted_mean_f = np.average(xf, axis=1, weights=dft).reshape(-1, 1)
    dft_first_coef = dft[:, :5]
    dft_max_coef = np.zeros((x.shape[0], 5))
    dft_max_coef_f = np.zeros((x.shape[0], 5))
    for row in range(x.shape[0]):
        extrema_ind = argrelextrema(dft[row, :], np.greater, axis=0)
        extrema_row = sorted([(dft[row, :][j], xf[row, j]) for j in extrema_ind[0]],
                             key=operator.itemgetter(0), reverse=True)[:5]
        for i, ext in enumerate(extrema_row):
            dft_max_coef[row, i] = ext[0]
            dft_max_coef_f[row, i] = ext[1]
    
    return np.concatenate((dft_features, dft_weighted_mean_f, dft_first_coef, dft_max_coef, dft_max_coef_f), axis=1)

def make_feature_vector(*args, Te=1.0):
    features = []
    for arg in args:
        features.append(stat_area_features(arg, Te=Te))
        features.append(stat_area_features((arg[:,1:]-arg[:,:-1])/Te, Te=Te))
        features.append(frequency_domain_features(arg, Te=1/Te))
        features.append(frequency_domain_features((arg[:,1:]-arg[:,:-1])/Te, Te=1/Te))
    
    features = np.concatenate(features, axis=1)
    
    cor = np.empty((args[0].shape[0], len(args) * (len(args) - 1) // 2))
    k = 0
    for i in range(len(args)):
        for j in range(i + 1, len(args)):
            for row in range(args[i].shape[0]):
                cor[row, k] = np.corrcoef(args[i][row, :], args[j][row, :])[0, 1]
            k += 1

    features = np.concatenate((features, cor), axis=1)
    x_clean = np.nan_to_num(features, nan=0.0)
    return x_clean