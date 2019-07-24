import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import copy
from scipy.signal import medfilt
from scipy import signal

def load_data(directory):
    df = pd.read_csv(directory)
    data = df.values[3:, np.asarray([2, 3, 5, 6])]
    new_df = pd.DataFrame(data=data)
    df = new_df.astype(float)
    df.columns = ["y_left", "p_left", "y_right", "p_right"]
    data_dict = df.to_dict('list')
    new_dict = {}
    for key in data_dict.keys():
        new_dict[key] = np.asarray(data_dict[key])
    return new_dict

def remove_low_confidence(data_dict, threshold = 0.9, filter='median', padding='zero'):
    index_left = np.where(data_dict["p_left"] <= threshold)[0]
    index_right = np.where(data_dict["p_right"] <= threshold)[0]

    padding_num = 0
    data_dict_filted = copy.deepcopy(data_dict)

    data_dict_filted["y_left"][index_left] = padding_num
    data_dict_filted["y_right"][index_right] = padding_num
    if padding == 'zero':
        padding_num = 0
        data_dict_filted["y_left"][index_left] = padding_num
        data_dict_filted["y_right"][index_right] = padding_num

    elif padding == "linespace":
        slice_left = find_gap(index_left)
        slice_right = find_gap(index_right)
        data_dict_filted = linespace_padding(slice_left, data_dict_filted, "y_left")
        data_dict_filted = linespace_padding(slice_right, data_dict_filted, "y_right")

    if filter == 'median':
        data_dict_filted["y_left"] = medfilt(data_dict_filted["y_left"], kernel_size=11)
        data_dict_filted["y_right"] = medfilt(data_dict_filted["y_right"], kernel_size=11)
    else:
        b, a = signal.butter(7, 0.1, 'lowpass')
        data_dict_filted["y_left"] = signal.filtfilt(b, a, data_dict_filted["y_left"])
        data_dict_filted["y_right"] = signal.filtfilt(b, a, data_dict_filted["y_right"])

    return data_dict, data_dict_filted

def find_gap(index_list):
    slice_list = []
    start_point = None
    end_point = None

    for i in range(index_list.shape[0] - 1):
        if start_point is None:
            start_point = index_list[i]
            end_point = start_point + 1
        if index_list[i + 1] - index_list[i] == 1:
            end_point = index_list[i + 1]
        else:
            slice_list.append((start_point, end_point))
            start_point = None
            end_point = None

    return slice_list

def find_changes(data_array):

    current_state = None
    change_points = []
    for i in range(data_array.shape[0] - 1):

        if data_array[i + 1] - data_array[i] > 0 :
            if current_state != 1:
                change_points.append(i)
            current_state = 1
        elif data_array[i + 1] - data_array[i] < 0 :
            if current_state != -1:
                change_points.append(i)
            current_state = -1

    return change_points

def linespace_padding(slice_list, data_dict, dict_key):
    for i in range(len(slice_list)):
        (start_index, end_index) = slice_list[i]
        start_value = data_dict[dict_key][start_index - 1]
        end_value = data_dict[dict_key][end_index + 1]
        if start_index == 0:
            padding_list = np.linspace(end_value, end_value, end_index - start_index + 2)
        else:
            padding_list = np.linspace(start_value, end_value, end_index - start_index + 2)
        data_dict[dict_key][start_index: end_index + 2] = padding_list
    return data_dict

def plot(data_dict, data_dict_med, result_dict):
    plt.figure(1)
    plt.subplot(211)
    y_left = data_dict["y_left"]
    y_left_med = data_dict_med["y_left"]
    x_left = np.linspace(0, y_left.shape[0], y_left.shape[0])

    plt.scatter(result_dict["index_left"], result_dict["y_left"], s=40, zorder=2)
    plt.plot(x_left, y_left, 'k')
    plt.plot(x_left, y_left_med, 'r')

    plt.subplot(212)
    y_right = data_dict["y_right"]
    y_right_med = data_dict_med["y_right"]

    x_right = np.linspace(0, y_right.shape[0], y_right.shape[0])
    plt.scatter(result_dict["index_right"], result_dict["y_right"], s=40, zorder=2)
    plt.plot(x_right, y_right, 'k')
    plt.plot(x_right, y_right_med, 'r')
    plt.show()


def save_result(input_dir, output_dir, data_dict):
    basename = os.path.basename(input_dir)
    save_dir = os.path.join(output_dir, "result_" +basename)
    df = pd.DataFrame(data_dict)
    df.to_csv(save_dir)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_slide_window(i, size, window_size=21):
    if i < (window_size - 1)/2:
        start = 0
        end = window_size
    elif i + (window_size - 1)/2 > size - 1:
        start = size - window_size - 1
        end = size - 1
    else:
        start = i - (window_size - 1)/2
        end = i + (window_size - 1)/2
    return np.linspace(start, end, end - start + 1).astype(np.int)


def summary_dict(data_dict, data_dict_med, points_left, points_right, threshold=0.9):
    result_dict = {"index_left":[], "y_left":[], "index_gap_left":[], "value_gap_left":[], "index_right":[], "y_right":[], "index_gap_right":[], "value_gap_right":[]}

    length = len(data_dict["y_left"])
    last_window_index = 0
    for i in range(len(points_left) - 1):
        window = get_slide_window(points_left[i], length)
        window = window[np.where(window >= last_window_index)]
        candidates = data_dict["y_left"][window]
        p_candidates = data_dict["p_left"][window]

        if data_dict_med["y_left"][points_left[i]] > data_dict_med["y_left"][points_left[i + 1]]:
            candidates[np.where(p_candidates < threshold)] = 0
            index = np.argmax(candidates)
            result_dict["index_left"].append(window[index])
            result_dict["y_left"].append(data_dict["y_left"][window[index]])
        else:
            candidates[np.where(p_candidates < threshold)] = 65336
            index = np.argmin(candidates)
            result_dict["index_left"].append(window[index])
            result_dict["y_left"].append(data_dict["y_left"][window[index]])
        if i == 0:
            result_dict["index_gap_left"].append(0)
            result_dict["value_gap_left"].append(0)
        else:
            result_dict["index_gap_left"].append(abs(result_dict["index_left"][i] - result_dict["index_left"][i - 1]))
            result_dict["value_gap_left"].append(result_dict["y_left"][i] - result_dict["y_left"][i - 1])
        last_window_index = window[index]



    length = len(data_dict["y_right"])
    last_window_index = 0
    for i in range(len(points_right) - 1):
        window = get_slide_window(points_right[i], length)
        window = window[np.where(window >= last_window_index)]
        candidates = data_dict["y_right"][window]
        p_candidates = data_dict["p_right"][window]
        if data_dict_med["y_right"][points_right[i]] > data_dict_med["y_right"][points_right[i + 1]]:
            candidates[np.where(p_candidates < threshold)] = 0
            index = np.argmax(candidates)
            result_dict["index_right"].append(window[index])
            result_dict["y_right"].append(data_dict["y_right"][window[index]])
        else:
            candidates[np.where(p_candidates < threshold)] = 65336
            index = np.argmin(candidates)
            result_dict["index_right"].append(window[index])
            result_dict["y_right"].append(data_dict["y_right"][window[index]])
        if i == 0:
            result_dict["index_gap_right"].append(0)
            result_dict["value_gap_right"].append(0)
        else:
            result_dict["index_gap_right"].append(abs(result_dict["index_right"][i] - result_dict["index_right"][i - 1]))
            result_dict["value_gap_right"].append(result_dict["y_right"][i] - result_dict["y_right"][i - 1])
        last_window_index = window[index]

    max_length = max(len(points_left), len(points_right))
    if len(points_left) > len(points_right):
        zeros = [0 for i in range(max_length - len(points_right))]
        result_dict["index_right"].extend(zeros)
        result_dict["y_right"].extend(zeros)
        result_dict["index_gap_right"].extend(zeros)
        result_dict["value_gap_right"].extend(zeros)
    else:
        zeros = [0 for i in range(max_length - len(points_left))]
        result_dict["index_left"].extend(zeros)
        result_dict["y_left"].extend(zeros)
        result_dict["index_gap_left"].extend(zeros)
        result_dict["value_gap_left"].extend(zeros)


    return result_dict

DEBUG = False

if not DEBUG:
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', help='The full directory of input file name.', dest='input_file', type=str)
    parser.add_argument('--o', help='The directory for saving result', dest='output_directory', type=str)
    parser.add_argument('--s', help='Plot the result or not', dest='show', default=False, type=boolean_string)

    args = parser.parse_args()

    assert os.path.isfile(args.input_file), "Error: The input file name is not a file."
    assert os.path.isdir(args.output_directory), "Error, The output directory is not a directory."

    data_dict = load_data(args.input_file)
    data_dict, data_dict_med = remove_low_confidence(data_dict, filter='lowpass', padding="linespace")
    points_left = find_changes(data_dict_med['y_left'])
    points_right = find_changes(data_dict_med['y_right'])
    result_dict = summary_dict(data_dict, data_dict_med, points_left, points_right)

    save_result(args.input_file, args.output_directory, result_dict)
    if args.show:
        plot(data_dict, data_dict_med, result_dict)

else:
    input_file = "/home/silasi/mathew/data/32833-01172019170245DeepCut_resnet50_String Pull MergeMay22shuffle1_493000.csv"
    data_dict = load_data(input_file)
    data_dict, data_dict_med = remove_low_confidence(data_dict, filter='lowpass', padding="linespace")

    points_left = find_changes(data_dict_med['y_left'])
    points_right = find_changes(data_dict_med['y_right'])
    result_dict = summary_dict(data_dict, data_dict_med, points_left, points_right)

    plot(data_dict, data_dict_med, result_dict)
    save_result(input_file, "/home/silasi/mathew/output", result_dict)




