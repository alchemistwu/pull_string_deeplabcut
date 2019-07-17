import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

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

def remove_low_confidence(data_dict, threshold = 0.75, padding='zero'):
    index_left = np.where(data_dict["p_left"] <= threshold)[0]
    index_right = np.where(data_dict["p_right"] <= threshold)[0]

    padding_num = 0
    data_dict["y_left"][index_left] = padding_num
    data_dict["y_right"][index_right] = padding_num

    if padding == 'zero':
        return data_dict
    elif padding == "linespace":
        slice_left = find_gap(index_left)
        slice_right = find_gap(index_right)
        data_dict = linespace_padding(slice_left, data_dict, "y_left")
        data_dict = linespace_padding(slice_right, data_dict, "y_right")

    return data_dict

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

def linespace_padding(slice_list, data_dict, dict_key):
    for i in range(len(slice_list)):
        (start_index, end_index) = slice_list[i]
        start_value = data_dict[dict_key][start_index - 1]
        end_value = data_dict[dict_key][end_index + 1]
        padding_list = np.linspace(start_value, end_value, end_index - start_index + 2)
        data_dict[dict_key][start_index - 1: end_index + 1] = padding_list
    return data_dict

def plot(data_dict):
    plt.figure(1)
    plt.subplot(211)
    y_left = data_dict["y_left"]
    x_left = np.linspace(0, y_left.shape[0], y_left.shape[0])
    plt.plot(x_left, y_left, 'k')

    plt.subplot(212)
    y_right = data_dict["y_right"]
    x_right = np.linspace(0, y_right.shape[0], y_right.shape[0])
    plt.plot(x_right, y_right, 'k')
    plt.show()


def save_result(input_dir, output_dir, data_dict):
    basename = os.path.basename(input_dir)
    save_dir = os.path.join(output_dir, basename)
    df = pd.DataFrame(data_dict)
    df.to_csv(save_dir)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--i', help='The full directory of input file name.', dest='input_file', type=str)
parser.add_argument('--o', help='The directory for saving result', dest='output_directory', type=str)
parser.add_argument('--s', help='Plot the result or not', dest='show', default=False, type=boolean_string)

args = parser.parse_args()

assert os.path.isfile(args.input_file), "Error: The input file name is not a file."
assert os.path.isdir(args.output_directory), "Error, The output directory is not a directory."

data_dict = load_data(args.input_file)
data_dict = remove_low_confidence(data_dict, padding="linespace")
save_result(args.input_file, args.output_directory, data_dict)
if args.show:
    plot(data_dict)






