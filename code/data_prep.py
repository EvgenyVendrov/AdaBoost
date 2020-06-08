from random import shuffle
import utils

'''
this file conatins all functions used to read data set from file (and handle its structure space-wise)
'''


# helping func for get_text_file_data
def handle_any_number_of_spaces(array_of_x_label_y, ):
    num_of_entrys = len(array_of_x_label_y)
    if num_of_entrys == 3:
        x_value = float(array_of_x_label_y[0])
        label = int(array_of_x_label_y[1])
        y_value = float(array_of_x_label_y[2])
    elif num_of_entrys == 2:
        x_value = float(array_of_x_label_y[0])
        splited_label_y = array_of_x_label_y[1].split("   ")
        label = int(splited_label_y[0])
        y_value = float(splited_label_y[1])
    else:
        array_of_x_label_y = array_of_x_label_y[0].split("   ")
        x_value = float(array_of_x_label_y[0])
        label = int(array_of_x_label_y[1])
        y_value = float(array_of_x_label_y[2])

    return (x_value, y_value, label)


# func to get the points_array which is tuple of - heat-heart-class and weight_array
def get_data_set_from_path(path):
    abs_file_path = utils.join_paths(path)
    points_array = []
    with open(abs_file_path, "r") as f:
        whole_line_wo_whitespaces = [line.rstrip() for line in f]
        for line in whole_line_wo_whitespaces:
            array_of_x_label_y = line.split("    ")
            points_array.append(handle_any_number_of_spaces(array_of_x_label_y))
    return (points_array)


def shuffle_dataset(whole_data):
    size_of_set = int(len(whole_data) / 2)
    shuffle(whole_data)
    train_set = []
    test_set = []
    for data_point in whole_data[:size_of_set]:
        train_set.append(data_point)
    for data_point in whole_data[size_of_set:]:
        test_set.append(data_point)
    return (train_set, test_set)
