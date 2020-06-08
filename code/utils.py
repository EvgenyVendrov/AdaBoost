from math import exp
import os
import hypothesis

'''
utilities used for implementing main function
1. weights updating and normalizing
2.logging weights
3. labeling a point using whole H
4. calc the percentage of acc on some H and data set  
'''


def update_weights(alpha, hypo, dataset, weights, size_of_set, hypo_type):
    for i in range(0, size_of_set):
        if not hypothesis.is_point_labled_right(hypo_type, hypo[0], hypo[1], dataset[i], hypo[2]):
            weights[i] *= exp(alpha)
        else:
            weights[i] *= exp(-1 * alpha)


def normalize_weights(weights, size_of_set):
    sum_of_weights = sum(weights)
    for i in range(0, size_of_set):
        weights[i] /= sum_of_weights


def log_weights(runing_string_for_log, path):
    abs_file_path = join_paths(path)
    with open(abs_file_path, 'w') as f:
        f.write(runing_string_for_log)
    print("weights of whole execution logged to ", abs_file_path)


def join_paths(path):
    script_dir = os.path.dirname(__file__)[:os.path.dirname(__file__).index("\code")]
    return os.path.join(script_dir, path)


def calc_accuracy(dataset, list_of_hypos, hypo_type):
    list_of_labels_for_set = []
    for data_point in dataset:
        list_of_labels_for_set.append(hypothesis.label_point_with_all_hypos(data_point, list_of_hypos, hypo_type))
    return calc_percentage(dataset, list_of_labels_for_set)


def calc_avg(list_of_acc):
    return (sum(list_of_acc) / len(list_of_acc))


def calc_percentage(dataset, labels):
    true_labeled = 0
    index_for_label = 0
    for point in dataset:
        if labels[index_for_label] == point[2]:
            true_labeled += 1
        index_for_label += 1
    return (true_labeled / len(dataset))
