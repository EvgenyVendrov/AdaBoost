import data_prep
import utils
import hypothesis
from math import log as ln

'''
main file to run the algorithm
'''


# adaBoost algorithm
def AdaBoost_Algo(train_set, how_many_times_to_run, hypo_function):
    weights = []
    for i in range(0, len(train_set)):  # give equal weights to any data point
        weights.append(1 / len(train_set))
    H_set_of_hypos = []
    for i in range(0, how_many_times_to_run):  # run r times
        lowest_err_hypo = hypo_function(train_set, weights)  # get the best hypo (weighted err wise)
        alpha_for_hypo = (1 / 2) * ln((1 - lowest_err_hypo[3]) / lowest_err_hypo[3])
        H_set_of_hypos.append((lowest_err_hypo, alpha_for_hypo))  # collect all hypos for this round
        if lowest_err_hypo[
            3] >= 0.5:  # if eps (hypos weighted err) is at least half you can "skip" round - as alpha zero
            break
        utils.update_weights(alpha_for_hypo, lowest_err_hypo, train_set, weights, len(train_set),
                             hypo_function)  # update and normalize weights
        utils.normalize_weights(weights, len(weights))
    return H_set_of_hypos


# external wrapping function to run adaBoost as written in assignment
def external_entry_point(hypo_to_use, path_for_data_set=r"dataset\HC_Body_Temperature.txt", num_of_runs=8,
                         length_of_round=100):
    (data_set) = data_prep.get_data_set_from_path(path_for_data_set)  # read data set from path
    for i in range(1, num_of_runs + 1):
        list_of_acc_on_train = []
        list_of_acc_on_test = []
        for j in range(0, length_of_round):
            (shuffled_train_set, shuffled_test_set) = data_prep.shuffle_dataset(
                data_set)  # shuffle data every iteration
            H_set_of_hypos = AdaBoost_Algo(shuffled_train_set, i, hypo_to_use)  # get i best hypos
            train_acc = utils.calc_accuracy(shuffled_train_set, H_set_of_hypos,
                                            hypo_to_use)  # calc train acc on this hypo group
            list_of_acc_on_train.append(train_acc)  # collect to calc avg
            test_acc = utils.calc_accuracy(shuffled_test_set, H_set_of_hypos, hypo_to_use)
            list_of_acc_on_test.append(test_acc)
        print("avg TRAIN acc for round: ", i, " is: ", "%.3f" % utils.calc_avg(list_of_acc_on_train))
        print("avg TEST acc for round: ", i, " is: ", "%.3f" % utils.calc_avg(list_of_acc_on_test))


# main
print("=" * 20, "RECT", "=" * 20)
external_entry_point(hypothesis.Rectangle)
print("=" * 20, "CIRCL", "=" * 20)
external_entry_point(hypothesis.Circle)
