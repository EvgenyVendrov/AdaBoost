from itertools import combinations
from math import sqrt

'''
hypothesis used for separation of data points in adaBoost algo, module contains: 
1. implementation of rectangle hypothesis
2. implementation of circle hypothesis
3. helping functions: to find lowest err hypo from list, to sum all train errors, label points by hypos and checking whether a point tagged correctly
'''


def Rectangle(train_set, weights):
    '''
    this function draws a rectangle on every two data points (n choose 2) form the train set -
    and returns the one with the lowest weighted error on the train set
    '''
    n_choose_2 = combinations(train_set, 2)
    list_of_rects = []
    for pair in n_choose_2:
        error_on_positive_direction = sum_all_train_error_weigths_on_this_hypo(pair[0], pair[1], train_set,
                                                                               weights, 1, Rectangle)
        error_on_negative_direction = sum_all_train_error_weigths_on_this_hypo(pair[0], pair[1], train_set,
                                                                               weights, -1, Rectangle)
        if error_on_negative_direction < error_on_positive_direction:  # every rectangle can separate negative or positive points inside - so we'll choose the best one
            list_of_rects.append((pair[0], pair[1], -1, error_on_negative_direction))
        else:
            list_of_rects.append((pair[0], pair[1], 1, error_on_positive_direction))
    return find_lowest_err_hypo(list_of_rects)


def Circle(train_set, weights):
    '''
        this function draws a circles on every two data points (n choose 2) form the train set -
        and returns the one with the lowest weighted error on the train set
        '''
    n_choose_2 = combinations(train_set, 2)
    list_of_circles = []
    for pair in n_choose_2:  # every two points are 4 circles - switching the center and perimeter and negativity on the inside
        error_on_positive_direction_first_center = sum_all_train_error_weigths_on_this_hypo(pair[0], pair[1],
                                                                                            train_set,
                                                                                            weights, 1,
                                                                                            Circle)

        error_on_positive_direction_sec_center = sum_all_train_error_weigths_on_this_hypo(pair[1], pair[0],
                                                                                          train_set,
                                                                                          weights, 1,
                                                                                          Circle)

        error_on_negative_direction_first_center = sum_all_train_error_weigths_on_this_hypo(pair[0], pair[1],
                                                                                            train_set,
                                                                                            weights, -1,
                                                                                            Circle)

        error_on_negative_direction_sec_center = sum_all_train_error_weigths_on_this_hypo(pair[1], pair[0],
                                                                                          train_set,
                                                                                          weights, -1,
                                                                                          Circle)
        min_err = min([error_on_positive_direction_first_center, error_on_positive_direction_sec_center,
                       error_on_negative_direction_first_center,
                       error_on_negative_direction_sec_center])  # lowest weighted error is chosen out of the four

        if min_err == error_on_positive_direction_first_center:
            list_of_circles.append((pair[0], pair[1], 1, error_on_positive_direction_first_center))
        elif min_err == error_on_positive_direction_sec_center:
            list_of_circles.append((pair[1], pair[0], 1, error_on_positive_direction_sec_center))
        elif min_err == error_on_negative_direction_first_center:
            list_of_circles.append((pair[0], pair[1], -1, error_on_negative_direction_first_center))
        else:
            list_of_circles.append((pair[1], pair[0], -1, error_on_negative_direction_sec_center))
    return find_lowest_err_hypo(list_of_circles)


# helping function used to keep only lowest err hypo in the hypothesis functions
def find_lowest_err_hypo(list_of_hypos):
    lowest_err = list_of_hypos[0][3]
    tuple_to_ret = list_of_hypos[0]
    for hypo_tuple in list_of_hypos:
        if hypo_tuple[3] < lowest_err:
            lowest_err = hypo_tuple[3]
            tuple_to_ret = hypo_tuple
    return tuple_to_ret


# helping function used to sum all errors weights for given hypo in the hypothesis functions
def sum_all_train_error_weigths_on_this_hypo(point_of_hypo1, point_of_hypo2, set_of_points_to_check, weights,
                                             dir_of_hypo,
                                             hypo_type):
    tot_err = 0
    for i in range(0, len(set_of_points_to_check)):
        if not is_point_labled_right(hypo_type, point_of_hypo1, point_of_hypo2, set_of_points_to_check[i],
                                     dir_of_hypo):
            tot_err += weights[i]
    return tot_err


# helping function used in "utils" module to determine whether the label a data point got by hypo is right
def is_point_labled_right(hypo_type, point_of_hypo1, point_of_hypo2, point_to_check, dir_of_hypo):
    if hypo_type.__name__ == "Rectangle":
        label = tag_point_rect(point_of_hypo1, point_of_hypo2, point_to_check, dir_of_hypo)
        if point_to_check[2] == label:
            return True
        else:
            return False
    else:
        label = tag_point_circle(point_of_hypo1, point_of_hypo2, point_to_check, dir_of_hypo)
        if point_to_check[2] == label:
            return True
        else:
            return False


# helping function used in "utils" module to label a data point by using hypos and their weights
def label_point_with_all_hypos(datapoint, list_of_hypos_and_weights, hypo_type):
    sum_of_hypos_weights_for_this_data_point = 0
    for j in range(0, len(list_of_hypos_and_weights)):
        if hypo_type.__name__ == "Rectangle":
            sum_of_hypos_weights_for_this_data_point += tag_point_rect(list_of_hypos_and_weights[j][0][0],
                                                                       list_of_hypos_and_weights[j][0][1],
                                                                       datapoint,
                                                                       list_of_hypos_and_weights[j][0][2]) * \
                                                        list_of_hypos_and_weights[j][1]

        else:
            sum_of_hypos_weights_for_this_data_point += tag_point_circle(list_of_hypos_and_weights[j][0][0],
                                                                         list_of_hypos_and_weights[j][0][1],
                                                                         datapoint,
                                                                         list_of_hypos_and_weights[j][0][
                                                                             2]) * \
                                                        list_of_hypos_and_weights[j][1]
    if sum_of_hypos_weights_for_this_data_point < 0:
        return -1
    else:
        return 1


# helping function used in label_point_with_all_hypos function to determine whether data point is inside given rect
def tag_point_rect(point_of_hypo1, point_of_hypo2, point_to_check, dir_of_hypo):
    x_val_high = point_of_hypo1[0] if point_of_hypo1[0] > point_of_hypo2[0] else point_of_hypo2[0]
    x_val_low = point_of_hypo1[0] if point_of_hypo1[0] < point_of_hypo2[0] else point_of_hypo2[0]
    y_val_high = point_of_hypo1[1] if point_of_hypo1[1] > point_of_hypo2[1] else point_of_hypo2[1]
    y_val_low = point_of_hypo1[1] if point_of_hypo1[1] < point_of_hypo2[1] else point_of_hypo2[1]
    if x_val_high == x_val_low:
        if point_to_check[0] <= x_val_high and point_to_check[1] >= y_val_low and point_to_check[1] <= y_val_high:
            if dir_of_hypo == 1:
                return 1
            else:
                return -1
        else:
            if dir_of_hypo == 1:
                return -1
            else:
                return 1
    elif y_val_low == y_val_high:
        if point_to_check[1] <= y_val_high and point_to_check[0] >= x_val_low and point_to_check[0] <= x_val_high:
            if dir_of_hypo == 1:
                return 1
            else:
                return -1
        else:
            if dir_of_hypo == 1:
                return -1
            else:
                return 1
    else:
        if point_to_check[0] <= x_val_high and point_to_check[0] >= x_val_low and point_to_check[1] >= y_val_low and \
                point_to_check[1] <= y_val_high:
            if dir_of_hypo == 1:
                return 1
            else:
                return -1
        else:
            if dir_of_hypo == 1:
                return -1
            else:
                return 1


# helping function used in label_point_with_all_hypos function to determine whether data point is inside given circle
def tag_point_circle(center_of_circle, point_on_perimeter, point_to_check, dir_of_hypo):
    radius_of_circle = calc_euclid_dist(center_of_circle, point_on_perimeter)
    dist_from_center_to_point = calc_euclid_dist(center_of_circle, point_to_check)
    if dist_from_center_to_point <= radius_of_circle:
        if dir_of_hypo == 1:
            return 1
        else:
            return -1
    else:
        if dir_of_hypo == 1:
            return -1
        else:
            return 1


# helping function used in tag_point_circle to calc euclidean distance given two points
def calc_euclid_dist(point1, point2):
    x_diff = point1[0] - point2[0]
    y_diff = point1[1] - point2[1]
    x_diff_by_2 = x_diff ** 2
    y_diff_by_2 = y_diff ** 2
    euclid_dist = sqrt(x_diff_by_2 + y_diff_by_2)
    return euclid_dist
