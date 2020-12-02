import numpy as np


def print_result_gridsearch(grid):
    means_test = grid.cv_results_['mean_test_score']
    stds_test = grid.cv_results_['std_test_score']
    stds_train = grid.cv_results_['std_train_score']
    means_train = grid.cv_results_['mean_train_score']
    params = grid.cv_results_['params']
    for mean_train, std_train, mean_test, std_test, param in zip(means_train, stds_train, means_test, stds_test, params):
        print("Training accuracy: {:.2%}, StdTrain: {:.2}, Test accuracy: {:.2%}, StdTest: {:.2} , with: {}".format(mean_train, std_train, mean_test,
                                                                                                                    std_test, param))
        print('Variance:' + str(std_test**2))
        print('\n')
    print("Best test accuracy: {:.2%} using {}".format(grid.best_score_, grid.best_params_))
    return grid.best_params_








