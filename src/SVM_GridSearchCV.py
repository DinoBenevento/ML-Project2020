
from sklearn.model_selection import GridSearchCV
from sklearn import svm


def svm_grid_search_cv(X, y, cv_value):

    C = [1, 10, 100, 1000]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma = [1, 0.1, 0.01, 0.001, 0.0001]

    param_grid = dict(C=C, kernel=kernel, gamma=gamma)
    SVC = svm.SVC()
    grid = GridSearchCV(SVC, param_grid, cv=cv_value, scoring='accuracy', return_train_score=True)
    grid.fit(X, y)
    means_test = grid.cv_results_['mean_test_score']
    means_train = grid.cv_results_['mean_train_score']
    params = grid.cv_results_['params']
    for mean_train, mean_test, param in zip(means_train, means_test, params):
        print("Training accuracy: {:.2%}, Test accuracy: {:.2%} with: {}".format(mean_train, mean_test, param))
    print("Best test accuracy: {:.2%} using {}".format(grid.best_score_, grid.best_params_))
    return grid.cv_results_
