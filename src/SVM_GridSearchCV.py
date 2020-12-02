
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from src.PrintAndPlotResult import print_result_gridsearch


def svm_grid_search_cv(X, y, cv_value):

    C = [0.1, 1, 10, 100]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma = ['auto', 'scale', 10, 1, 0.1, 0.01, 0.001, 0.0001]

    param_grid = dict(C=C, kernel=kernel, gamma=gamma)
    SVC = svm.SVC()
    grid = GridSearchCV(SVC, param_grid, cv=cv_value, scoring='accuracy', return_train_score=True)
    grid.fit(X, y)
    print_result_gridsearch(grid)
    return grid.cv_results_
