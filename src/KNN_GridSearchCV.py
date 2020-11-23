
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def knn_grid_search_cv(nn_start, nn_end, X, y, cv_value):

    k_n = range(nn_start, nn_end)
    leaf_size = range(1, 30)
    weight_options = ['uniform', 'distance']
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    ps = [1, 2]
    metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

    param_grid = dict(n_neighbors=k_n, leaf_size=leaf_size, weights=weight_options, p=ps, algorithm=algorithms, metric=metrics)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=cv_value, scoring='accuracy', return_train_score=True)
    grid.fit(X, y)
    print("Best test accuracy: {:.2%} using {}" .format(grid.best_score_, grid.best_params_))
    means_test = grid.cv_results_['mean_test_score']
    means_train = grid.cv_results_['mean_train_score']
    params = grid.cv_results_['params']
    for mean_train, mean_test, param in zip(means_train, means_test, params):
        print("Training accuracy: {:.2%}, Test accuracy: {:.2%} with: {}" .format(mean_train, mean_test, param))
    return grid.cv_results_
    #df = pd.DataFrame(grid.cv_results_)
    # summarize results
    #print(df[['param_n_neighbors', 'param_p', 'param_weights', 'mean_test_score', 'mean_train_score']])
