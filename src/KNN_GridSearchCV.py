
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from src.PrintAndPlotResult import print_result_gridsearch


def knn_grid_search_cv(nn_start, nn_end, X, y, cv_value):

    k_n = range(nn_start, nn_end)
    leaf_size = range(1, 30)
    weight_options = ['uniform', 'distance']
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    ps = [1, 2]
    metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

    cv_list = []

    param_grid = dict(n_neighbors=k_n, weights=weight_options, p=ps, algorithm=algorithms, metric=metrics)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=cv_value, scoring='accuracy', return_train_score=True)
    grid.fit(X, y)
    print_result_gridsearch(grid)



    #df = pd.DataFrame(grid.cv_results_)
    # summarize results
    #print(df[['param_n_neighbors', 'param_p', 'param_weights', 'mean_test_score', 'mean_train_score']])
