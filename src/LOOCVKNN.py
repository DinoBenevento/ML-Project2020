
# sensitivity analysis of k in k-fold cross-validation
from numpy import mean
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold


def LOOV_KNN():
    k_n = range(1, 31)
    weight_options = ['uniform', 'distance']
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    ps = [1, 2]
    metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

    monk1_train = pd.read_csv("../data/monks-1.train", delimiter=" ")
    # split into input (X) and output (Y) variables
    X = np.array((monk1_train.iloc[:, 1:7]))
    y = np.array(monk1_train['result'])
    means = []
    params = []
    cv = LeaveOneOut()
    for kn in k_n:
        for weight in weight_options:
            for alg in algorithms:
                for p in ps:
                    for metric in metrics:
                        params = []
                        params.append(kn)
                        params.append(weight)
                        params.append(alg)
                        params.append(p)
                        params.append(metric)
                        model = KNeighborsClassifier(n_neighbors=kn, weights=weight, p=p, algorithm=alg, metric=metric)
                        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
                        print(params)
                        means.append(np.mean(scores))
                        print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    print('Max Accuracy: %.3f' % (np.max(means)))


LOOV_KNN()