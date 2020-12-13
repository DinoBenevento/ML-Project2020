
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


monk1_train = pd.read_csv("../data/monks-1.train", delimiter=" ")
# split into input (X) and output (Y) variables
X = np.array((monk1_train.iloc[:, 1:7]))
y = np.array(monk1_train['result'])

skfold = StratifiedKFold(n_splits=16, shuffle=True)
cvscores = []

k_n = range(1, 31)
#leaf_size = range(1, 30)
weight_options = ['uniform', 'distance']
algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
ps = [1, 2]
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']


for train_index, test_index in skfold.split(X, y):
    model = KNeighborsClassifier(n_neighbors=24, weights='distance', p=1, algorithm='auto', metric='chebyshev')
    model.fit(X[train_index], y[train_index])
    score = model.score(X[test_index], y[test_index])
    print("Model " + "Test Mean Accuracy: %.2f%%" % (score * 100))
    cvscores.append(score * 100)
print("GLobal --> Accuracy mean : %.2f%%, Standard deviation: (+/- %.2f%%), Model Variance: %.2f" % (np.mean(cvscores), np.std(cvscores), np.std(cvscores)**2))






