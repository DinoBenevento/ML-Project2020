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



# create the dataset
def get_dataset():
    monk1_train = pd.read_csv("../data/monks-1.train", delimiter=" ")
    # split into input (X) and output (Y) variables
    X = np.array((monk1_train.iloc[:, 1:7]))
    y = np.array(monk1_train['result'])
    return X, y
# retrieve the model to be evaluate
def get_model():
    model = KNeighborsClassifier(n_neighbors=24, weights='distance', p=1, algorithm='auto', metric='chebyshev')
    return model

# evaluate the model using a given test condition
def evaluate_model(cv):
    # get the dataset
    X, y = get_dataset()
    # get the model
    model = get_model()
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # return scores
    return mean(scores), scores.min(), scores.max()

# calculate the ideal test condition
ideal, _, _ = evaluate_model(LeaveOneOut())
print('Ideal: %.3f' % ideal)
# define folds to test
folds = range(2,31)
# record mean and min/max of each set of results
means, mins, maxs = list(),list(),list()
# evaluate each k value
for k in folds:
    # define the test condition
    cv = StratifiedKFold(n_splits=k, shuffle=True)
    # evaluate k value
    k_mean, k_min, k_max = evaluate_model(cv)
    # report performance
    print('> folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
    # store mean accuracy
    means.append(k_mean)
    # store min and max relative to the mean
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)
# line plot of k mean values with min/max error bars
pyplot.errorbar(folds, means, yerr=[mins, maxs], fmt='o')
# plot the ideal case in a separate color
pyplot.plot(folds, [ideal for _ in range(len(folds))], color='r')
# show the plot
pyplot.show()