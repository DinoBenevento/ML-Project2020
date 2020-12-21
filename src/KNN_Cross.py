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
from src.Scaling import scaling_method
from sklearn.model_selection import cross_validate


def KNN_cross():
    cv = LeaveOneOut()
    monk1_train = pd.read_csv("../data/monks-1.train", delimiter=" ")
    X = np.array((monk1_train.iloc[:, 1:7]))
    y = np.array(monk1_train['result'])
    X_train_scaled = scaling_method(X, "Standard")
    model = KNeighborsClassifier(n_neighbors=10, weights='uniform', p=1, algorithm='auto', metric='manhattan')
    score = cross_validate(model, X_train_scaled, y, scoring='accuracy', cv=cv, n_jobs=-1, return_train_score=True)
    print('Test Score')
    print(score['test_score'])
    print('Train Score')
    print(score['train_score'])
    print('Mean Test Score')
    print(np.mean(score['test_score']))
    print('Mean Train Score')
    print(np.mean(score['train_score']))
    print('STD Test')
    print(np.std(score['test_score']))
    print('STD Bias')
    print(np.std(score['train_score']))
    #print(np.mean(score))

KNN_cross()


