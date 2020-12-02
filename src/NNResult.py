# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from src.PrintAndPlotResult import print_result_gridsearch
from src.Scaling import scaling_method
from keras.optimizers import SGD
from sklearn.model_selection import KFold, StratifiedKFold


def create_model():
    model = Sequential()
    model.add(Dense(6, input_dim=6, kernel_initializer='glorot_uniform', activation='relu'))
    #model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(lr=0.1, momentum=0.8)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# load dataset
monk1_train = pd.read_csv("../data/monks-1.train", delimiter=" ")
# split into input (X) and output (Y) variables
X = np.array((monk1_train.iloc[:, 1:7]))
y = np.array(monk1_train['result'])
X_train_scaled = scaling_method(X, "MinMax")
# create model
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
cvscores = []
for train, test in kfold.split(X, y):
    model = Sequential()
    model.add(Dense(6, input_dim=6, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(lr=0.1, momentum=0.8)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X[train], y[train], epochs=100, batch_size=15, verbose=0)
    scores = model.evaluate(X[test], y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))



