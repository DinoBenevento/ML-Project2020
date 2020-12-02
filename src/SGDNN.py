# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from src.PrintAndPlotResult import print_result_gridsearch
from src.Scaling import scaling_method
from keras.optimizers import SGD


def create_model(learn_rate, momentum):
    model = Sequential()
    model.add(Dense(6, input_dim=6, activation='relu'))
    #model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# load dataset
monk1_train = pd.read_csv("../data/monks-1.train", delimiter=" ")
# split into input (X) and output (Y) variables
X = np.array((monk1_train.iloc[:, 1:7]))
y = np.array(monk1_train['result'])
X_train_scaled = scaling_method(X, "MinMax")
# create model
model = KerasClassifier(build_fn=create_model, verbose=0, batch_size=30, epochs=70)
# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, return_train_score=True, scoring='accuracy')
grid.fit(X, y)
print_result_gridsearch(grid)