# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from src.PrintAndPlotResult import print_result_gridsearch

# Function to create model, required for KerasClassifier
def create_model():
    model = Sequential()
    model.add(Dense(6, input_dim=6, activation='relu'))
    #model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# load dataset
monk1_train = pd.read_csv("../data/monks-1.train", delimiter=" ")
# split into input (X) and output (Y) variables
X = np.array((monk1_train.iloc[:, 1:7]))
y = np.array(monk1_train['result'])
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
epochs = [10, 20, 30, 50, 60, 70, 80, 90, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, return_train_score=True, scoring='accuracy')
grid.fit(X, y)
print_result_gridsearch(grid)