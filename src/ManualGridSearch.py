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
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

def create_model():
    model = Sequential()
    model.add(Dense(6, input_dim=6, kernel_initializer='glorot_uniform', activation='relu'))
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
cvscores = []
index_model = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
epochs = [10, 20, 30, 50, 60, 70, 80, 90, 100]
batch_size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
skfold = StratifiedKFold(n_splits=10, shuffle=True)
cvscores = []
model_list = []
index_model = 1
best_batch = 0
best_epoch = 0
best_accuracy = 0
saved_history = 0

for train_index, test_index in skfold.split(X, y):
    for batch in batch_size:
        for epoch in epochs:
            model = Sequential()
            model.add(Dense(6, input_dim=6, kernel_initializer='glorot_uniform', activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            optimizer = SGD(lr=0.1, momentum=0.8)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            history = model.fit(X[train_index], y[train_index], epochs=epoch, batch_size=batch, verbose=0)
            #print("Model " + str(index_model) + ": Train Loss: %s, Train Accuracy: %s" % (history.history['loss'], history.history['accuracy']))
            scores = model.evaluate(X[test_index], y[test_index], verbose=0)
            print("Model " + ' Batch: ' + str(batch) + ' Epoch: ' + str(epoch) + ": Test Loss: %.2f%%, Test Accuracy: %.2f%%" % (scores[0] * 100, scores[1] * 100))
            cvscores.append(scores[1] * 100)
            if scores[1]*100 > best_accuracy:
                best_batch = batch
                best_epoch = epoch
                best_accuracy = scores[1] * 100
                saved_history = history
            index_model += 1
print("Accuracy mean : %.2f%%, Standard deviation: (+/- %.2f%%), Model Variance: %.2f%%, Best Accuracy: %.2f%%, Batch: %i, Epoch: %i " % (np.mean(cvscores), np.std(cvscores), np.std(cvscores)**2, best_accuracy, best_epoch))
plt.plot(saved_history.history['loss'])
plt.plot(saved_history.history['accuracy'])
plt.title('model accuracy Model ' + str(index_model))
plt.ylabel('loss/accuracy')
plt.xlabel('epoch')
plt.legend(['loss', 'accuracy'], loc='upper left')
plt.show()
