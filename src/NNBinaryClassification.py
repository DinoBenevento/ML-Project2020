
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import number
from sklearn.model_selection import GridSearchCV
from src.PrintAndPlotResult import print_result_gridsearch

def  set_model_binary_classification(numbers_layers, input_dim, neurons, init_mode, activation_function, optimizer, learn_rate, momentum,
                                    ifDroput, dropout_rate):
    model = Sequential()
    layer = 1
    while layer <= numbers_layers:
        if layer == 1:
            model.add(Dense(neurons, kernel_initializer=init_mode, input_dim=input_dim, activation=activation_function))
        if ifDroput:
            model.add(Dropout(dropout_rate))
        elif layer > 1:
            model.add(Dense(neurons, kernel_initializer=init_mode, activation=activation_function))
        layer += 1
    model.add(Dense(1, activation='sigmoid'))
    if optimizer is "SGD":
        optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def set_batch_epochs_NN_binary_():

    model = Sequential()
    model.add(Dense(6, input_dim=6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




def create_model_gridsearchcv(numbers_layers, input_dim, neurons, init_mode, activation_function, optimizer, learn_rate, momentum,
                                    ifDroput, dropout_rate, X, y):

    model = KerasClassifier(build_fn=set_model_binary_classification(numbers_layers, input_dim, neurons, init_mode, activation_function, optimizer, learn_rate, momentum,
                                    ifDroput, dropout_rate), epochs=100, batch_size=10, verbose=0)
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    activation = ['softplus', 'softsign', 'relu', 'tanh', 'linear']
    param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, learn_rate=learn_rate, momentum=momentum,
                      init_mode=init_mode, activation=activation)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid.fit(X, y)
    print_result_gridsearch(grid)

    return grid.cv_results_