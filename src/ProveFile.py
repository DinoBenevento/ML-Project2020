import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.KNN_GridSearchCV import knn_grid_search_cv
from src.SVM_GridSearchCV import svm_grid_search_cv
from src.Scaling import scaling_method
import seaborn as sns; sns.set()
sns.set_style("darkgrid")



monk1_train = pd.read_csv("../data/monks-1.train", delimiter=" ")

X = np.array((monk1_train.iloc[:, 1:7]))
y = np.array(monk1_train['result'])

X_train_scaled = scaling_method(X, "")
results = svm_grid_search_cv(X, y, 25)




