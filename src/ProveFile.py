import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.KNN_GridSearchCV import knn_grid_search_cv
from src.SVM_GridSearchCV import svm_grid_search_cv
import seaborn as sns; sns.set()
sns.set_style("darkgrid")



monk1_train = pd.read_csv("../data/monks-1.train", delimiter=" ")

X = np.array((monk1_train.iloc[:, 1:7]))
y = np.array(monk1_train['result'])

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X)


results = svm_grid_search_cv(X_train_scaled, y, 10)
sns.scatterplot(data=results, x='C', y="gamma")
plt.show()




