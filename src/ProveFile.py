import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


monk1_train = pd.read_csv("../data/monks-1.train", delimiter=" ")

X = np.array((monk1_train.iloc[:, 1:8]))
y = np.array(monk1_train['result'])

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X)

