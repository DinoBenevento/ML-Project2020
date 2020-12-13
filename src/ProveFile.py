import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.KNN_GridSearchCV import knn_grid_search_cv
from src.SVM_GridSearchCV import svm_grid_search_cv
from src.NNBinaryClassification import create_model_gridsearchcv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from src.NNBinaryClassification import set_batch_epochs_NN_binary_
from keras.wrappers.scikit_learn import KerasClassifier
from src.PrintAndPlotResult import print_result_gridsearch


from src.Scaling import scaling_method
import seaborn as sns; sns.set()
sns.set_style("darkgrid")



monk1_train = pd.read_csv("../data/monks-1.train", delimiter=" ")
#dataframe = pd.read_csv("../data/sonar.all-data.csv", delimiter=',')
#dataset = dataframe.values
# split into input (X) and output (Y) variables
#X = dataset[:,0:60].astype(float)
#y = dataset[:,60]

#encoder = LabelEncoder()
#encoder.fit(y)
#encoded_Y = encoder.transform(y)

X = np.array((monk1_train.iloc[:, 1:7]))
y = np.array(monk1_train['result'])
#monk1_train.plot(kind='scatter', x='data_1', y='data_3', color='red')
#plt.show()

#dataframe.plot(kind='scatter', x='data_1', y='result', color='red')
#plt.show()

plt.figure(figsize=(16, 6))


monk1_dataframe = pd.read_csv("../data/monks-1.train", delimiter=" ")

heatmap = sns.heatmap(monk1_dataframe.corr(), vmin=-1, vmax=1,annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
#plt.show()

X_train_scaled = scaling_method(X, "MinMax")

knn_grid_search_cv(1,31, X, y, 5)






