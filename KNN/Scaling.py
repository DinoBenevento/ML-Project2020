
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def scaling_method(X, method):
    X_train_scaled = X
    if method == "Standard":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
    elif method == "MinMax":
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X)
    return X_train_scaled


