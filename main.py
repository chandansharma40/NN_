import numpy as np
from test import DataFunc
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

enc = OneHotEncoder(n_values=10,sparse=False,dtype=np.int8)
scaler = StandardScaler()

data_loader = DataFunc()

trainImages = data_loader._load_img("train-images-idx3-ubyte.gz")
trainLabels = data_loader._load_label("train-labels-idx1-ubyte.gz")

#print(trainImages.shape)
#print(trainLabels.shape)

#trainLabels = enc.fit_transform(trainLabels.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(trainImages, trainLabels, test_size=0.33, shuffle=True, random_state=42)

#scaler.fit_transform(X_train)

#log_regression = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', verbose=1, max_iter=50, tol=1e-1)  # acc:92.19%
#log_regression = LogisticRegression(penalty='l2', solver='sag', multi_class='multinomial', verbose=1, max_iter=50, tol=1e-1)  # acc: 91.91%
#log_regression = LogisticRegression(penalty='l2', solver='sag', multi_class='ovr', verbose=1, max_iter=50, tol=1e-1)  # acc: 91.22%
log_regression = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', verbose=1, max_iter=50, tol=1e-1) # acc: 92.22%

log_regression.fit(X_train, y_train)

predict = log_regression.predict(X_test)

print(accuracy_score(y_test, predict))

predict = enc.fit_transform(predict.reshape(-1,1))

np.savetxt("lr.csv", predict, delimiter=",", fmt='%i')