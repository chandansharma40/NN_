import numpy as np
from test import DataFunc
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

enc = OneHotEncoder(n_values=10,sparse=False,dtype=np.int8)
scaler = StandardScaler()

data_loader = DataFunc()

trainImages = data_loader._load_img("train-images-idx3-ubyte.gz")
trainLabels = data_loader._load_label("train-labels-idx1-ubyte.gz")
testImages = data_loader._load_img("t10k-images-idx3-ubyte.gz")
testLabels = data_loader._load_label("t10k-labels-idx1-ubyte.gz")

arr = np.append(trainImages, trainLabels[:,None], axis=1)
np.random.shuffle(arr)
trainImages = arr[:,0:784]
trainLabels = arr[:,784]

'''
Logistic Regression
'''

#log_regression = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', verbose=1, max_iter=50, tol=1e-1)  # acc:92.19%
#log_regression = LogisticRegression(penalty='l2', solver='sag', multi_class='multinomial', verbose=1, max_iter=50, tol=1e-1)  # acc: 91.91%
#log_regression = LogisticRegression(penalty='l2', solver='sag', multi_class='ovr', verbose=1, max_iter=50, tol=2e-2)  # acc: 91.22%
log_regression = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', verbose=1, max_iter=50, tol=1e-1) # acc: 92.63%
log_regression.fit(trainImages, trainLabels)
predict = log_regression.predict(testImages)
print(accuracy_score(testLabels, predict))
predict = enc.fit_transform(predict.reshape(-1,1))
np.savetxt("lr.csv", predict, delimiter=",", fmt='%i')

'''
Random Forest Regression
'''
random_regression = RandomForestClassifier(verbose=1) # acc: 94.76%
random_regression.fit(trainImages, trainLabels)
predict_rf = random_regression.predict(testImages)
print(accuracy_score(testLabels, predict_rf))
predict_rf = enc.fit_transform(predict_rf.reshape(-1,1))
np.savetxt("rf.csv", predict_rf, delimiter=",", fmt='%i')