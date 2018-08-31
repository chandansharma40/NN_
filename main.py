from test import DataFunc
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

enc = OneHotEncoder(n_values=10,sparse=False)

data_loader = DataFunc()

trainImages = data_loader._load_img("train-images-idx3-ubyte.gz")
trainLabels = data_loader._load_label("train-labels-idx1-ubyte.gz")

print(trainImages.shape)
print(trainLabels.shape)

trainLabels = enc.fit_transform(trainLabels.reshape(-1,1))