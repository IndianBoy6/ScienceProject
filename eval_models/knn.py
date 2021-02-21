import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics

data_x = []
data_y = []

str_data = [i.replace("\n", "") for i in open("RAWDAT", "r")]

for index, line in enumerate(str_data):
    data_x.append([float(i)  for i in line.strip("][").split(",")])
    data_y.append(index % 5)

data_x_train = data_x[:25]
data_y_train = data_y[:25]

data_x_test = data_x[25:]
data_y_test = data_y[25:]


model = KNeighborsClassifier(n_neighbors = 5)
model.fit(data_x_train, data_y_train)

predicted = model.predict(data_x_test)
acc = metrics.accuracy_score(data_y_test, predicted)
print(acc)

# 67%