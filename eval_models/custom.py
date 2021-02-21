import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics

data_x = []
data_y = []

str_data = [i.replace("\n", "") for i in open("FORMDAT", "r")]

for index, line in enumerate(str_data):
    data_x.append([float(i)  for i in line.strip("][").split(",")])
    data_y.append(index % 5)


sp = 25
data_x_train = data_x[:sp]
data_y_train = data_y[:sp]

data_x_test = data_x[sp:]
data_y_test = data_y[sp:]


model = KNeighborsClassifier(n_neighbors = 2)
model.fit(data_x_train, data_y_train)

predicted = model.predict(data_x_test)
acc = metrics.accuracy_score(data_y_test, predicted)
print(acc)

# 86%