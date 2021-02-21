import tensorflow.keras as keras
import random
import numpy as np
from tensorflow.python.keras.backend import one_hot
import json
import random
import pickle as pkl


def normalize(r):
    return (r / 64) - 1

data_x = []
data_y = []

str_data = [i.replace("\n", "") for i in open("RAWDAT", "r")]

for index, line in enumerate(str_data):
    data_x.append(np.array([normalize(float(i)) for i in line.strip("][").split(",")]))
    data_y.append(one_hot(index % 5, 5))

data_x_norm = data_x
data_y_norm = data_y

data_x = []
data_y = []

for index in range(len(data_x_norm)):
    rnum = random.randrange(0, len(data_x_norm))
    data_x.append(data_x_norm.pop(rnum))
    data_y.append(data_y_norm.pop(rnum))

data_x_train = data_x[:25]
data_y_train = data_y[:25]

data_x_test = data_x[25:]
data_y_test = data_y[25:]



model = keras.Sequential([
    keras.layers.Dense(8),
    keras.layers.Dense(16, activation = "sigmoid"),
    keras.layers.Dense(32, activation = "sigmoid"),
    keras.layers.Dense(64, activation = "sigmoid"),
    keras.layers.Dense(128, activation = "sigmoid"),
    keras.layers.Dense(75, activation = "sigmoid"),
    keras.layers.Dense(20, activation = "sigmoid"),
    keras.layers.Dense(10, activation = "sigmoid"),
    keras.layers.Dense(5)
])

model.compile(
    optimizer="adam",
    loss = keras.losses.MeanSquaredError(),
    metrics = ["accuracy"]
)


model.fit(np.array(data_x_train), np.array(data_y_train), epochs=100)

acc = model.evaluate(np.array(data_x_test), np.array(data_y_test))
print(str(acc[1]))

# 18%