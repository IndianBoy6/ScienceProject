import tensorflow.keras as keras
import json
import random
import numpy as np
import pickle as pkl

from tensorflow.python.keras.layers.core import Dense


data = None
with open("data.json", "r") as file:
    data = json.load(file)


data_x = []
data_y = []


def normalize(r):
    return (r / 64) - 1

for i in range(len(data)):
    for j in range(100):
        r = random.uniform(-1.0, 1.0)
        r = 0

        data_x.append([normalize(p) + r for p in data[i][1][4:]])
        data_y.append(data[i][0])


data_x_rand = []
data_y_rand = []

while len(data_x) != 0:
    rand = random.randrange(0, len(data_x))
    x = data_x.pop(rand)
    y = data_y.pop(rand)

    data_x_rand.append(x)
    data_y_rand.append(y)




# TODO try just getting more data, that might be the problem

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


model.fit(data_x_rand, data_y_rand, epochs=100)

model.save("model")
