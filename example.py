import json
import math
import pickle as pkl
import random
import threading
import warnings
from collections import deque
from operator import add
from threading import Lock, Thread

import myo
import numpy as np
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")



def get_movement(buffer):
    sum = 0
    count = 0
    for i in buffer:
        sum += i if i > 0 else -i
        count += 1
    return "%.3f" % (sum / count)


formatted_data = {
    "orientation": (0, 0, 0),
    "emg_avg": [0, 0, 0, 0, 0, 0, 0, 0]
}


class EmgCollector(myo.DeviceListener):
    """
    Collects EMG data in a queue with *n* maximum number of elements.
    """

    def __init__(self, n):
        self.queue_frame = False
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_orientation(self, event):
        formatted_data["orientation"] = event.orientation

    def on_emg(self, event):

        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))


class Plot(object):

    def __init__(self, listener):
        self.n = listener.n
        self.listener = listener
        self.fig, self.ax = plt.subplots()
        self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
        [(ax.set_ylim([-100, 100])) for ax in self.axes]
        self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[
            0] for ax in self.axes]

        self.texts = [plt.text(540, 1680 - i * 240, "") for i in range(8)]

        plt.ion()

    def update_plot(self):
        emg_data = self.listener.get_emg_data()
        emg_data = np.array([x[1] for x in emg_data]).T
        print_data = []

        for g, data, i in zip(self.graphs, emg_data, range(8)):

            movement = get_movement(data)
            self.texts[i].set_text(movement)
            print_data.append(movement)
            formatted_data["emg_avg"][i] = float(movement)

            if len(data) < self.n:
               # Fill the left side with zeroes.
                data = np.concatenate([np.zeros(self.n - len(data)), data])
            g.set_ydata(data)


    def main(self):
        while True:
            self.update_plot()
            plt.pause(1.0 / 30)


def n_dist(dims, a, b):
    output_sqrd = 0.0
    for i in range(len(a)):
        output_sqrd += abs(a[i] - b[i])

    return math.sqrt(output_sqrd)


def start_input():

    data_x = []
    data_y = []

    str_data = [i.replace("\n", "") for i in open("FORMDAT", "r")]

    for index, line in enumerate(str_data):
        data_x.append([float(i)  for i in line.strip("][").split(",")])
        data_y.append(index % 5)


    knnmodel = KNeighborsClassifier(n_neighbors = 2)
    knnmodel.fit(data_x, data_y)

    model = keras.models.load_model("model")





    while True:
        _ = input("Press Enter To Evaluate The Current Gesture")

        norm_data = []
        for i in list(formatted_data["emg_avg"]):
            norm_data.append((i / 64) - 1)
        

        first_pred = model.predict(np.array([norm_data]))[0]
        pred = knnmodel.predict(np.array([first_pred]))


        print(pred)

        continue


def main():
    myo.init(sdk_path=".")
    hub = myo.Hub()
    listener = EmgCollector(100)
    with hub.run_in_background(listener.on_event):
        Plot(listener).main()


if __name__ == '__main__':
    bg = threading.Thread(target=start_input)
    bg.start()
    main()
