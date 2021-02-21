import threading
import math
import random
from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import myo
import json
import numpy as np
import warnings
from operator import add
import pickle as pkl
import tensorflow.keras as keras
from your_data import *

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
        global m_max
        global m_min

        with self.lock:
            for i in event.emg:
                if i > m_max:
                    m_max = i
                elif i < m_min:
                    m_min = i
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

    model = keras.models.load_model("model.2")

    data = []

    positions = ["Down", "Bottom Angle", "Mid", "Top Angle", "Up"]

    while True:

        norm_data = []
        for i in list(formatted_data["emg_avg"]):
            norm_data.append((i / 64) - 1)

        # + formatted_data["orientation"]
        pred = model.predict(np.array([norm_data]))[0]

        gest_one = [0.10663057, 0.55458367, 1.0739942,  0.10519223, 0.40350884]
        gest_two = [0.09234684, 0.73495364, 1.0668597,  0.12956208, 0.45013475]



        avg_distances = []
        for gesture in gestures:
            avg_distances.append(
                sum([n_dist(5, pred, i) for i in gesture]) / len(gestures)
            )

        least = np.argmin(avg_distances)

        _ = input("Press Enter To Evaluate The Current Gesture")
        print(str(gesture_names[least]))

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
