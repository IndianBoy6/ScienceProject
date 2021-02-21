import threading
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
import sys
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

m_max = 0
m_min = 0

running = True

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
                data = np.concatenate([np.zeros(self.n - len(data)), data])
            g.set_ydata(data)


    def main(self):
        while True:
            self.update_plot()
            plt.pause(1.0 / 30)
        running = False


def start_input():
    data = []
    data_takes = []

    for pinky in range(5):
        for ring in range(5):
            for middle in range(5):
                for pointer in range(5):
                    for _ in range(10):
                        finger_pos = [0, pointer, middle, ring, pinky]
                        data_takes.append(finger_pos)

    predicting = (sys.argv[1])
    if sys.argv[1].lower() != "true" and sys.argv[1].lower() != "false":
        print("The argument after python gather_data.py must be true or false")
    
    predicting = sys.argv == "true"

    model = None
    if predicting == True:
        try:
            model = keras.models.load_model("model")
        except IOError:
            print("If you have already trained the model, make sure the model folder exists in this directory. Otherwise, ignore this error")
            pass

    data = []

    positions = ["Down", "Bottom Angle", "Mid", "Top Angle", "Up"]

    completed_testing_gestures = 0
    while running:

        #	print(list(formatted_data["emg_avg"]))
        #	continue

        if predicting:
            #_ = input("enter: ")
            _ = input("")

            norm_data = []
            for i in list(formatted_data["emg_avg"]):
                norm_data.append((i / 64) - 1)

            pred = model.predict(np.array([norm_data]))[0]
            #print([1 if i > 0.5 else 0 for i in pred])
            print("[", end="")
            print(*pred, sep=",", end="")
            print("]")
            print(list(formatted_data["emg_avg"]))

            completed_testing_gestures += 1
            print("done", completed_testing_gestures)
            continue

        #take = random.choice(data_takes)
        # data_takes.remove(take)
        take = data_takes.pop(0)
        message = ""

        message += "Thumb: {}\n ".format(positions[take[0]])
        message += "Pointer: {}\n ".format(positions[take[1]])
        message += "Middle: {}\n ".format(positions[take[2]])
        message += "Ring: {}\n ".format(positions[take[3]])
        message += "Pinky: {}\n ".format(positions[take[4]])

        print(message)

        inp = input("(%i)>" % len(data_takes))

        data.append([take, list(formatted_data["orientation"]) +
                     list(formatted_data["emg_avg"])])

        print("\033[2J")

        with open("dataFINAL.json", "w") as file:
            json.dump(data, file)
        """
		GET TO 5250 TODAY!!!

		"""


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
