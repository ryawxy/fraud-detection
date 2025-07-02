import numpy as np

from clustering import KMeansClustering
from driver import Driver
from hmm_model import HMM

from config.config import *


def get_input():
    while True:
        new_transaction = input('Please add your new transaction : ')
        if int(new_transaction) == TERMINATE:
            break
        new_transaction = k.predict(int(new_transaction))
        new_observation = np.append(observations[1:], [new_transaction])

        if h.detect_fraud(observations, new_observation, THRESHOLD):
            print('Fraud')
        else:
            print('Normal')


if __name__ == '__main__':
    d = Driver('/Users/raya/Desktop/fraud-detection/simulated-dataset/hidden_markov_model/data/train_data.txt')

    h = HMM(n_states=STATES, n_possible_observations=CLUSTERS)
    k = KMeansClustering()

    observations = k.run(d.get_data()[0:192])
    h.train_model(observations=list(observations), steps=STEPS)

    get_input()