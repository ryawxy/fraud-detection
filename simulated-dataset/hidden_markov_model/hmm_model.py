import numpy as np
from hidden_markov import hmm

def divide(num, denom):
    if num == 0:
        return 0

    return num / denom


class HMM:
    def __init__(self, n_states, n_possible_observations):
        self.n_states = n_states
        self.n_possible_observations = n_possible_observations
        self.states, self.possible_observations = self.__init_names()
        self.pi_prob, self.transition_prob, self.emission_prob = self.__init_probabilities()

        self.__model = hmm(states=list(self.states),
                           observations=list(self.possible_observations),
                           start_prob=np.matrix(self.pi_prob),
                           trans_prob=np.matrix(self.transition_prob),
                           em_prob=np.matrix(self.emission_prob))

    def __init_names(self):
        states = np.array(range(self.n_states))
        possible_observations = np.array(range(self.n_possible_observations))
        return states, possible_observations

    def __init_probabilities(self):
        pi_prob = np.zeros(self.n_states)
        transition_prob = np.zeros((self.n_states, self.n_states))
        emission_prob = np.zeros((self.n_states, self.n_possible_observations))

        for i in range(self.n_states):
            pi_prob[i] = 1 / self.n_states

        for i in range(self.n_states):
            for j in range(self.n_states):
                transition_prob[i][j] = 1 / self.n_states

        for i in range(self.n_states):
            for j in range(self.n_possible_observations):
                emission_prob[i][j] = 1 / self.n_possible_observations

        return pi_prob, transition_prob, emission_prob

    def train_model(self, observations, steps):
        print('HMM is training ...')
        pi_prob = np.zeros(self.n_states)
        transition_prob = np.zeros((self.n_states, self.n_states))
        emission_prob = np.zeros((self.n_states, self.n_possible_observations))

        for _ in range(steps):
            fwd = self.forward_process(observations)
            bwd = self.backward_process(observations)

            for i in range(self.n_states):
                pi_prob[i] = self.calculate_gamma(i, 0, fwd, bwd)

            for i in range(self.n_states):
                for j in range(self.n_states):
                    num, denom = 0, 0
                    for t in range(len(observations)):
                        num += self.calculate_path_probability(t, i, j, observations, fwd, bwd)
                        denom += self.calculate_gamma(i, t, fwd, bwd)

                    transition_prob[i][j] = divide(num, denom)

            for i in range(self.n_states):
                for k in range(self.n_possible_observations):
                    num, denom = 0, 0
                    for t in range(len(observations)):
                        g = self.calculate_gamma(i, t, fwd, bwd)
                        if k == observations[t]:
                            num += g
                        denom += g

                    emission_prob[i][k] = divide(num, denom)

        self.pi_prob = pi_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        print('HMM has successfully trained.')

    def forward_process(self, observations):
        fwd = np.zeros((self.n_states, len(observations)))
        for i in range(self.n_states):
            fwd[i][0] = self.pi_prob[i] * self.emission_prob[i][observations[0]]

        for t in range(len(observations) - 1):
            for j in range(self.n_states):
                fwd[j][t + 1] = 0
                for i in range(self.n_states):
                    fwd[j][t + 1] += (fwd[i][t] * self.transition_prob[i][j])

                fwd[j][t + 1] = (fwd[j][t + 1] * self.emission_prob[j][observations[t + 1]])

        return fwd

    def backward_process(self, observations):
        bwd = np.zeros((self.n_states, len(observations)))

        for i in range(self.n_states):
            bwd[i][len(observations) - 1] = 1

        for t in range(len(observations) - 2, -1, -1):
            for i in range(self.n_states):
                bwd[i][t] = 0
                for j in range(self.n_states):
                    bwd[i][t] += (
                            bwd[j][t + 1] * self.transition_prob[i][j] * self.emission_prob[j][observations[t + 1]])

        return bwd

    def calculate_gamma(self, cur_state, t, fwd, bwd):
        num = fwd[cur_state][t] * bwd[cur_state][t]
        denom = 0
        for i in range(self.n_states):
            denom += (fwd[i][t] * bwd[i][t])

        return divide(num, denom)

    def calculate_path_probability(self, t, i, j, observations, fwd, bwd):
        num, denom = 0, 0
        if t == len(observations) - 1:
            num = fwd[i][t] * self.transition_prob[i][j]
        else:
            num = fwd[i][t] * self.transition_prob[i][j] * self.emission_prob[j][observations[t + 1]] * bwd[j][t + 1]

        for k in range(self.n_states):
            denom += (fwd[k][t] * bwd[k][t])

        return divide(num, denom)

    def calculate_occurrence_probability(self, observations):
        fwd = self.forward_process(observations)
        bwd = self.backward_process(observations)
        result = np.zeros(len(observations))

        for i in range(len(observations)):
            for j in range(self.n_states):
                result[i] += fwd[j][i] * bwd[j][i]

        return result

    def detect_fraud(self, observations, new_observation, threshold):
        print('Fraud evaluation ...')
        alpha_1 = self.calculate_occurrence_probability(observations)
        alpha_2 = self.calculate_occurrence_probability(new_observation)
        delta = alpha_1[0] - alpha_2[0]
        delta = delta / alpha_1[0]

        if delta > threshold:
            return True
        else:
            return False