import random

class Perceptron(object):
    def __init__(self, samples, outputs, learning_rate=0.05, seasons=1000, limiar=-1):
        self.samples = samples
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.seasons = seasons
        self.limiar = limiar
        self.n_samples = len(samples)
        self.n_features = len(samples[0])
        self.n_seasons = 0
        self.weights = []

    def train(self):
        # add weight bias
        for sample in self.samples:
            sample.insert(0, -1)

        # initializer random weights
        self.initialize_weights()

        while True:
            error = False

            for sample in range(self.n_samples):
                # aggregator
                u = 0
                for feature in range(self.n_features + 1):
                    u += self.weights[feature] * self.samples[sample][feature]

                # activation function
                y = self.activation_function(u)

                # weights adjustment
                if y != self.outputs[sample]:
                    for i in range(self.n_features + 1):
                        self.weights[i] = self.weights[i] + self.learning_rate * (self.outputs[sample] - y) * self.samples[sample][i]
                    error = True

            # count seasons
            self.n_seasons += 1

            # exit
            if not error or self.n_seasons > self.seasons:
                break

            print("weights: {0}".format(self.weights))
        print("seasons: {0}".format(self.n_seasons))

    def predict(self, sample):
        sample.insert(0, -1)
        u = 0
        for i in range(self.n_features + 1):
            u += self.weights[i] * sample[i]

        y = self.activation_function(u)
        return y

    def initialize_weights(self):
        for weight in range(self.n_features):
            self.weights.append(random.random())

        self.weights.insert(0, self.limiar)

    # degrau function
    def activation_function(self, u):
        if u >= 0:
            return 1
        return 0