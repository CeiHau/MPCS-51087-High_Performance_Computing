import random
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)

class NeuralNetwork:
    def __init__(self, sizes):
        # an array represents the number of nodes in each layer
        self.sizes = sizes
        self.layers_num = len(sizes) # L

        # initialize the biases and weights
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]] # Length = L - 1
        self.weights = [np.random.randn(j, i) for i, j in zip(sizes[:-1], sizes[1:])] # Length = l - 1

    def predict(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def feedforword(self, x):
        a = x
        self.a_L = [x]
        self.z_L = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            self.z_L.append(z)
            a = sigmoid(z)
            self.a_L.append(a)

    def backpropogate(self, y):
        # Initiate derv_b, derv_w
        derv_b = [np.zeros(b.shape) for b in self.biases]
        derv_w = [np.zeros(w.shape) for w in self.weights]

        # Output error: δx, L = ∇aC ⊙ σ′(zx, l)
        delta = (self.a_L[-1] - y) * sigmoid_derivative(self.z_L[-1])
        derv_b[-1] = delta
        derv_w[-1] = np.dot(delta, self.a_L[-2].T)

        for l in range(self.layers_num - 3, -1, -1):
            z = self.z_L[l]
            #  δx,l = ((wl+1)T δx,l+1) ⊙ σ′(zl)
            delta = np.dot(self.weights[l + 1].T, delta) * sigmoid_derivative(z)  # change to .T?

            derv_b[l] = delta  # Equation (14)
            derv_w[l] = np.dot(delta, self.a_L[l].T)  # Equation (15)

        return derv_b, derv_w

    def train_small_batch(self, batch, alpha):
        m = len(batch)  # size of batch
        # Initiate derv_b, derv_w
        sum_derv_b = [np.zeros(b.shape) for b in self.biases]
        sum_derv_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            self.feedforword(x)

            # Backpropogate
            derv_b, derv_w = self.backpropogate(y)

            # Caculate the changing of weights and bias
            sum_derv_b = [s_b + b for s_b, b in zip(sum_derv_b, derv_b)]
            sum_derv_w = [s_w + w for s_w, w in zip(sum_derv_w, derv_w)]

        # Gradient Descent
        self.weights = [w - (alpha / m) * d_w for w, d_w in zip(self.weights, sum_derv_w)]
        self.biases = [b - (alpha / m) * d_b for b, d_b in zip(self.biases, sum_derv_b)]

    def train(self, training_data, epochs, batch_size, alpha, test_data):
        for epoch in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]
            for batch in batches:
                self.train_small_batch(batch, alpha)
            num = self.evaluate(test_data)
            print(f"Epoch {epoch}, accuracy = {num/len(test_data)}")

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.predict(x)), y)
                        for (x, y) in test_data]
        return sum(int(y[x]) for (x, y) in test_results)


