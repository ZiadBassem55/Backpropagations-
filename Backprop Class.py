import numpy as np
import random as r

class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.w1, self.w2, self.b1, self.b2 = self.init_parameters()

    def init_parameters(self):
        r.seed(42)
        w1 = np.random.uniform(-0.5, 0.5, (self.input_neurons, self.hidden_neurons))
        b1 = 0.5
        w2 = np.random.uniform(-0.5, 0.5, (self.hidden_neurons, self.output_neurons))
        b2 = 0.7
        return w1, w2, b1, b2

    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def tanh_derivative(self, z):
        return 1 - self.tanh(z) ** 2

    def forward_step(self, x):
        self.z1 = np.dot(self.w1, x) + self.b1
        self.F1 = self.tanh(self.z1)
        self.z2 = np.dot(self.w2, self.F1) + self.b2
        self.F2 = self.tanh(self.z2)
        return self.F2, self.F1, self.z1, self.z2

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward_step(self, x, y_true, y_pred, hidden_output, learning_rate=0.01):
        d_loss_d_y_pred = 2 * (y_pred - y_true) / y_true.size
        d_loss_d_Z2 = d_loss_d_y_pred * self.tanh_derivative(self.z2)
        d_loss_d_w2 = np.dot(hidden_output.T, d_loss_d_Z2)
        d_loss_d_b2 = np.sum(d_loss_d_Z2, axis=0, keepdims=True)
        d_loss_d_A1 = np.dot(d_loss_d_Z2, self.w2.T)
        d_loss_d_Z1 = d_loss_d_A1 * self.tanh_derivative(self.z1)
        d_loss_d_w1 = np.dot(x.T, d_loss_d_Z1)
        d_loss_d_b1 = np.sum(d_loss_d_Z1, axis=0, keepdims=True)

        self.w1 -= learning_rate * d_loss_d_w1
        self.b1 -= learning_rate * d_loss_d_b1
        self.w2 -= learning_rate * d_loss_d_w2
        self.b2 -= learning_rate * d_loss_d_b2

    def train(self, x, y_true, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            y_pred, hidden_output, z1, z2 = self.forward_step(x)
            error = self.mean_squared_error(y_true, y_pred)
            self.backward_step(x, y_true, y_pred, hidden_output, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {error}")

    def predict(self, x):
        y_pred, _, _, _ = self.forward_step(x)
        return y_pred


# Example usage
input_neurons = 2
hidden_neurons = 2
output_neurons = 2

x = np.array([[0.5, 0.3], [0.2, 0.8]])
y_true = np.array([[0.1, 0.9], [0.8, 0.2]])

nn = NeuralNetwork(input_neurons, hidden_neurons, output_neurons)
nn.train(x, y_true, epochs=1000, learning_rate=0.01)
y_pred = nn.predict(x)
print("Predicted Output:\n", y_pred)
