import numpy as np
import random as r

input_neurons = 2
hidden_neurons = 2
output_neurons = 2

def init_parameters():
    r.seed(42)
    w1 = np.random.uniform(-0.5, 0.5, (input_neurons, hidden_neurons))  
    b1 = np.full((1, hidden_neurons), 0.5)    
    w2 = np.random.uniform(-0.5, 0.5, (hidden_neurons, output_neurons))  
    b2 = np.full((1, output_neurons), 0.7)  
    return w1, w2, b1, b2

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

def forwardstep(w1, w2, b1, b2, x):
    z1 = np.dot(x, w1) + b1
    F1 = tanh(z1)
    z2 = np.dot(F1, w2) + b2
    F2 = tanh(z2)
    return z1, F1, z2, F2

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def backward_step(w1, w2, b1, b2, x, y_true, y_pred, hidden_output, Z1, Z2, learning_rate=0.01):
    d_loss_d_y_pred = 2 * (y_pred - y_true) / y_true.size
    
    d_loss_d_Z2 = d_loss_d_y_pred * tanh_derivative(Z2)
    
    d_loss_d_w2 = np.dot(hidden_output.T, d_loss_d_Z2)
    d_loss_d_b2 = np.sum(d_loss_d_Z2, axis=0, keepdims=True)
    
    d_loss_d_A1 = np.dot(d_loss_d_Z2, w2.T)
    
    d_loss_d_Z1 = d_loss_d_A1 * tanh_derivative(Z1)
    
    d_loss_d_w1 = np.dot(x.T, d_loss_d_Z1)
    d_loss_d_b1 = np.sum(d_loss_d_Z1, axis=0, keepdims=True)
    
    w1 -= learning_rate * d_loss_d_w1
    b1 -= learning_rate * d_loss_d_b1
    w2 -= learning_rate * d_loss_d_w2
    b2 -= learning_rate * d_loss_d_b2
    
    return w1, w2, b1, b2

x = np.array([[0.5, 0.3], [0.2, 0.8]])
y_true = np.array([[0.1, 0.9], [0.8, 0.2]])

w1, w2, b1, b2 = init_parameters()

Z1, hidden_output, Z2, y_pred = forwardstep(w1, w2, b1, b2, x)

error = mean_squared_error(y_true, y_pred)

w1, w2, b1, b2 = backward_step(w1, w2, b1, b2, x, y_true, y_pred, hidden_output, Z1, Z2, learning_rate=0.01)

print("Input (X):\n", x)
print("Initial Weights (W1):\n", w1)
print("Initial Weights (W2):\n", w2)
print("Initial Bias (b1):\n", b1)
print("Initial Bias (b2):\n", b2)
print("Hidden Layer Output (A1):\n", hidden_output)
print("Output Layer Output (A2):\n", y_pred)
print("Mean Squared Error (MSE):\n", error)
print("Target (y_true):\n", y_true)
print("Updated Weights (W1):\n", w1)
print("Updated Weights (W2):\n", w2)
print("Updated Bias (b1):\n", b1)
print("Updated Bias (b2):\n", b2)
