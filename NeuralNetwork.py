import numpy as np 
import tensorflow as tf
from PIL import Image
np.random.seed(89)
import tensorflow as tf

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten the images to (num_samples, 784)
X_train = X_train.reshape(X_train.shape[0], -1).T  # 28x28 -> 784
X_test = X_test.reshape(X_test.shape[0], -1).T

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10).T
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10).T

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

def init_params():
    W1 = np.random.randn(10, 784) * np.sqrt(1 / 784)
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * np.sqrt(1 / 10)
    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0,Z)

def deriv_ReLU(Z):
    return Z > 0

def sigmoid(Z):
    return 1 / (1+np.exp(-Z))

def deriv_sigmoid(Z):
    return sigmoid(Z)*(1-sigmoid(Z))


def softmax(Z):
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X)+b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = softmax(Z2) 
    return Z1, A1, Z2, A2


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.shape[1]
    one_hot_Y = Y
    dZ2 = A2-one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis = 1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * deriv_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1= (1 / m) * np.sum(dZ1, axis = 1, keepdims = True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    return W1, b1, W2, b2

def get_accuracy(predictions, Y):
    pred_indices = np.argmax(predictions, axis=0)
    true_indices = np.argmax(Y, axis=0)
    correct_predictions = np.sum(pred_indices == true_indices)
    accuracy = correct_predictions / Y.shape[1]
    print(predictions, get_predictions(Y))
    return accuracy

def get_predictions(A2):
    max_indices = np.argmax(A2, axis=0)
    return max_indices

def get_accuracy(A2, y):
    accuracy = np.mean(A2 == y)
    return accuracy

def cost(y_hat, y, batch_size):
    losses = - ((y*np.log(y_hat) + (1-y)*np.log(1-y_hat)))
    summed_losses = (1 / batch_size) * np.sum(losses, axis = 0)
    return np.sum(summed_losses)

def gradient_descent(X, y, iterations, learning_rate):
    
    W1, b1, W2, b2 = init_params()
    batch_size = 30
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X[:,i*batch_size:i*batch_size+batch_size])
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X[:,i*batch_size:i*batch_size+batch_size], y[:,i*batch_size:i*batch_size+batch_size])
        W1, b1, W2, b2= update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if i % 100 == 0:
            print("Iteration: ", i)
            print("Accuracy:", get_accuracy(get_predictions(A2), get_predictions(y[:,i*batch_size:i*batch_size+batch_size])))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, y_train, 2000, 0.04)


    