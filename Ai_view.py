import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class Neural_Network():
    
    def init_params(inputs, outputs, hidden_layers_neurons, num_hidden_layers):
        W1 = np.random.rand(hidden_layers_neurons, inputs) - 0.5
        b1 = np.random.rand(hidden_layers_neurons, 1) - 0.5

        W_hidden = []
        b_hidden = []
        for _ in range(num_hidden_layers - 1):
            W_hidden.append(np.random.rand(hidden_layers_neurons, hidden_layers_neurons) - 0.5)
            b_hidden.append(np.random.rand(hidden_layers_neurons, 1) - 0.5)

        W2 = np.random.rand(outputs, hidden_layers_neurons) - 0.5
        b2 = np.random.rand(outputs, 1) - 0.5

        return W1, b1, W_hidden, b_hidden, W2, b2

    def ReLU(Z):
        return np.maximum(Z, 0)

    def softmax(Z):
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        return A

    def forward_prop(W1, b1, W_hidden, b_hidden, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = Neural_Network.ReLU(Z1)

        Ays = [A1]
        Zys = [Z1]

        # Forward through hidden layers
        for i in range(len(W_hidden)):
            Z = W_hidden[i].dot(Ays[-1]) + b_hidden[i]
            A = Neural_Network.ReLU(Z)
            Ays.append(A)
            Zys.append(Z)

        # Final layer (softmax output)
        Z2 = W2.dot(Ays[-1]) + b2
        A2 = Neural_Network.softmax(Z2)

        return Zys, Ays, Z2, A2

    def ReLU_deriv(Z):
        return Z > 0

    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(Zys, Ays, Z2, A2, W1, W_hidden, W2, X, Y):
        one_hot_Y = Neural_Network.one_hot(Y)
        
        # Output layer gradients
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / X.shape[1] * dZ2.dot(Ays[-1].T)
        db2 = 1 / X.shape[1] * np.sum(dZ2, axis=1, keepdims=True)

        # Gradients for hidden layers
        dW_hidden = []
        db_hidden = []
        dA = W2.T.dot(dZ2)
        
        for i in reversed(range(len(W_hidden))):
            dZ = dA * Neural_Network.ReLU_deriv(Zys[i+1])
            dW = 1 / X.shape[1] * dZ.dot(Ays[i].T)
            db = 1 / X.shape[1] * np.sum(dZ, axis=1, keepdims=True)
            dA = W_hidden[i].T.dot(dZ)
            
            dW_hidden.insert(0, dW)
            db_hidden.insert(0, db)

        # First layer gradients
        dZ1 = dA * Neural_Network.ReLU_deriv(Zys[0])
        dW1 = 1 / X.shape[1] * dZ1.dot(X.T)
        db1 = 1 / X.shape[1] * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW_hidden, db_hidden, dW2, db2


    def update_params(W1, b1, W_hidden, b_hidden, W2, b2, dW1, db1, dW_hidden, db_hidden, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1

        for i in range(len(W_hidden)):
            W_hidden[i] = W_hidden[i] - alpha * dW_hidden[i]
            b_hidden[i] = b_hidden[i] - alpha * db_hidden[i]

        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

        return W1, b1, W_hidden, b_hidden, W2, b2

    def get_predictions(A2):
        return np.argmax(A2, axis=0)

    def get_accuracy(predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def train_step(X, Y, W1, b1, W_hidden, b_hidden, W2, b2, alpha):
        Zys, Ays, Z2, A2 = Neural_Network.forward_prop(W1, b1, W_hidden, b_hidden, W2, b2, X)
        dW1, db1, dW_hidden, db_hidden, dW2, db2 = Neural_Network.backward_prop(Zys, Ays, Z2, A2, W1, W_hidden, W2, X, Y)
        W1, b1, W_hidden, b_hidden, W2, b2 = Neural_Network.update_params(W1, b1, W_hidden, b_hidden, W2, b2, dW1, db1, dW_hidden, db_hidden, dW2, db2, alpha)
        predictions = Neural_Network.get_predictions(A2)
        accuracy = Neural_Network.get_accuracy(predictions, Y)
        return W1, b1, W_hidden, b_hidden, W2, b2, accuracy


# Initialize global variables for storing points and neural network
points = []
labels = []
W1, b1, W_hidden, b_hidden, W2, b2 = Neural_Network.init_params(inputs=2, outputs=2, hidden_layers_neurons=25, num_hidden_layers=4)
learning_rate = 0.1

# Function to handle mouse click events
def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        # Capture point and its label (0 for left click, 1 for right click)
        x, y = event.xdata, event.ydata
        label = 0 if event.button == 1 else 1  # Left click = 0, Right click = 1
        points.append([x, y])
        labels.append(label)
        print(f"Point added: ({x}, {y}), label: {label}")

# Function to generate grid for predictions
def generate_grid(step=0.05):
    x = np.arange(0, 1, step)
    y = np.arange(0, 1, step)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid

# Function to update the plot and train the model
def update(frame):
    global W1, b1, W_hidden, b_hidden, W2, b2, learning_rate
    if len(points) > 1:
        # Convert points and labels to numpy arrays
        X = np.array(points).T  # Transpose for correct input format
        Y = np.array(labels)

        # Train the neural network with current points
        W1, b1, W_hidden, b_hidden, W2, b2, accuracy = Neural_Network.train_step(X, Y, W1, b1, W_hidden, b_hidden, W2, b2, learning_rate)

        # Generate predictions on a grid
        xx, yy, grid = generate_grid(step=0.01)
        grid_T = grid.T  # Transpose for correct input format
        _, _, _, grid_pred = Neural_Network.forward_prop(W1, b1, W_hidden, b_hidden, W2, b2, grid_T)
        Z = Neural_Network.get_predictions(grid_pred)
        Z = Z.reshape(xx.shape)

        # Clear the plot and replot the colormap
        plt.clf()
        plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')

        # Plot the points
        points_np = np.array(points)
        plt.scatter(points_np[:, 0], points_np[:, 1], c=labels, cmap='coolwarm', edgecolor='k')
        # Redraw the canvas
        plt.draw()

# Set up the plot
fig, ax = plt.subplots()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_title('Click to add points (left=0, right=1)')

# Connect the click event to the plot
cid = fig.canvas.mpl_connect('button_press_event', on_click)

# Animation to update the plot every second
ani = FuncAnimation(fig, update, interval=10)

# Show the plot
plt.show()
