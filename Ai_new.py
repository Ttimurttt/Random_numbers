import numpy as np

Data = np.sum(np.random.randint(1, 10))
Answer = np.sum(Data)

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
        A = np.exp(Z) / sum(np.exp(Z))
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
        return np.argmax(A2, 0)

    def get_accuracy(predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(X, Y, learning_rate, iterations, hidden_neurons, num_hidden_layers, output):
        inputs = X.shape[0]
        outputs = output

        W1, b1, W_hidden, b_hidden, W2, b2 = Neural_Network.init_params(inputs, outputs, hidden_neurons, num_hidden_layers)

        for i in range(iterations):
            Zys, Ays, Z2, A2 = Neural_Network.forward_prop(W1, b1, W_hidden, b_hidden, W2, b2, X)
            dW1, db1, dW_hidden, db_hidden, dW2, db2 = Neural_Network.backward_prop(Zys, Ays, Z2, A2, W1, W_hidden, W2, X, Y)
            W1, b1, W_hidden, b_hidden, W2, b2 = Neural_Network.update_params(W1, b1, W_hidden, b_hidden, W2, b2, dW1, db1, dW_hidden, db_hidden, dW2, db2, learning_rate)

            if i % 10 == 0:
                predictions = Neural_Network.get_predictions(A2)
                accuracy = Neural_Network.get_accuracy(predictions, Y)
                print(f"Iteration {i}, Accuracy: {accuracy:.4f}")

        return W1, b1, W_hidden, b_hidden, W2, b2

# Test with a small number of iterations
W1, b1, W_hidden, b_hidden, W2, b2 = Neural_Network.gradient_descent(Data, Answer, 0.1, 1000, hidden_neurons=50, num_hidden_layers=1, output=1)
