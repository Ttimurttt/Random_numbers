import numpy as np

class LSTM:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weight matrices for LSTM gates
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  # Forget gate
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  # Input gate
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  # Candidate hidden state
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  # Output gate

        # Biases for LSTM gates
        self.bf = np.zeros((hidden_size, 1))  # Forget gate bias
        self.bi = np.zeros((hidden_size, 1))  # Input gate bias
        self.bc = np.zeros((hidden_size, 1))  # Candidate state bias
        self.bo = np.zeros((hidden_size, 1))  # Output gate bias

        # Output layer weights and biases
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output
        self.by = np.zeros((output_size, 1))  # Output bias

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def tanh(self, Z):
        return np.tanh(Z)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward(self, X):
        h = np.zeros((self.hidden_size, 1))  # Initial hidden state
        c = np.zeros((self.hidden_size, 1))  # Initial cell state

        outputs = []
        hidden_states = []
        cell_states = []

        for t in range(X.shape[0]):
            x_t = X[t].reshape(-1, 1)  # Current time step input
            
            # Concatenate input and previous hidden state
            concat = np.vstack((h, x_t))

            # Forget gate
            f_t = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
            # Input gate
            i_t = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            # Candidate hidden state
            c_hat_t = self.tanh(np.dot(self.Wc, concat) + self.bc)
            # Cell state update
            c = f_t * c + i_t * c_hat_t
            # Output gate
            o_t = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
            # Hidden state update
            h = o_t * self.tanh(c)

            hidden_states.append(h)
            cell_states.append(c)

            # Output prediction
            Z = np.dot(self.Why, h) + self.by
            output = Z  # For non-text data, this could be raw output, like regression
            
            outputs.append(output)

        return outputs, hidden_states, cell_states

    def backward(self, X, Y, outputs, hidden_states, cell_states):
        dWf, dWi, dWc, dWo = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wc), np.zeros_like(self.Wo)
        dbf, dbi, dbc, dbo = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bc), np.zeros_like(self.bo)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros_like(hidden_states[0])
        dc_next = np.zeros_like(cell_states[0])

        for t in reversed(range(len(Y))):
            dZ = outputs[t] - Y[t].reshape(-1, 1)  # Loss gradient wrt output (for regression tasks)
            dWhy += np.dot(dZ, hidden_states[t].T)
            dby += dZ

            dh = np.dot(self.Why.T, dZ) + dh_next  # Gradient wrt hidden state
            do_t = dh * np.tanh(cell_states[t])
            do_t = do_t * hidden_states[t] * (1 - hidden_states[t])

            # Concatenate input and previous hidden state
            x_t = X[t].reshape(-1, 1)  # Current input
            h_prev = hidden_states[t-1] if t > 0 else np.zeros_like(hidden_states[0])  # Previous hidden state
            concat = np.vstack((h_prev, x_t))  # Concatenate previous hidden state and current input

            dWo += np.dot(do_t, concat.T)
            dbo += do_t

            # You would also need to backpropagate through the other gates here (input gate, forget gate, etc.)

        # Additional backpropagation logic here for other gates (dWf, dWi, dWc, etc.)

        return dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dWhy, dby
    
    def update_params(self, dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dWhy, dby, learning_rate):
        self.Wf -= learning_rate * dWf
        self.Wi -= learning_rate * dWi
        self.Wc -= learning_rate * dWc
        self.Wo -= learning_rate * dWo
        self.bf -= learning_rate * dbf
        self.bi -= learning_rate * dbi
        self.bc -= learning_rate * dbc
        self.bo -= learning_rate * dbo
        self.Why -= learning_rate * dWhy
        self.by -= learning_rate * dby
        
    def train(self, X, Y, learning_rate=0.01, iterations=1000):
        for i in range(iterations):
            outputs, hidden_states, cell_states = self.forward(X)
            dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dWhy, dby = self.backward(X, Y, outputs, hidden_states, cell_states)
            self.update_params(dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dWhy, dby, learning_rate)

            if i % 100 == 0:
                loss = self.mean_squared_loss(outputs, Y)
                print(f"Iteration {i}, Loss: {loss}")

    def mean_squared_loss(self, outputs, Y):
        loss = 0
        for t in range(len(Y)):
            loss += np.sum((outputs[t] - Y[t].reshape(-1, 1)) ** 2)
        return loss / len(Y)

# Stock prices for days 1 to 6
stock_prices = np.array([1, 0.2, 0.4, 0.7, 0.9, 1])

# Input (stock prices for day 1 to day 5) and target output (stock price for day 6)
X_train = stock_prices[:].reshape(-1, 1)  # Input: stock prices for day 1 to day 5
Y_train = stock_prices[:].reshape(-1, 1)   # Output: stock prices for day 2 to day 6

print("X_train:", X_train)
print("Y_train:", Y_train)

# Define the LSTM model
input_size = 1  # One feature (stock price) per time step
hidden_size = 50  # Hidden units in the LSTM
output_size = 1   # Predicting a single value (the stock price)

lstm = LSTM(input_size, hidden_size, output_size)

# Train the LSTM on the stock price data
lstm.train(X_train, Y_train, learning_rate=0.3, iterations=100000)

# Input for prediction: stock prices from day 1 to day 6
X_test = np.array([0, 0.2, 0.4, 0.7, 0.9, 0]).reshape(-1, 1)

# Use the LSTM to predict the stock price for day 6
outputs, hidden_states, cell_states = lstm.forward(X_test)

# Print the predicted output for day 7 (which is the last output in the sequence)
predicted_value_day_7 = outputs[-1]
print("Predicted stock price on day 7:", predicted_value_day_7)
