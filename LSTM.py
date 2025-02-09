import numpy as np
import matplotlib.pyplot as plt
from lib.ltsm import LSTM

# Initialize model
model = LSTM(lr=0.1)

# Sample inputs and labels
true_lab = np.reshape(np.arange(0, 1.1, 0.1), (11, 1))
false_lab = np.reshape([.7, .6, .4]*11, (11, 3))
input_data = np.append(true_lab, false_lab, axis = 1)
labels = np.array(np.arange(0, 1.1, 0.1))

outputs = []
losses = []

# Train the model
for epoch in range(250000): 
    i = np.random.randint(0, np.shape(input_data)[0])
    batch = (input_data[i], labels[i])
    loss, output_i = model.training_step(batch)
    losses.append(loss)
    outputs.append(output_i)

# Plotting the losses
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(np.arange(len(losses)), losses, s = 0.1)
plt.title('Losses over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plotting the outputs
plt.subplot(1, 2, 2)
plt.scatter(np.arange(len(outputs)), outputs, s = 0.1)
plt.title('Outputs over epochs')
plt.xlabel('Epoch')
plt.ylabel('Output')
plt.tight_layout()
plt.show()

# Test the model
print("Final output for input [1., 0.5, 0.25, 1.]:", model.forward_mult([1., 0.5, 0.25, 1.]))
print("Final output for input [0., 0.5, 0.25, 1.]:", model.forward_mult([0., 0.5, 0.25, 1.]))
