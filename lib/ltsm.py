import numpy as np


class LSTM:
    
    def __init__(self, lr, it_count, Long_term=0, Short_term=0):
        # Initialize weights and biases
        self.Iws = np.random.normal(0, 0.1, 4)  # Input weights
        self.Sws = np.random.normal(0, 0.1, 4)  # Short-term weights
        self.Bs = np.random.normal(0, 0.1, 4)   # Biases
        self.Long_term = Long_term  # Long-term memory
        self.Short_term = Short_term  # Short-term memory
        self.lr = lr  # Learning rate
        self.x = 0  # Training step counter
        self.it_count = it_count
        
    def forget_gate(self, Input):
        self.sum_f = summed = Input * self.Iws[0] + self.Short_term * self.Sws[0] + self.Bs[0]
        percentage_to_leave = self.sigmoid(summed)
        self.Long_term *= percentage_to_leave  # Update long-term memory
    
    def input_gate(self, Input):
        self.sum_i = summed1 = Input * self.Iws[1] + self.Short_term * self.Sws[1] + self.Bs[1]
        self.i_percent = self.sigmoid(summed1)  # Input gate activation
        
        self.sum_g = summed2 = Input * self.Iws[2] + self.Short_term * self.Sws[2] + self.Bs[2]
        self.g_add = np.tanh(summed2)  # New information
        
        self.Long_term += self.g_add * self.i_percent  # Update long-term memory
        
    def output_gate(self, Input):
        Potential_short_term = np.tanh(self.Long_term)  # Memory to output
        self.sum_o = summed = Input * self.Iws[3] + self.Short_term * self.Sws[3] + self.Bs[3]
        self.out_percentage = self.sigmoid(summed)  # Output gate activation
        return Potential_short_term * self.out_percentage  # Final output
    
    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    def forward(self, Input):
        # Perform forward pass
        self.Long_term = 0
        self.Short_term = 0
        self.forget_gate(Input)
        self.input_gate(Input)
        self.Short_term = self.output_gate(Input)
        return self.Short_term
    
    def forward_mult(self, Input):
        self.Long_term = 0
        self.Short_term = 0
        for num in Input:
            self.forget_gate(num)
            self.input_gate(num)
            self.Short_term = self.output_gate(num)
        return self.Short_term

    def training_step(self, batch):
        input_seq, label_i = batch  # input_seq is a sequence
        output_i = self.forward_mult(input_seq)
        loss = (output_i - label_i) ** 2  # Squared loss
        
        # Calculate gradient of loss
        d_loss = 2 * (output_i - label_i)
        
        # Perform gradient descent
        self.gradient_descent(d_loss, input_seq)
        
        if self.x % 10000 == 0:
            print(f"Train loss: {loss}, True output: {label_i}, Output: {output_i}, {self.x} / {self.it_count}")
        self.x += 1
        return loss, output_i
    
    def gradient_descent(self, d_loss, input):
        # Compute gradients and update weights
        for num in reversed(input):
            # Output gate gradients
            dE_dw_ino = d_loss * np.tanh(self.Long_term) * self.sigmoid(self.sum_o) * (1 - self.sigmoid(self.sum_o)) * num
            dE_dw_sto = d_loss * np.tanh(self.Long_term) * self.sigmoid(self.sum_o) * (1 - self.sigmoid(self.sum_o)) * self.Short_term
            dE_db_o = d_loss * np.tanh(self.Long_term) * self.sigmoid(self.sum_o) * (1 - self.sigmoid(self.sum_o))

            # Input gate gradients
            dE_dw_ini = d_loss * self.out_percentage * (1 - np.tanh(self.Long_term) ** 2) * self.g_add * self.sigmoid(self.sum_i) * (1 - self.sigmoid(self.sum_i)) * num
            dE_dw_sti = d_loss * self.out_percentage * (1 - np.tanh(self.Long_term) ** 2) * self.g_add * self.sigmoid(self.sum_i) * (1 - self.sigmoid(self.sum_i)) * self.Short_term
            dE_db_i = d_loss * self.out_percentage * (1 - np.tanh(self.Long_term) ** 2) * self.g_add * self.sigmoid(self.sum_i) * (1 - self.sigmoid(self.sum_i))
            dE_dw_ing  = d_loss * self.out_percentage * (1-np.tanh (self.Long_term)**2) * self.i_percent * (1 - np.tanh(self.sum_g)**2)*num
            dE_dw_stg  = d_loss * self.out_percentage * (1-np.tanh(self.Long_term)**2) * self.i_percent * (1 - np.tanh(self.sum_g)**2)*self.Short_term
            dE_db_g  = d_loss * self.out_percentage * (1-np.tanh(self.Long_term)**2) * self.i_percent * (1 - np.tanh(self.sum_g)**2)

            # Forget gate gradients
            dE_dw_inf = d_loss * self.out_percentage * (1 - np.tanh(self.Long_term) ** 2) * self.sigmoid(self.sum_f) * (1 - self.sigmoid(self.sum_f)) * num
            dE_dw_stf = d_loss * self.out_percentage * (1 - np.tanh(self.Long_term) ** 2) * self.sigmoid(self.sum_f) * (1 - self.sigmoid(self.sum_f)) * self.Short_term
            dE_db_f = d_loss * self.out_percentage * (1 - np.tanh(self.Long_term) ** 2) * self.sigmoid(self.sum_f) * (1 - self.sigmoid(self.sum_f))
            
            # Update weights
            self.Iws[0] -= self.lr * dE_dw_inf
            self.Sws[0] -= self.lr * dE_dw_stf
            self.Bs[0] -= self.lr * dE_db_f
            
            self.Iws[1] -= self.lr * dE_dw_ini
            self.Sws[1] -= self.lr * dE_dw_sti
            self.Bs[1] -= self.lr * dE_db_i
            
            self.Iws[2] -= self.lr * dE_dw_ing
            self.Sws[2] -= self.lr * dE_dw_stg
            self.Bs[2] -= self.lr * dE_db_g
            
            self.Iws[3] -= self.lr * dE_dw_ino
            self.Sws[3] -= self.lr * dE_dw_sto
            self.Bs[3] -= self.lr * dE_db_o

