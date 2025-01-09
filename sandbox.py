import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt

def generate_data(seq_length, num_samples):
    x_data = []
    y_data = []
    for i in range(num_samples):
        x = np.linspace(i * 2 * np.pi, (i + 1) * 2 * np.pi, seq_length + 1)
        sine_wave = np.sin(x)
        x_data.append(sine_wave[:-1])  # input sequence
        y_data.append(sine_wave[1:])   # target sequence
    return np.array(x_data), np.array(y_data)

# Torch RNN 🔥
class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TorchRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

        # Model properties
        self.loss_function = nn.MSELoss()
        # Note: Adam is smarter than SGD for simplicity we will use SGD
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

        # Model parameters
        self.input_to_hidden_w = self.rnn.weight_ih_l0
        self.hidden_to_hidden_w = self.rnn.weight_hh_l0
        self.input_to_hidden_b = self.rnn.bias_ih_l0
        self.hidden_to_hidden_b = self.rnn.bias_hh_l0
        self.linear_out_w = self.linear_out.weight
        self.linear_out_b = self.linear_out.bias

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        out = self.linear_out(out)
        return out

    def runner(self, x_train, y_train, x_test, y_test, epochs):
        for epoch in range(epochs):
            output = self.forward(x_train)
            loss = self.loss_function(output, y_train.unsqueeze(-1))
            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Check model performance 10 epochs interval
            if (epoch + 1) % 10 == 0: print(f'Epoch [{epoch+1}/{epoch}], Loss: {loss.item():.4f}')
    
        # Make predictions
        with torch.no_grad(): predictions = self.forward(x_test).squeeze(2).numpy()

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(y_test[0].numpy(), label='True')
        plt.plot(predictions[0], label='Predicted')
        plt.legend()
        plt.savefig('prediction.png')
        plt.show()

def runner():
    # Models
    TORCH_MODEL = TorchRNN(input_size=1, hidden_size=20, output_size=1)

    # Model Properties
    EPOCHS = 100
    NUM_SAMPLES = 1000
    LOSS_FUNCTION = nn.MSELoss()
    OPTIMIZER = torch.optim.SGD(TORCH_MODEL.parameters(), lr=0.1)

    # Model Parameters
    input_to_hidden_w = TORCH_MODEL.rnn.weight_ih_l0.detach().numpy()[np.newaxis, :]
    hidden_to_hidden_w = TORCH_MODEL.rnn.weight_hh_l0.detach().numpy()[np.newaxis, :]
    input_to_hidden_b = TORCH_MODEL.rnn.bias_ih_l0.detach().numpy()[np.newaxis, :]
    hidden_to_hidden_b = TORCH_MODEL.rnn.bias_hh_l0.detach().numpy()[np.newaxis, :]
    linear_out_w = TORCH_MODEL.linear_out.weight.detach().numpy()
    linear_out_b = TORCH_MODEL.linear_out.bias.detach().numpy()

    # Generate Data
    X, Y = generate_data(seq_length=200, num_samples=NUM_SAMPLES)
    # Split Data
    x_train, y_train = torch.tensor(X[:int(NUM_SAMPLES * 0.9)], dtype=torch.float32), torch.tensor(Y[:int(NUM_SAMPLES * 0.9)], dtype=torch.float32)
    x_test, y_test = torch.tensor(X[int(NUM_SAMPLES * 0.9):], dtype=torch.float32), torch.tensor(Y[int(NUM_SAMPLES * 0.9):], dtype=torch.float32)

    # Torch model runner 🔥
    TORCH_MODEL.runner(x_train, y_train, x_test, y_test, EPOCHS)

runner()
