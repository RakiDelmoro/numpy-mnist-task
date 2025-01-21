import torch
import numpy as np
import torch.nn as nn
from rnn_model.torch_rnn import TorchRNN
from rnn_model.numpy_rnn import NumpyRNN
from rnn_model.numpy_rnn_v2 import NumpyRNNV2

def generate_data(seq_length, num_samples):
    x_data = []
    y_data = []
    for i in range(num_samples):
        x = np.linspace(i * 2 * np.pi, (i + 1) * 2 * np.pi, seq_length + 1)
        sine_wave = np.sin(x)
        x_data.append(sine_wave[:-1])  # input sequence
        y_data.append(sine_wave[1:])   # target sequence
    return np.array(x_data), np.array(y_data)

def runner():
    # TORCH Model🔥
    TORCH_MODEL = TorchRNN(input_size=1, hidden_size=20, output_size=1)

    # Model Properties 
    EPOCHS = 100
    NUM_SAMPLES = 1000
    LOSS_FUNCTION = nn.MSELoss()
    OPTIMIZER = torch.optim.SGD(TORCH_MODEL.parameters(), lr=0.1)

    # Generate Data
    X, Y = generate_data(seq_length=200, num_samples=NUM_SAMPLES)
    # Split Data
    x_train, y_train = torch.tensor(X[:int(NUM_SAMPLES * 0.9)], dtype=torch.float32), torch.tensor(Y[:int(NUM_SAMPLES * 0.9)], dtype=torch.float32)
    x_test, y_test = torch.tensor(X[int(NUM_SAMPLES * 0.9):], dtype=torch.float32), torch.tensor(Y[int(NUM_SAMPLES * 0.9):], dtype=torch.float32)

    # 🔥🏃‍♂️‍➡️
    print("🔥 Result...")
    TORCH_MODEL.runner(x_train, y_train, x_test, y_test, EPOCHS)

runner()