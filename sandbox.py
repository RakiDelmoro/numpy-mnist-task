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
        h_0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h_0)
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
        with torch.no_grad():
            predictions = self.forward(x_test).squeeze(2).numpy()

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(y_test[0].numpy(), label='True')
        plt.plot(predictions[0], label='Predicted')
        plt.legend()
        plt.savefig('prediction.png')
        plt.show()

# Numpy RNN 🧠
def NumpyRNN(weight_ih, weight_hh, bias_ih, bias_hh, ln_weight, ln_bias):
    # Forward pass
    def forward(x, batch_first=True):
        # RNN Layer
        x = x.unsqueeze(-1).numpy()
        if batch_first: x = x.transpose(1, 0, 2)
        seq_len, batch_size, _ = x.shape
        # mdel memory start with zeros Thinking...
        starting_memory = np.zeros(shape=(batch_size, weight_hh.shape[0]))
        current_memory = starting_memory
        memories = [starting_memory]
        produce_memories = []
        for t in range(seq_len):
            current_state = np.matmul(x[t], weight_ih.T) + bias_ih
            memory_state = np.matmul(current_memory, weight_hh.T) + bias_hh
            activation = np.tanh(current_state + memory_state)
            produce_memories.append(activation)
            memories.append(activation)
            current_memory = activation
        rnn_output = np.stack(produce_memories)
        if batch_first: rnn_output = rnn_output.transpose(1, 0, 2)

        # Linear layer
        output = np.matmul(rnn_output, ln_weight.T) + ln_bias
        return output, memories, x.transpose(1, 0, 2)

    def backward(model_pred, expected, activations):
        _, seq_len = expected.shape
        expected = expected.unsqueeze(-1).numpy()
        loss = np.mean((model_pred - expected)**2)
        neuron_stress = 2*(model_pred - expected)
        # stress propagated in axons
        stress_propagated_ln_out_axons = np.matmul(neuron_stress, ln_weight)
        memory_neurons_stress = []
        for t in range(seq_len):
            # activation neuron stress
            memory_neuron_stress = (1 - activations[-(t+1)]**2) * stress_propagated_ln_out_axons[:, t, :]
            # stress propagated to memory to activation axons
            memory_to_activation_stress = np.matmul(memory_neuron_stress, weight_hh)
            memory_neurons_stress.append(memory_to_activation_stress)
        return loss, neuron_stress, memory_neurons_stress

    def update_params(input_activations, activations, last_neurons_stress, memories_stress):
        # modifiable parameters
        nonlocal weight_ih, weight_hh, bias_ih, bias_hh, ln_weight, ln_bias
    
        # output layer params
        memory_neurons = np.stack(activations[1:], axis=1).transpose(0, 2, 1)
        output_params_nudge = np.mean(np.matmul(memory_neurons, last_neurons_stress), axis=0).transpose(1, 0)
        ln_weight -= 0.1 * output_params_nudge
        ln_bias -= 0.1 * np.mean(np.sum(last_neurons_stress, axis=1), axis=0)

        # hidden to hidden params
        previous_memories = np.stack(activations[:-1], axis=1).transpose(0, 2, 1)
        predicted_memories = np.stack(memories_stress, axis=1)
        hh_axon_nudge = np.mean(np.matmul(previous_memories, predicted_memories), axis=0).transpose(1, 0)
        weight_hh -= 0.1 * hh_axon_nudge
        bias_hh -= 0.1 * np.mean(np.sum(predicted_memories, axis=1), axis=0)

        # input to hidden params
        memories_stress = np.stack(activations[1:], axis=1)
        ih_axon_nudge = np.mean(np.matmul(input_activations.transpose(0, 2, 1), memories_stress), axis=0).transpose(1, 0)
        weight_ih -= 0.1 * ih_axon_nudge
        bias_ih -= 0.1 * np.mean(np.sum(memories_stress, axis=1), axis=0)

    def runner(x_train, y_train, epochs):
        for _ in range(epochs):
            model_pred, model_activations, input_activation = forward(x_train)
            loss, last_neurons_stress, memories_neurons_stress = backward(model_pred, y_train, model_activations)
            update_params(input_activation, model_activations, last_neurons_stress, memories_neurons_stress)            
    return runner

def runner():
    # TORCH Model🔥
    TORCH_MODEL = TorchRNN(input_size=1, hidden_size=20, output_size=1)
    
    # Model Parameters
    # structure will change depends on how many RNN layers we have l0, l1, l2, ... (current we only have one RNN layer)
    input_to_hidden_w = TORCH_MODEL.rnn.weight_ih_l0.detach().numpy()
    hidden_to_hidden_w = TORCH_MODEL.rnn.weight_hh_l0.detach().numpy()
    input_to_hidden_b = TORCH_MODEL.rnn.bias_ih_l0.detach().numpy()
    hidden_to_hidden_b = TORCH_MODEL.rnn.bias_hh_l0.detach().numpy()
    linear_out_w = TORCH_MODEL.linear_out.weight.detach().numpy()
    linear_out_b = TORCH_MODEL.linear_out.bias.detach().numpy()

    # NUMPY Model🪲
    NUMPY_MODEL = NumpyRNN(input_to_hidden_w, hidden_to_hidden_w, input_to_hidden_b, hidden_to_hidden_b, linear_out_w, linear_out_b)

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
    # TORCH_MODEL.runner(x_train, y_train, x_test, y_test, EPOCHS)
    # 🪲🏃‍♂️‍➡️
    NUMPY_MODEL(x_train, y_train, EPOCHS)
    # print(torch_output.shape, numpy_output.shape)

runner()
