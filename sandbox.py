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
    # Model stress initializer start with zeros Model is 🏖️🌴
    weight_ih_stress = np.zeros_like(weight_ih)
    bias_ih_stress = np.zeros_like(bias_ih)
    weight_hh_stress = np.zeros_like(weight_hh)
    bias_hh_stress = np.zeros_like(bias_hh)
    ln_weight_stress = np.zeros_like(ln_weight)
    ln_bias_stress = np.zeros_like(ln_bias)

    # 🧠⏩
    def forward(x, batch_first=True):
        # RNN Layer
        x = x.unsqueeze(-1).numpy()
        if batch_first: x = x.transpose(1, 0, 2)
        seq_len, batch_size, _ = x.shape
        # model memory start with zeros🤔
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
        return output, memories

    def backward(model_pred, expected, activations, input_activation):
        nonlocal weight_ih_stress, bias_ih_stress, weight_hh_stress, bias_hh_stress, ln_weight_stress, ln_bias_stress

        batch, seq_len = expected.shape
        expected = expected.unsqueeze(-1).numpy()
        loss = np.mean((model_pred - expected)**2)
        neuron_stress = 2*(model_pred - expected)

        # Output neurons stress 
        ln_weight_stress += np.mean(np.matmul(np.stack(activations[1:], axis=1).transpose(0, 2, 1), neuron_stress), axis=0).transpose(1, 0)
        ln_bias_stress += np.mean(np.sum(neuron_stress, axis=1), axis=0)

        # output neurons stress propagated➡️
        stress_propagated = np.matmul(neuron_stress, ln_weight)

        # network neuron memories stress (model needs to correct it's memories😬)
        neuron_memories_stress = np.zeros(shape=(batch, stress_propagated.shape[-1]))
        for t in reversed(range(seq_len)):
            current_neuron_memory_stress = stress_propagated[:, t, :] + neuron_memories_stress
            # apply tanh differentiable
            neuron_activation_stress = (1 - activations[t]**2) * current_neuron_memory_stress
            # apply stress to the network
            weight_hh_stress += np.matmul(activations[t-1].T, neuron_activation_stress).transpose(1, 0)
            weight_ih_stress += np.matmul(input_activation[:, t, :].T, neuron_activation_stress).transpose(1, 0)
            # memory stress for next iteration
            neuron_memories_stress = np.matmul(neuron_activation_stress, weight_hh)
        return loss, [weight_ih_stress, bias_ih_stress, weight_hh_stress, bias_hh_stress, ln_weight_stress, ln_bias_stress]

    def update_params(network_stresses):
        # modifiable network stress
        nonlocal weight_ih_stress, bias_ih_stress, weight_hh_stress, bias_hh_stress, ln_weight_stress, ln_bias_stress
        # modifiable network parameters
        nonlocal weight_ih, weight_hh, bias_ih, bias_hh, ln_weight, ln_bias

        # output layer params
        ln_weight -= 0.1 * network_stresses[4]
        ln_bias -= 0.1 * network_stresses[5]

        # hidden to hidden params
        weight_hh -= 0.1 * network_stresses[2]
        bias_hh -= 0.1 * network_stresses[3]

        # input to hidden params
        weight_ih -= 0.1 * network_stresses[0]
        bias_ih -= 0.1 * network_stresses[1]

        # model back to 🏖️🌴
        weight_ih_stress = np.zeros_like(weight_ih)
        bias_ih_stress = np.zeros_like(bias_ih)
        weight_hh_stress = np.zeros_like(weight_hh)
        bias_hh_stress = np.zeros_like(bias_hh)
        ln_weight_stress = np.zeros_like(ln_weight)
        ln_bias_stress = np.zeros_like(ln_bias)

    def runner(x_train, y_train, epochs):
        nonlocal weight_ih, weight_hh, bias_ih, bias_hh, ln_weight, ln_bias
        for _ in range(epochs):
            model_pred, model_activations  = forward(x_train)
            loss, network_stresses = backward(model_pred, y_train, model_activations, x_train.unsqueeze(-1).numpy())
            update_params(network_stresses)

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
