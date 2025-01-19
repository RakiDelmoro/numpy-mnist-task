import numpy as np
import matplotlib.pyplot as plt

# Numpy RNN 🧠
def NumpyRNNV2(weight_ih, weight_hh, bias_ih, bias_hh, ln_weight, ln_bias):
    # Model stress initializer start with zeros Model is 🏖️🌴
    weight_ih_stress = np.zeros_like(weight_ih)
    bias_ih_stress = np.zeros_like(bias_ih)
    weight_hh_stress = np.zeros_like(weight_hh)
    bias_hh_stress = np.zeros_like(bias_hh)
    ln_weight_stress = np.zeros_like(ln_weight)
    ln_bias_stress = np.zeros_like(ln_bias)

    # Generate_random_params (weight to transport our loss)
    hidden_to_hidden = np.random.rand(ln_weight.shape[0], weight_hh.shape[0])

    # 🧠⏩
    def forward(x, batch_first=True):
        # RNN layer forward pass
        x = x.unsqueeze(-1).numpy()
        if batch_first: x = x.transpose(1, 0, 2)
        seq_len, batch_size, _ = x.shape
        # model memory start with zeros🤔
        starting_memory = np.zeros(shape=(batch_size, weight_hh.shape[0]))
        previous_memory = starting_memory
        memories = [starting_memory]
        produce_memories = []
        for t in range(seq_len):
            current_memory_state = np.matmul(x[t], weight_ih.T) + bias_ih
            previous_memory_state = np.matmul(previous_memory, weight_hh.T) + bias_hh
            activation = np.tanh(current_memory_state + previous_memory_state)
            produce_memories.append(activation)
            memories.append(activation)
            previous_memory = activation
        rnn_output = np.stack(produce_memories)
        if batch_first: rnn_output = rnn_output.transpose(1, 0, 2)

        # output layer
        output = np.matmul(rnn_output, ln_weight.T) + ln_bias
        return output, memories

    def backward(model_pred, expected, activations, input_activation):
        nonlocal weight_ih_stress, bias_ih_stress, weight_hh_stress, bias_hh_stress, ln_weight_stress, ln_bias_stress

        batch, seq_len = expected.shape
        expected = expected.unsqueeze(-1).numpy()
        loss = np.mean((model_pred - expected)**2)
        neuron_stress = 2*(model_pred - expected)

        # Output neurons stress 
        neurons_memories = np.stack(activations[1:], axis=1).reshape(batch*seq_len, -1)
        ln_weight_stress += (np.matmul(neuron_stress.reshape(batch*seq_len, -1).transpose(1, 0), neurons_memories) / neurons_memories.shape[0])
        ln_bias_stress += np.mean(np.mean(neuron_stress, axis=1), axis=0)
        # Memory neurons stress
        memories_stress = np.matmul(neuron_stress, hidden_to_hidden)
        weight_ih_stress += (np.matmul(memories_stress.reshape(batch*seq_len, -1).transpose(1, 0), input_activation.reshape(batch*seq_len, -1)) / (batch*seq_len))
        bias_ih_stress += np.mean(np.mean(memories_stress, axis=1), axis=0)

        return loss

    def update_params():
        # modifiable network stress
        nonlocal weight_ih_stress, bias_ih_stress, weight_hh_stress, bias_hh_stress, ln_weight_stress, ln_bias_stress
        # modifiable network parameters
        nonlocal weight_ih, weight_hh, bias_ih, bias_hh, ln_weight, ln_bias

        # output layer params
        ln_weight -= 0.1 * ln_weight_stress
        ln_bias -= 0.1 * ln_bias_stress
        # NOTE: We don't update the weights connecting to previous memory and current memory (Can this learn???)
        # input to hidden params
        weight_ih -= 0.1 * weight_ih_stress
        bias_ih -= 0.1 * bias_ih_stress

        # model back to 🏖️🌴
        weight_ih_stress = np.zeros_like(weight_ih)
        bias_ih_stress = np.zeros_like(bias_ih)
        weight_hh_stress = np.zeros_like(weight_hh)
        bias_hh_stress = np.zeros_like(bias_hh)
        ln_weight_stress = np.zeros_like(ln_weight)
        ln_bias_stress = np.zeros_like(ln_bias)

    def runner(x_train, y_train, x_test, y_test, epochs):
        nonlocal weight_ih, weight_hh, bias_ih, bias_hh, ln_weight, ln_bias
        for epoch in range(epochs):
            model_pred, model_activations  = forward(x_train)
            loss = backward(model_pred, y_train, model_activations, x_train.unsqueeze(-1).numpy())
            update_params()
            # Check model performance 10 epochs interval
            if (epoch + 1) % 10 == 0: print(f'Epoch [{epoch+1}/{epoch}], Loss: {loss.item():.4f}')

        predictions = forward(x_test)[0].squeeze(2)
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(y_test[0], label='True')
        plt.plot(predictions[0], label='Predicted')
        plt.legend()
        plt.savefig('numpy_prediction_v2.png')
        plt.show()

    return runner
