import numpy as np
import matplotlib.pyplot as plt
from neurons import recurrent_neurons

# Numpy RNN 🧠
def NumpyRNNV2(weight_ih, weight_hh, bias_ih, bias_hh, ln_con_w, ln_con_b, ln_weight, ln_bias):
    # Model stress initializer start with zeros Model is 🏖️🌴
    weight_ih_stress = np.zeros_like(weight_ih)
    bias_ih_stress = np.zeros_like(bias_ih)
    ln_weight_con_stress = np.zeros_like(ln_con_w)
    ln_bias_con_stress = np.zeros_like(ln_con_b)
    ln_out_weight_stress = np.zeros_like(ln_weight)
    ln_out_bias_stress = np.zeros_like(ln_bias)

    # Generate_random_params (weight to transport our loss)
    hidden_to_hidden_mem_1 = np.random.rand(ln_weight.shape[0], weight_hh.shape[0])
    hidden_to_hidden_mem_2 = np.random.rand(ln_weight.shape[0], weight_hh.shape[0])

    # 🧠⏩
    def forward(x, batch_first=True):
        # RNN layer forward pass
        rnn_output1, memories1 = recurrent_neurons(x, [weight_ih, bias_ih], [weight_hh, bias_hh])
        # lin_out = np.matmul(rnn_output1, ln_con_w.T) + ln_con_b
        rnn_output2, memories2 = recurrent_neurons(rnn_output1, [ln_con_w, ln_con_b], [weight_hh, bias_hh])

        # output layer
        output = np.matmul(rnn_output2, ln_weight.T) + ln_bias
        return output, memories1, memories2

    def backward(model_pred, expected, mem_activations1, mem_activations2, input_activation):
        nonlocal weight_ih_stress, bias_ih_stress, ln_weight_con_stress, ln_bias_con_stress, ln_out_weight_stress, ln_out_bias_stress

        batch, seq_len = expected.shape
        expected = expected.unsqueeze(-1).numpy()
        loss = np.mean((model_pred - expected)**2)
        neuron_stress = 2*(model_pred - expected)

        # Output neurons stress 
        neurons_memories_2 = np.stack(mem_activations2[1:], axis=1).reshape(batch*seq_len, -1)
        ln_out_weight_stress += (np.matmul(neuron_stress.reshape(batch*seq_len, -1).transpose(1, 0), neurons_memories_2) / neurons_memories_2.shape[0])
        ln_out_bias_stress += np.mean(np.mean(neuron_stress, axis=1), axis=0)
        # Memory 2 neurons stress
        neurons_memories_1 = np.stack(mem_activations1[1:], axis=1).reshape(batch*seq_len, -1)
        memories_stress_2 = np.matmul(neuron_stress, hidden_to_hidden_mem_2)
        ln_weight_con_stress += (np.matmul(memories_stress_2.reshape(batch*seq_len, -1).transpose(1, 0), neurons_memories_1) / (batch*seq_len))
        ln_bias_con_stress += np.mean(np.mean(memories_stress_2, axis=1), axis=0)
        # Memory 1 neurons stress
        memories_stress_1 = np.matmul(neuron_stress, hidden_to_hidden_mem_1)
        weight_ih_stress += (np.matmul(memories_stress_1.reshape(batch*seq_len, -1).transpose(1, 0), input_activation.reshape(batch*seq_len, -1)) / (batch*seq_len))
        bias_ih_stress += np.mean(np.mean(memories_stress_1, axis=1), axis=0)
        return loss

    def update_params():
        # modifiable network stress
        nonlocal weight_ih_stress, bias_ih_stress, ln_weight_con_stress, ln_bias_con_stress, ln_out_weight_stress, ln_out_bias_stress
        # modifiable network parameters
        nonlocal weight_ih, weight_hh, bias_ih, bias_hh, ln_con_w, ln_con_b, ln_weight, ln_bias

        # output layer params
        ln_weight -= 0.1 * ln_out_weight_stress
        ln_bias -= 0.1 * ln_out_bias_stress
        ln_con_w -= 0.1 * ln_weight_con_stress
        ln_con_b -= 0.1 * ln_bias_con_stress
        weight_ih -= 0.1 * weight_ih_stress
        bias_ih -= 0.1 * bias_ih_stress

        # model back to 🏖️🌴
        weight_ih_stress = np.zeros_like(weight_ih)
        bias_ih_stress = np.zeros_like(bias_ih)
        ln_weight_con_stress = np.zeros_like(ln_con_w)
        ln_bias_con_stress = np.zeros_like(ln_con_b)
        ln_out_weight_stress = np.zeros_like(ln_weight)
        ln_out_bias_stress = np.zeros_like(ln_bias)

    def runner(x_train, y_train, x_test, y_test, epochs):
        nonlocal weight_ih, weight_hh, bias_ih, bias_hh, ln_weight, ln_bias
        for epoch in range(epochs):
            input_for_model = x_train.unsqueeze(-1).numpy()
            model_pred, mem_1, mem_2  = forward(input_for_model)
            loss = backward(model_pred, y_train, mem_1, mem_2, input_for_model)
            update_params()
            # Check model performance 10 epochs interval
            if (epoch + 1) % 10 == 0: print(f'Epoch [{epoch+1}/{epoch}], Loss: {loss.item():.4f}')

        predictions = forward(x_test.unsqueeze(-1).numpy())[0].squeeze(2)
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(y_test[0], label='True')
        plt.plot(predictions[0], label='Predicted')
        plt.legend()
        plt.savefig('numpy_prediction_v2.png')
        plt.show()

    return runner
