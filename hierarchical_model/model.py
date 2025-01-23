import random
import numpy as np
from features import GREEN, RED, RESET
from hierarchical_model.utils import init_params, neuron, softmax, one_hot_encoded, init_model_parameters, init_weights_stress_transport

def network(neuron_properties=[784, 50, 50], output_neurons_size=10):
    # Params init
    model_parameters = init_model_parameters(neuron_properties, output_neurons_size)
    stress_transport_parameters = init_weights_stress_transport(neuron_properties, output_neurons_size)

    # readout parameter is shared for all neurons
    readout_parameter = init_params(50, 1)

    # Neurons
    neurons = [neuron(model_parameters[each_neuron], readout_parameter) for each_neuron in range(output_neurons_size)]

    def forward(input_neurons):
        neurons = []
        neurons_memories = []
        for i in range(10):
            neuron_activation, neuron_memories = neuron(input_neurons, parameters[i], readout_parameter)
            neurons.append(neuron_activation)
            neurons_memories.append(neuron_memories)
        output_neurons = softmax(np.concatenate(neurons, axis=1, dtype=np.float32))
        return output_neurons, neurons_memories

    def neurons_stress(model_output, expected_output):
        avg_neurons_loss = -np.mean(np.sum(expected_output * np.log(model_output + 1e-15), axis=1))
        return avg_neurons_loss

    def update_each_neuron(input_neurons, neuron_memory, weight, neuron_parameters, neuron_stress):
        stress = neuron_stress.reshape(-1, 1)

        readout_weights = readout_parameter[0]
        readout_bias = readout_parameter[1]
        readout_weights -= 0.01 * np.matmul(neuron_memory.transpose(), stress) / input_neurons.shape[0]
        readout_bias -= 0.01 * np.sum(stress, axis=0) / input_neurons.shape[0]

        neurons_stress = np.matmul(stress, weight)

        # Input to neuron weights
        input_weights = neuron_parameters[0]
        input_bias = neuron_parameters[1]
        input_weights -= 0.01 * np.matmul(input_neurons.transpose(), neurons_stress) / input_neurons.shape[0]
        input_bias -= 0.01 * np.sum(neurons_stress, axis=0) / input_neurons.shape[0]

    def backward(prediction, expected, input_neurons, neurons_memories):
        total_output_neurons = prediction.shape[-1]
        for neuron_idx in range(total_output_neurons):
            weight_transport = input_weight_transport[neuron_idx]
            neuron_parameters = parameters[neuron_idx]
            neuron_memory = neurons_memories[neuron_idx]
            neuron_activation = prediction[:, neuron_idx]
            expected_neuron_activation = expected[:, neuron_idx]
            # Mean squared error for a neuron
            neuron_stress = 2*(neuron_activation - expected_neuron_activation)
            update_each_neuron(input_neurons, neuron_memory, weight_transport, neuron_parameters,  neuron_stress)

    def training_phase(dataloader):
        batch_losses = []
        for batch_image, batch_expected in dataloader:
            input_batch_image = batch_image
            one_hot_encoded_expected = one_hot_encoded(batch_expected)
            prediction, neurons_memories = forward(input_batch_image)
            avg_neurons_stress = neurons_stress(prediction, one_hot_encoded_expected)
            backward(prediction, one_hot_encoded_expected, input_batch_image, neurons_memories)
            print(avg_neurons_stress)
            batch_losses.append(avg_neurons_stress)

        return np.mean(np.array(batch_losses))

    def testing_phase(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            input_image = batched_image
            prediction, _ = forward(input_image)
            batch_accuracy = (prediction.argmax(axis=-1) == batched_label).mean()
            for each in range(len(batched_label)//10):
                model_prediction = prediction[each].argmax(-1)
                if model_prediction == batched_label[each]: correctness.append((model_prediction.item(), batched_label[each].item()))
                else: wrongness.append((model_prediction.item(), batched_label[each].item()))
            print(f'Number of samples: {i+1}\r', end='', flush=True)
            accuracy.append(np.mean(batch_accuracy))
        random.shuffle(correctness)
        random.shuffle(wrongness)
        print(f'{GREEN}Model Correct Predictions{RESET}')
        [print(f"Digit Image is: {GREEN}{expected}{RESET} Model Prediction: {GREEN}{prediction}{RESET}") for i, (prediction, expected) in enumerate(correctness) if i < 5]
        print(f'{RED}Model Wrong Predictions{RESET}')
        [print(f"Digit Image is: {RED}{expected}{RESET} Model Prediction: {RED}{prediction}{RESET}") for i, (prediction, expected) in enumerate(wrongness) if i < 5]
        return np.mean(np.array(accuracy)).item()

    return training_phase, testing_phase
