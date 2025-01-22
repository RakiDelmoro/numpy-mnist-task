import math
import gzip
import torch
import pickle
import random
import numpy as np
from torch.nn import init
from neurons import linear_neurons
from features import GREEN, RED, RESET
from torch.nn.init import kaiming_uniform_

def dataloader(image_arrays, label_arrays, batch_size: int, shuffle: bool):
    num_samples = image_arrays.shape[0]
    indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(indices)

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield image_arrays[indices[start:end]], label_arrays[indices[start:end]]

def one_hot_encoded(y_train):
    one_hot_expected = np.zeros(shape=(y_train.shape[0], 10))
    one_hot_expected[np.arange(len(y_train)), y_train] = 1
    return one_hot_expected

def init_params(input_size, output_size):
    gen_w_matrix = torch.empty(size=(input_size, output_size))
    gen_b_matrix = torch.empty(size=(output_size,))
    weights = kaiming_uniform_(gen_w_matrix, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias = init.uniform_(gen_b_matrix, -bound, bound)
    return [np.array(weights), np.array(bias)]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def neuron(input_neurons, parameters, readout_parameters):
    activation = linear_neurons(input_neurons, parameters)
    neuron_activation = linear_neurons(activation, readout_parameters)
    return neuron_activation, activation

def network():
    parameters = [init_params(784, 250) for _ in range(10)]
    # readout parameter is shared for all neurons
    readout_parameter = init_params(250, 1)

    # Weight transport for neuron stress
    input_weight_transport = [np.random.rand(1, 250) for _ in range(10)]

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

        # Propagate stress -> input to memory weights
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

def runner():
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28

    # Load MNIST-Data into memory
    with gzip.open('./mnist-data/mnist.pkl.gz', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    # Validate data shapes 
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT * IMAGE_WIDTH

    train_model, test_model = network()

    for epoch in range(500):
        training_loader = dataloader(train_images, train_labels, batch_size=2098, shuffle=True)
        validation_loader = dataloader(test_images, test_labels, batch_size=2098, shuffle=True)
        loss_avg = train_model(training_loader)
        accuracy = test_model(validation_loader)
        print(f"EPOCH: {epoch+1} Loss: {loss_avg} Accuracy: {accuracy}")

runner()
