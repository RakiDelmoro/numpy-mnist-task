import numpy as np
from model.utils import parameters_init, softmax

def forward_pass(batched_image, parameters):
    activated_neurons = batched_image
    neurons_activations = [batched_image]
    for idx, (axons, dentrites) in enumerate(parameters):
        last_layer = idx == len(parameters) - 1
        neurons = np.matmul(activated_neurons, axons) + dentrites
        activated_neurons = softmax(neurons) if last_layer else neurons
        neurons_activations.append(activated_neurons)
    return neurons_activations

def cross_entropy_loss(model_output, expected_output):
    # Convert true labels to one-hot encoding
    one_hot_expected = np.zeros_like(model_output)
    one_hot_expected[np.arange(len(expected_output)), expected_output] = 1
    # calculate loss
    avg_neurons_loss = -np.mean(np.sum(one_hot_expected * np.log(model_output), axis=-1))
    propagate_loss = model_output - one_hot_expected
    return avg_neurons_loss, propagate_loss

def backward_pass(model_neurons_activations, propagate_loss, parameters):
    reversed_parameters = parameters[::-1]
    layers_neurons_stress = [propagate_loss]
    neurons_stress = propagate_loss
    for axons, _ in reversed_parameters:
        neurons_stress = np.matmul(neurons_stress, axons.T)
        layers_neurons_stress.append(neurons_stress)
    return layers_neurons_stress

def update_parameters(model_neurons_activations, layers_neurons_stress, parameters, learning_rate):
    for idx, (axons, dentrites) in enumerate(parameters):
        axons_nudge = np.matmul(model_neurons_activations[idx].T, layers_neurons_stress[-idx-2]) / model_neurons_activations[idx].shape[0]
        dentrites_nudge = np.sum(layers_neurons_stress[-idx-2], axis=0) / model_neurons_activations[idx].shape[0]
        axons -= learning_rate * axons_nudge
        dentrites -= learning_rate * dentrites_nudge

def train_model(dataloader, parameters, learning_rate):
    per_batch_stress = []
    for i, (batched_image, batched_label) in enumerate(dataloader):
        neurons_activations = forward_pass(batched_image, parameters)
        avg_loss, propagate_loss = cross_entropy_loss(neurons_activations[-1], batched_label)
        neurons_stresses = backward_pass(neurons_activations, propagate_loss, parameters)
        update_parameters(neurons_activations, neurons_stresses, parameters, learning_rate)
        print(avg_loss)
        per_batch_stress.append(avg_loss)

    return np.mean(np.array(per_batch_stress))

def test_model():
    pass

def model(model_size, train_loader, test_loader, learning_rate, epochs):
    parameters = [parameters_init(model_size[i], model_size[i+1]) for i in range(len(model_size)-1)]
    for epoch in range(epochs):
        batch_avg_loss = train_model(train_loader, parameters, learning_rate)
        print(f'EPOCH: {epoch} Loss: {batch_avg_loss}')
        train_loader = train_loader