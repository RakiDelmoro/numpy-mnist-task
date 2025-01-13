import random
import numpy as np
from torch import tensor
from torch.nn.functional import relu, softmax
from modelv2.utils import initialize_params
from features import GREEN, RED, RESET

def mlp(size):
    paramaters = initialize_params(size)

    gen_matrix = np.random.randn(784, 10)

    def forward(input_neurons):
        activations = [input_neurons]
        neurons_activation = input_neurons
        for layer_idx in range(len(paramaters)):
            last_layer = layer_idx == len(paramaters) - 1
            pre_activation = np.matmul(neurons_activation, paramaters[layer_idx])
            if last_layer: neurons_activation = np.array(softmax(tensor(pre_activation), dim=-1), dtype=np.float32)
            else: neurons_activation = np.array(relu(tensor(pre_activation)), dtype=np.float32)
            activations.append(neurons_activation)
        return activations

    def calculate_loss(model_output, expected_output):
        one_hot_expected = np.zeros_like(model_output)
        one_hot_expected[np.arange(len(expected_output)), expected_output] = 1
        avg_neurons_loss = np.array(-np.mean(np.sum(one_hot_expected * np.log(model_output), axis=-1)))
        return avg_neurons_loss

    def update_params(activations, loss):
        for i in range(len(paramaters)):
            previous_activation = activations[-(i+2)]
            layer_params = paramaters[-(i+1)]
            #TODO: Include previous neurons activation into weight update
            layer_params -= 0.01 * (layer_params * loss)

    def training(dataloader):
        losses = []
        for x_train, y_train in dataloader:
            activations = forward(x_train)
            global_loss = calculate_loss(activations[-1], y_train)
            update_params(activations, global_loss)
            losses.append(global_loss)
            print(global_loss)
        return np.mean(np.array(losses))
    
    def test(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            neurons_activations = forward(batched_image)
            batch_accuracy = (neurons_activations[-1].argmax(axis=-1) == batched_label).mean()
            for each in range(len(batched_label)//10):
                model_prediction = neurons_activations[-1][each].argmax()
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
    
    return training, test

    def runner(training_loader, validation_loader, epochs):
        for _ in range(epochs):
            avg_loss = training(training_loader)
            accuracy = test(validation_loader)

    return runner
