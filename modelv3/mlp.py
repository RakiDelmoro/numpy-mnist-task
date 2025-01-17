import random
import numpy as np
from tqdm import tqdm, trange
from torch import tensor
from features import GREEN, RED, RESET
from modelv3.utils import initialize_params, one_hot_encoded, initialize_random_params, relu, softmax

def model(size):
    paramaters = initialize_params(size)
    random_parameters = initialize_random_params(size)

    def forward(input_neurons):
        activations = [input_neurons]
        neurons_activation = input_neurons
        for layer_idx in range(len(paramaters)):
            weights, bias = paramaters[layer_idx]
            last_layer = layer_idx == len(paramaters) - 1
            pre_activation = np.matmul(neurons_activation, weights) + bias
            neurons_activation = softmax(pre_activation) if last_layer else np.tanh(pre_activation)
            activations.append(neurons_activation)
        return activations

    def calculate_loss(model_output, expected_output):
        avg_neurons_loss = -np.mean(np.sum(expected_output * np.log(model_output + 1e-15 ), axis=1))
        neurons_loss = model_output - expected_output
        return avg_neurons_loss, neurons_loss

    def update_params(forward_pass_act, layer_neurons_loss, learning_rate):
        loss_to_minimize = layer_neurons_loss
        for i in range(len(paramaters)):
            previous_activation = forward_pass_act[-(i+2)]
            # Parameters to update
            weights = paramaters[-(i+1)][0]
            bias = paramaters[-(i+1)][1]
            # Parameters update
            weights -= learning_rate * (np.matmul(previous_activation.transpose(1, 0), loss_to_minimize)) / previous_activation.shape[0]
            bias -= learning_rate * (np.sum(loss_to_minimize, axis=0)) / previous_activation.shape[0]
            # Propagate loss using random weight
            loss_to_minimize = np.matmul(layer_neurons_loss, random_parameters[i].transpose(1, 0))

    def training(dataloader, learning_rate, total_samples):
        losses = []
        num_losses = 0
        for i in (t := trange(total_samples, leave=False)):
            x_train, y_train = next(iter(dataloader))
            one_hot_y = one_hot_encoded(x_train, y_train)
            forward_activations = forward(x_train)
            global_loss, neurons_loss = calculate_loss(forward_activations[-1], one_hot_y)
            update_params(forward_activations, neurons_loss, learning_rate)
            losses.append(global_loss)
            if num_losses == 50:
                print()
                print(f'Average loss: {np.mean(np.array(losses))} Loss: {global_loss}')
                num_losses = 0
            else:
                t.set_description(f'Loss: {global_loss}')
                num_losses += 1 
        return np.mean(np.array(losses))

    def test(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            neurons_activations = forward(batched_image)
            batch_accuracy = (neurons_activations[-1].argmax(axis=-1) == batched_label).mean()
            model_prediction = neurons_activations[-1].argmax()
            if model_prediction == batched_label.item(): correctness.append((model_prediction.item(), batched_label.item()))
            else: wrongness.append((model_prediction.item(), batched_label.item()))
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
