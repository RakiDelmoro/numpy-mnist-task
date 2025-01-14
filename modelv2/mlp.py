import random
import numpy as np
from torch import tensor
from torch.nn.functional import relu, softmax
from modelv2.utils import initialize_params, one_hot_encoded
from features import GREEN, RED, RESET

def mlp(size):
    paramaters = initialize_params(size)

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

    def backward(expected_neurons):
        activations = []
        neurons_activation = expected_neurons
        for layer_idx in range(len(paramaters)-1):
            pre_activation = np.matmul(neurons_activation, paramaters[-(layer_idx+1)].transpose(1, 0))
            neurons_activation = np.array(relu(tensor(pre_activation)), dtype=np.float32)
            activations.append(neurons_activation)
        return activations
    
    def calculate_layer_stress(last_neurons_stress, forward_pass_act, backward_pass_act):
        layers_stresses = [last_neurons_stress]
        for i in range(len(backward_pass_act)):
            dissimalirity = forward_pass_act[-(i+2)] - backward_pass_act[-(i+1)]
            layers_stresses.append(dissimalirity)
        return layers_stresses

    def calculate_loss(model_output, expected_output):
        avg_neurons_loss = np.array(-np.mean(np.sum(expected_output * np.log(model_output), axis=-1)))
        neurons_loss = model_output - expected_output
        return avg_neurons_loss, neurons_loss

    def update_params(forward_pass_act, layer_neurons_loss):
        for i in range(len(paramaters)):
            previous_activation = forward_pass_act[-(i+2)]
            neurons_loss = layer_neurons_loss[i]
            layer_params = paramaters[-(i+1)]
            layer_params -= 0.1 * (np.matmul(previous_activation.transpose(1, 0), neurons_loss) / previous_activation.shape[0])

    def training(dataloader):
        losses = []
        for x_train, y_train in dataloader:
            one_hot_y = one_hot_encoded(x_train, y_train)
            forward_activations = forward(x_train)
            backward_activations = backward(one_hot_y)
            global_loss, neurons_loss = calculate_loss(forward_activations[-1], one_hot_y)
            layers_stresses = calculate_layer_stress(neurons_loss, forward_activations, backward_activations)
            update_params(forward_activations, layers_stresses)
            losses.append(global_loss)
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
