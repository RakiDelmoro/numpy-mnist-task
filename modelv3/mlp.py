import random
import numpy as np
from torch import tensor
from torch.nn.functional import softmax
from modelv3.utils import initialize_params, one_hot_encoded, initialize_random_params, relu
from features import GREEN, RED, RESET

def mlp(size):
    paramaters = initialize_params(size)
    random_parameters = initialize_random_params(size)

    def forward(input_neurons):
        activations = [input_neurons]
        neurons_activation = input_neurons
        for layer_idx in range(len(paramaters)):
            last_layer = layer_idx == len(paramaters) - 1
            pre_activation = np.matmul(neurons_activation, paramaters[layer_idx])
            if last_layer: neurons_activation = np.array(softmax(tensor(pre_activation), dim=-1), dtype=np.float32)
            else: neurons_activation = relu(pre_activation)
            activations.append(neurons_activation)
        return activations

    def calculate_loss(model_output, expected_output):
        avg_neurons_loss = -np.mean(np.sum(expected_output * np.log(model_output), axis=-1))
        neurons_loss = model_output - expected_output
        return avg_neurons_loss, neurons_loss

    def update_params(forward_pass_act, layer_neurons_loss):
        for i in range(len(paramaters)):
            current_activation = forward_pass_act[-(i+1)]
            previous_activation = forward_pass_act[-(i+2)]
            layer_params = paramaters[-(i+1)]
            # layer_params += 0.1 * ((rule_2 - rule_1) / current_activation.shape[0])
            if i == 0:
                layer_params -= 0.1 * (np.matmul(previous_activation.transpose(1, 0), layer_neurons_loss) / previous_activation.shape[0])
            else:
                neurons_loss = np.matmul(layer_neurons_loss, random_parameters[-(i+1)].transpose(1, 0) / previous_activation.shape[0])
                layer_params -= 0.1 * (np.matmul(previous_activation.transpose(1, 0), neurons_loss) / previous_activation.shape[0])

    def training(dataloader):
        losses = []
        for x_train, y_train in dataloader:
            one_hot_y = one_hot_encoded(x_train, y_train)
            forward_activations = forward(x_train)
            global_loss, neurons_loss = calculate_loss(forward_activations[-1], one_hot_y)
            update_params(forward_activations, neurons_loss)
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
