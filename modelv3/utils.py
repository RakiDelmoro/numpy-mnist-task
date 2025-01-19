import math
import torch
import numpy as np
from torch.nn import init
from torch.nn.init import kaiming_uniform_

def count_parameters(network_connections):
    return sum([param.shape[0]*param.shape[1] for param in network_connections])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def relu(input_data, return_derivative=False):
    if return_derivative: return np.where(input_data > 0, 1, 0)
    else: return np.maximum(0, input_data)

def one_hot_encoded(x_train, y_train):
    one_hot_expected = np.zeros(shape=(x_train.shape[0], 10))
    one_hot_expected[np.arange(len(y_train)), y_train] = 1
    return one_hot_expected

def initialize_params(model_size):
    parameters = []
    for i in range(len(model_size)-1):
        gen_w_matrix = torch.empty(size=(model_size[i], model_size[i+1]))
        gen_b_matrix = torch.empty(size=(model_size[i+1],))
        weights = kaiming_uniform_(gen_w_matrix, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bias = init.uniform_(gen_b_matrix, -bound, bound)
        parameters.append([np.array(weights), np.array(bias)])
    return parameters

def initialize_random_params(model_size):
    gen_parameters = []
    for i in range(len(model_size)-1):
        gen_matrix = np.random.rand(model_size[-(i+2)], model_size[-1])
        gen_parameters.append(gen_matrix)
    return gen_parameters 

def initialize_stress(model_size):
    return [np.zeros(shape=(model_size[i], model_size[i+1])) for i in range(len(model_size)-1)]
