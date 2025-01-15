import math
import torch
import numpy as np
from torch.nn.init import kaiming_uniform_
from torch.nn import init

def relu(input_data, return_derivative=False):
    if return_derivative: return np.where(input_data > 0, 1, 0)
    else: return np.maximum(0, input_data)

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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)
