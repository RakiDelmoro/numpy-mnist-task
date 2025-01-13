import math
import torch
import numpy as np
from torch.nn.init import kaiming_uniform_

def initialize_params(model_size):
    parameters = []
    for i in range(len(model_size)-1):
        gen_matrix = np.empty(shape=(model_size[i], model_size[i+1]))
        weights = np.array(kaiming_uniform_(torch.tensor(gen_matrix), a=math.sqrt(5)))
        parameters.append(weights)
    return parameters

def initialize_stress(model_size):
    return [np.zeros(shape=(model_size[i], model_size[i+1])) for i in range(len(model_size)-1)]
