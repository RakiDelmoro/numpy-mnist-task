import math
import torch
import numpy as np

def parameters_init(input_feature, output_feature):
    weights = torch.empty((input_feature, output_feature))
    bias = torch.empty(output_feature)
    torch.nn.init.kaiming_normal_(weights, a=math.sqrt(5))
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    torch.nn.init.uniform_(bias, -bound, bound)
    return [np.array(weights), np.array(bias)]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)