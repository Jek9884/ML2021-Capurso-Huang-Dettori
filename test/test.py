import numpy as np
import sys
sys.path.append('../')

from network import Network
from layer import Layer
from function import act_dict

def forward_test(network, in_vec, exp_res):

    out = network.forward(in_vec)

    return np.equal(out, exp_res)

net = Network(np.array([2, 2, 2]), act_dict["identity"], act_dict["identity"])

print(forward_test(net, np.array([3, 3]), np.array([12., 12.])))
