import numpy as np


def softmax(a): # softmax regularization
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def sigmoid(x):
    return 1/(1 - np.exp(-x))


# identity functions, these functions return their input
def forget_layer(combine):      # identity function
    return combine
def candidate_layer(combine):   # identity function
    return combine
def input_layer(combine):       # identity function
    return combine
def output_layer(combine):      # identity function
    return combine


# what t(ft, it, ct, ot, ht) mean is time
def LSTMCELL(prev_ct, prev_ht, input):
    combine = prev_ht + input
    ft = forget_layer(combine)              # fotget layer
    candidate = candidate_layer(combine)    # candidate layer
    it = input_layer(combine)               # input layer
    ct = prev_ct * ft + candidate * it      # cell state
    ot = output_layer(combine)              # output layer
    ht = ot * np.tanh(ct)                   # hidden state
    return ht, ct


ct = np.array([0.0, 0.0, 0.0])  # time t(1, 2, 3...)th cell state, ct
ht = np.array([0.0, 0.0, 0.0])  # time t(1, 2, 3...)th hidden state, ht
inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
regularized_inputs = softmax(inputs)

for input in regularized_inputs:
    ct, ht = LSTMCELL(ct, ht, input)
    print(ht, ct)