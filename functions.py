import numpy as np

def sigmoid(z) :
    return 1/(1+np.exp(-z))

def sigmoid_derivative(a) :
    return a*(1-a)

def BCE_to_sigmoid_derivative(y, t) :
    # derivative of BCE in respect to sigmoid
    return y-t


def softmax(x):
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def softmax_derivative(a) :
    return a*(1-a)

def MCE_to_softmax_derivative(y,t) :
    return y-t

def ReLU(x) :
    return np.maximum(0,x)

def ReLU_derivative(x) :
    return np.where(x >= 0, 1, 0)
