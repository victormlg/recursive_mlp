import numpy as np

def add_bias(matrix):
    ones_row = np.ones((1, matrix.shape[1]), dtype=matrix.dtype)
    return np.vstack((ones_row, matrix))

class MLP : 
    start = None
    end = None
    
    def __init__(self, eta=0.01) :
        self.eta = eta

    # # # # # # # # # # # # # # # # # # # # # # # # #

    class Node:
        next = None 
        back = None
        
        def __init__(self,shape, parameters) :
            self.W = np.random.randn(*shape) # or np.ones(shape) or np.zeros(shape)
            self.p = parameters
            
        def forward(self, x) :
            self.input = x
            self.x = add_bias(x)
            self.output = self.p.activation_function(self.W @ self.x)
            return self.output
        
     # # # # # # # # # # # # # # # # # # # # # # # # #

    def add(self, shape, parameters) :
        N, m = shape
        tmp = self.Node((m, N+1), parameters)
        
        if self.start == None:
            self.start = tmp
            self.end = self.start
        else :
            tmp.back = self.end
            self.end.next = tmp
            self.end = tmp

    # predict :

    def _forward(self, tmp, x) :

        if tmp == None :
            return x 
        
        z = tmp.forward(x)
        return self._forward(tmp.next, z)
    
    def predict(self, x) :
        return self._forward(self.start, x) > 0.5
    
    # fit :
    
    def fit(self, x, t) :

        for i in range(7000) :
            y = self._forward(self.start, x)

            delta = self.end.p.activation_derivative(y) * self.end.p.loss_activation_derivative(y,t) 
            self._backwards(self.end, delta)

    def _backwards(self, tmp, delta) :

        if tmp == None :
            return

        tmp.W = tmp.W - self.eta * (delta @ tmp.x.T)

        delta = tmp.p.activation_derivative(tmp.input) * (tmp.W.T[1:] @ delta)

        return self._backwards(tmp.back, delta)
    
