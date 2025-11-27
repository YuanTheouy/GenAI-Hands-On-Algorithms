import numpy as np

class SoftmaxLayer:
    def __init__(self):
        self.softmax_out = None

    def forward(self, x):
        x_max = np.max(x, axis=1, keepdims=True)
        x_shifted = x - x_max

        exp_x = np.exp(x_shifted)

        sum_exp = np.sum(exp_x,axis=1,keepdims=True)
        self.softmax_out =exp_x/sum_exp

        return self.softmax_out
    
    def backward(self,dout):
        
        sum_s_dout = np.sum(self.softmax_out * dout, axis=1, keepdim=True)

        dx =self.softmax_out * (dout - sum_s_dout)
        
        return dx
    