import numpy as np

def online_softmax_step(new_val,curr_m,curr_d):
    next_m = max(curr_m,new_val)
    scale = np.exp(next_m - curr_m)
    next_d = curr_d * scale + np.exp(new_val - next_m)
    return next_m, next_d


