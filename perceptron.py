import numpy as np


inputs = np.array([0,0,2,3])
weights = np.array([0.1,0.5,0.7,1.2])
bias = -6

weightsum= np.dot(inputs,weights) + bias

def step_function(weightsum):
    if(weightsum >= 0):
        return 1
    else:
        return 0

step_function(weightsum)

# weightsum= dot(x,w)

bias = -6
# -1 =  dot([0,0,2,3] , [0.1,0.5,0.7,1.2]) + bias
# 0 = step_function(-1)

# bias = 6
# 1 =  dot([-4,0,2,-5] , [0.1,0.5,0.7,1.2]) + bias
# 1 = step_function(1)