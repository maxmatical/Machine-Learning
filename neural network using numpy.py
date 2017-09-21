# easy 1 hidden layer neural network
import numpy as np
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ]) # 4x3
y = np.array([[0,1,1,0]]).T # 4x1

# initialize weights
# w0 = 2*np.random.random((3,4)) - 1 #input to hidden layer
# w1 = 2*np.random.random((4,1)) - 1 # hidden to output layer

# gaussian initialization
w0 = np.random.normal(loc=0, scale= 0.001, size = (3,4))
w1 = np.random.normal(loc=0, scale= 0.001, size = (4,1))

rho = 0.001 # learning rate
for j in range(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,w0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,w1))))
    l2_delta = (y - l2)*(l2*(1-l2))  # derivative of sigmoid function is output_layer*(1-output_layer)
    l1_delta = l2_delta.dot(w1.T) * (l1 * (1-l1))
    w1 += rho*l1.T.dot(l2_delta)
    w0 += rho*X.T.dot(l1_delta)



