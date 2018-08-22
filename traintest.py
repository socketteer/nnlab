import numpy as np
from matplotlib import pyplot as plt
import nn

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

# some hyperparameters
step_size = 1e-0
reg = 1e-3  # regularization strength

neuralnet = nn.NN(input_size=D, output_size=K, num_hidden=1, hidden_size=100)
neuralnet.print_meta()

neuralnet.train(X, y, step_size=step_size, reg_strength=reg, epochs=10000)