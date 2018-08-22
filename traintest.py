import numpy as np
import nn
import visualnn
from matplotlib import pyplot as plt


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

visualize = visualnn.VisualNN(neuralnet)
visualize.save('traintest/frame0.png', "Epoch 0")

for step in range(25):
    loss = neuralnet.train(X, y, step_size=step_size, reg_strength=reg, epochs=200, compute_loss_every=100)
    visualize.update()
    visualize.save('traintest/frame%d.png' % (step + 1), "Epoch %d, Loss = %f" % ((step+1)*200, loss))