# nn test
import nn
import visualnn 
import numpy as np

neuralnet = nn.NN(input_size = 10, output_size = 5, num_hidden = 3, hidden_size = [8,7,3], nonlinearity = 'relu')

neuralnet.print_meta()


visualize2 = visualnn.VisualNN(neuralnet)
visualize2.draw('neural net')

outputs = neuralnet.fwdpass(np.random.randn(10))
print('output layer: %a' % outputs)

visualize2.update()
visualize2.draw('fwd propagation of random inputs')