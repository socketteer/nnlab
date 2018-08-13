import numpy as np
import exceptions

# TODO: backpropagation
# gradient check

# TODO: visualize:
# dropout
# loss: training vs validation
# if image: first layer weights

# representation semantics visualizations


#init from csv file with saved weights and architecture information
def init_from_checkpoint(file):
    return NN()


relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1.0/(1.0 + np.exp(-x))
tanh = lambda x: 2.0/(1.0 + np.exp(-2*x))


class NN:
    '''
    input size: int, size of input layer. defaults to 10
    output size: int, size of output layer. defaults to 1
    num_hidden: number of hidden layers. defaults to 1.
    hidden size: int or list(num_hidden) of hidden layer sizes. If int, all hidden layers are set to the same size. 
        defaults to 10. 
    activation: string. 'sigmoid', 'tanh', 'relu', or 'lrelu'.
    a: metaparameter for lrelu activation only
    labels: list(output_size) of output classification label names
    '''
    def __init__(self,
                 input_size=10,
                 output_size=1,
                 num_hidden=1, 
                 hidden_size=10, 
                 nonlinearity='relu',
                 a=0.01,
                 labels=None):
        
        if num_hidden < 1 or not isinstance(num_hidden, int):
            raise exceptions.InvalidNetworkInit('Invalid number of hidden layers %d. num_hidden should be '
                                                'an integer >= 1' % num_hidden)
        
        if input_size < 1 or not isinstance(input_size, int):
            raise exceptions.InvalidNetworkInit('Invalid input size %d. input_size should be an integer >= 1'
                                                % input_size)
        
        if output_size < 1 or not isinstance(output_size, int):
            raise exceptions.InvalidNetworkInit('Invalid output size %d. output_size should be an integer >= 1'
                                                % output_size)
        
        if nonlinearity == 'relu':
            self.f = relu
        elif nonlinearity == 'sigmoid':
            self.f = sigmoid
        elif nonlinearity == 'tanh':
            self.f = tanh
        elif nonlinearity == 'lrelu':
            self.f = lambda x: (x < 0) * (a * x) + sigmoid(x)
        else:
            raise exceptions.InvalidNetworkInit('Invalid nonlinearity type \'%s\'. '
                                                'Use relu \'rectified linear unit\', sigmoid, tanh, '
                                                'or lrelu \'leaky ReLU\'' % nonlinearity)
            
        if labels and not len(labels) == output_size:
            raise exceptions.InvalidNetworkInit('Number of labels should equal output size')
            
        self.depth = num_hidden + 1

        # constructing (empty) activations
        self.activations = []
        self.activations.append(np.zeros(input_size))

        # checking if hidden_size parameter is valid; if so, initilializing hidden layer activations
        if isinstance(hidden_size, int):
            if hidden_size < 1:
                raise exceptions.InvalidNetworkInit('Invalid hidden_size %d. hidden_size should be an integer >= 1'
                                                    % hidden_size)
            for i in range(num_hidden):
                self.activations.append(np.zeros(hidden_size))
        elif isinstance(hidden_size, list):
            if any(t < 1 for t in hidden_size) or not all(isinstance(t, int) for t in hidden_size):
                raise exceptions.InvalidNetworkInit('One or more values in hidden_size are invalid. '
                                                    'All values in hidden_size must be integers >= 1.')
            if not len(hidden_size) == num_hidden:
                raise exceptions.InvalidNetworkInit('hidden_size vector must be a nonzero integer or '
                                                    'list of length num_hidden of nonzero integers')
            else:
                for i in range(num_hidden):
                    self.activations.append(np.zeros(hidden_size[i]))
        else:
            raise exceptions.InvalidNetworkInit('hidden_size parameter must be a nonzero integer '
                                                'or list of nonzero integers.')
        
        # initializing output layer activations
        self.activations.append(np.zeros(output_size))
        
        # constructing weight matrices (random initialization)
        # TODO is this the right way to init weights??
        self.weights = []
        for i in range(self.depth):
            self.weights.append(0.01 * np.random.randn(len(self.activations[i]), len(self.activations[i+1])))
        
        # constructing bias vectors (zero initialization)
        self.biases = []
        for i in range(self.depth):
            self.biases.append(np.zeros(len(self.activations[i+1])))
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.labels = labels
    
    # init weights from file
    def init_weights(self, file):
        with open(file) as f:
            weights_in = [[float(num) for num in line.split(' ')] for line in f]
            line = 0
            try:
                for i, mat in enumerate(self.weights):
                    for j in range(mat.shape[0]):
                        mat[j, :] = weights_in[line]
                        line += 1
            except IndexError:
                raise exceptions.InitFromFileError('Data in file must conform to dimensions of weight matrices')
            except ValueError as e:
                raise exceptions.InitFromFileError('ValueError: ' + str(e)
                                                   + '. Data in file must conform to dimensions of '
                                                     'network weight matrices')
    
    # save weights to file
    def save_weights(self, file):
        with open(file, 'wb') as f:
            for mat in self.weights:
                np.savetxt(f, mat, fmt='%.5f', delimiter=' ')
    
    # save weights and architecture information to file
    def save_checkpoint(self, file):
        pass
    
    def fwdpass(self, x):
        # check if x is equal to input size
        if not len(x) == self.input_size:
            raise exceptions.InvalidInput('Data must have same dimensionality as network input layer size')
        else:
            self.activations[0] = x
            
        # computing hidden layer activations    
        for i in range(self.depth-1):
            # ReLU 
            self.activations[i+1] = self.f(np.dot(self.activations[i], self.weights[i]) + self.biases[i])
        
        # compute output layer without nonlinearity
        self.activations[self.depth] = np.dot(self.activations[self.depth-1], self.weights[self.depth-1]) \
                                       + self.biases[self.depth-1]
        
        return self.activations[self.depth]    
    
    # TODO this is stochastic gradient descent. Batch option?
    # TODO implement dropout
    '''
    X: training set ((num_examples, input_size))
    '''
    def train(self, X, y, dropout=1, step_size=1e-0, reg_strength=1e-3, epochs=1000):
        # check if parameters are valid
        if dropout <= 0 or dropout > 1:
            raise exceptions.InvalidTrainingParameter('Invalid dropout %f. Valid value are 0 <= dropout <= 1'
                                                      % dropout)
        if not X.shape[1] == self.input_size:
            raise exceptions.InvalidTrainingParameter('parameter X must contain instances with dimension input_size')
        
        
        num_examples = X.shape[0]
        
        # TODO how are batches processed?
        for epoch in range(epochs):
            print('epoch %d' % epoch)
            #loss = self.compute_loss(num_examples, y, reg_strength)
            #print('loss: %d' % loss)
            for i in range(num_examples):
                x = X[i,:]
                correct_class = y[i]
                self.fwdpass(x)
                dW, dB = self.backprop(x, correct_class, reg_strength)
                print(dW)
                print(dB)
                self.param_update(step_size, dW, dB)
                
        # eval

    #TODO figure out how to do this
    def compute_loss(self, batch_size, y, reg_strength):
        exp_scores = np.exp(self.activations[-1])
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(batch_size), y])
        data_loss = np.sum(correct_logprobs) / batch_size
        reg_loss = 0.5 * reg_strength * sum(np.sum(W) for W in self.weights)
        return data_loss + reg_loss

    # TODO dropout?
    def backprop(self, x, correct_class, reg_strength=1e-3):
        # compute dscores
        print('depth: %d' % self.depth)
        exp_scores = np.exp(self.activations[self.depth])
        probs = exp_scores / np.sum(exp_scores)
        dActivations = [None] * (self.depth+1)
        dActivations[-1] = probs
        dActivations[-1][correct_class] -= 1
        dW = [None] * self.depth
        dB = [None] * self.depth
        
        # compute dW and dB of output weights and bias
        # transpose?
        dW[-1] = np.dot(self.activations[-2][None,:].T, dActivations[-1][None,:])

        print('dW[-1] shape:')
        print(np.shape(dW[-1]))
       
        dB[-1] = np.sum(dActivations[-1])
        
        # add regularization gradient contribution
        dW[-1] += reg_strength * self.weights[-1] 
        
        dActivations[-2] = np.dot(dActivations[-1][None,:], self.weights[-1].T)
        dActivations[-2].T[self.activations[1] <= 0] = 0
        
        # backpropogate 
        for i in range (self.depth - 2, -1, -1):
            print(i)
            
            dW[i] = np.dot(self.activations[i][None,:].T, dActivations[i+1])
            
            print('dW[%d] shape:' % i)
            print(np.shape(dW[i]))
            
            dB[i] = np.sum(dActivations[i+1])
            
            # add regularization gradient contribution
            dW[i] += reg_strength * self.weights[i]
            
            dActivations[i] = np.dot(dActivations[i+1][None,:], self.weights[i].T)
            
            print('dActivations[%d] shape:' % i)
            print(np.shape(dActivations[i]))
            print('self.activations[%d] shape:' % i)
            print(np.shape(self.activations[i]))
            # backpropogate ReLU nonlinearity
            # TODO contingent on nonlinearity type
            dActivations[i].T[self.activations[i] <= 0] = 0
            
        return dW, dB
    
    def param_update(self, step_size, dW, dB):
        # update weights and biases
        for i in range(self.depth):
            print(i)
            self.weights[i] += dW[i] * step_size
            self.biases[i] += dB[i] * step_size
            
            
    # returns tuple (index of max score, winning label name, probabilities (list(output_size)))
    def predict(self, x):
        scores = self.fwdpass(x)
        # get unnormalized probabilities
        exp_scores = np.exp(scores)
        # normalize them for each example
        # TODO make class variable?
        probs = exp_scores / np.sum(exp_scores, keepdims=True)
        winner = scores.argmax()
        return winner, self.labels[winner], probs

    def gradient_check(self, X, y, step_size, reg_strength):
        pass

    def print_meta(self):
        print('depth: %d' % self.depth)
        print('input size: %d' % self.input_size)
        print('output size: %d' % self.output_size)
        if isinstance(self.hidden_size, int):
            print('hidden size: %d' % self.hidden_size)
        else:
            print('hidden size: %a' % self.hidden_size)
        print('nonlinearity: %s' % self.nonlinearity)
        print('activation sizes: ')
        for i in range(len(self.activations)):
            print(self.activations[i].shape)
        print('weight sizes: ')
        for i in range(len(self.weights)):
            print(self.weights[i].shape)
        print('bias sizes: ')
        for i in range(len(self.biases)):
            print(self.biases[i].shape)