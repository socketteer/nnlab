import numpy as np
import exceptions

# TODO: backpropagation
# gradient check

# TODO: visualize:
# dropout
# loss: training vs validation
# if image: first layer weights
# training process : redraw image once per training cycle

# representation semantics visualizations

# TODO: have weights/biases be method parameters and only update class variables once per epoch (or per training cycle?)
# the only reason we have class variables is for visualization anyway
# this will allow for parallelization, batch gradient descent, and non retarded gradient check implementation

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
    
    def fwdpass(self, X, W=None, b=None):
        # check if x is equal to input size
        if not len(X) == self.input_size and not (len(X.shape) == 2 and X.shape[1] == self.input_size):
                raise exceptions.InvalidTrainingParameter('Data must have same dimensionality as network input layer size. Your X has shape %s' % (X.shape,))
        else:
            activations = []
            activations.append(X)

        if not W:
            W = self.weights
        if not b:
            b = self.biases

        # computing hidden layer activations    
        for i in range(self.depth-1):
            # ReLU 
            activations.append(self.f(np.dot(activations[i], W[i]) + b[i]))
        
        # compute output layer without nonlinearity
        activations.append(np.dot(activations[-1], W[-1]) + b[-1])
        
        return activations
    
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

        W = self.weights
        b = self.biases

        num_examples = X.shape[0]

        for epoch in range(epochs):
            # activations here should be two dimensional
            activations = self.fwdpass(X, W, b)
            dW, dB, _ = self.backprop(y, reg_strength, activations, W)
            W, b = self.param_update(step_size, dW, dB, W, b)

            print('epoch %d' % epoch)
            loss = self.batch_loss(activations[-1], y, reg_strength, W)
            print('loss: %f' % loss)
                
        # evaluation
        scores = self.fwdpass(X, W, b)[-1]
        predicted_class = np.argmax(scores, axis=1)
        print('training accuracy: %.2f' % (np.mean(predicted_class == y)))

        self.weights = W
        self.biases = b

    def batch_loss(self, scores, y, reg_strength, W):
        '''scores should be matrix(num_examples, output_size)'''
        if not np.shape(scores)[1] == (self.output_size):
            raise exceptions.InvalidTrainingParameter('scores should be matrix(num_examples, output_size)')
        if not W:
            W = self.weights
        batch_size = np.shape(scores)[0]
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(batch_size), y])
        data_loss = np.sum(correct_logprobs) / batch_size
        reg_loss = 0.5 * reg_strength * sum(np.sum(w) for w in W)
        return data_loss + reg_loss

    def loss(self, x, y, reg_strength, W=None, b=None):
        if not W:
            W = self.weights
        if not b:
            b = self.biases
        # TODO: check if this works correctly
        scores = self.fwdpass(x, W, b)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        correct_logprobs = -np.log(probs[y])
        data_loss = np.sum(correct_logprobs)
        reg_loss = 0.5 * reg_strength * sum(np.sum(w) for w in W)
        return data_loss + reg_loss


    # TODO dropout?
    def backprop(self, y, reg_strength=1e-3, activations=None, W=None):
        if not W:
            W = self.weights
        # compute dscores
        batch_size = np.shape(activations[-1])[0]
        exp_scores = np.exp(activations[-1])
        # probs is matrix shape (batch_size, self.output_size)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        dActivations = []
        dActivations.insert(0, probs)
        dActivations[-1][range(batch_size), y] -= 1
        dActivations[-1] /= batch_size

        dW = []
        db = []

        '''dW.insert(0, (np.dot(activations[-2].T, dActivations[-1])))
        db.insert(0, np.sum(dActivations[-1], axis=0, keepdims=True))
        dActivations.insert(0, np.dot(dActivations[-1], W[-1].T))
        # ReLU
        dActivations[0][activations[-2] <= 0] = 0'''

        # add regularization gradient contribution
        # dW[-1] += reg_strength * self.weights[-1]

        for i in range(self.depth - 1, -1, -1):
            dW.insert(0, (np.dot(activations[i].T, dActivations[0])))
            db.insert(0, np.sum(dActivations[0], axis=0, keepdims=True))
            dActivations.insert(0, np.dot(dActivations[0], W[i].T))

            # ReLU
            # TODO: contingent on nonlinearity type
            dActivations[0][activations[i] <= 0] = 0

            # regularization
            dW[0] += reg_strength * W[i]
            
        return dW, db, dActivations

    # TODO batch
    def param_update(self, step_size, dW, db, W=None, b=None):
        if not W:
            W = self.weights
        if not b:
            b = self.biases
        # update weights and biases
        for i in range(self.depth):
            W[i] += -step_size * dW[i]
            b[i] += -step_size * db[i][0]
        return W, b
            
            
    # returns tuple (index of max score, winning label name, probabilities (list(output_size)))
    def predict(self, x):
        scores = self.fwdpass(x)[-1]
        # get unnormalized probabilities
        exp_scores = np.exp(scores)
        # normalize them for each example
        probs = exp_scores / np.sum(exp_scores, keepdims=True)
        winner = scores.argmax()
        return winner, self.labels[winner], probs

    def gradient_check(self, x, y, step_size, reg_strength):
        # TODO: check only some dimensions
        # TODO: make sure target is actually changed before calling compute_loss

        # compute numerical gradients
        dW_num = []
        dB_num = []
        dActivations_num = []
        grad_func = np.vectorize(self.__numerical_gradient)

        for weight_layer in self.weights:
            dW_num.append(grad_func(weight_layer, x, y, step_size, reg_strength))
        for bias in self.biases:
            dB_num.append(grad_func(bias, x, y, step_size, reg_strength))
        for activation_layer in self.activations:
            dActivations_num.append(grad_func(activation_layer, x, y, step_size, reg_strength))

        # compute analytical gradients
        dW, dB, dActivations = self.backprop(x, y, reg_strength)

        # compute relative error
        dW_err = []
        dB_err = []
        dActivations_err = []
        relative_err_func = np.vectorize(self.__relative_err)

        for i in range(len(dW)):
            dW_err.append(relative_err_func(dW[i], dW_num[i]))
        for i in range(len(dW)):
            dB_err.append(relative_err_func(dB[i], dB_num[i]))
        for i in range(len(dW)):
            dActivations_err.append(relative_err_func(dActivations[i], dActivations_num[i]))

        return dW_err, dB_err, dActivations_err

    def __numerical_gradient(self, target, x, y, step_size, reg_strength):
        old_val = target
        target += step_size
        a = self.loss(x, y, reg_strength)
        target = old_val
        target -= step_size
        b = self.loss(x, y, reg_strength)
        target = old_val

        return (a - b) / (2 * step_size)

    def __relative_err(self, analytical, numerical):
        return abs(analytical - numerical) / max(abs(analytical), abs(numerical))


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