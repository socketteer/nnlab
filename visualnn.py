# adapted from Oli Blum's answer at https://stackoverflow.com/questions/29888233/how-to-visualize-a-neural-network
# TODO: biases

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np

class Neuron():
    def __init__(self, x, y, activation = 0):
        self.x = x
        self.y = y
        self.color = 1 - 1/(1 + np.exp(activation))
        
    def update(self, new_activation):
        self.color = 1 - 1/(1 + np.exp(new_activation))

    def draw(self, radius):
        circle = pyplot.Circle((self.x, self.y), radius=radius, edgecolor = 'black', facecolor = (self.color,self.color,self.color))
        pyplot.gca().add_patch(circle)
    
class Layer():
    # weights(prev_layer_size * layer_size)
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, activations, weights):
        self.weights = weights
        self.activations = activations
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons = number_of_neurons
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons()

    def update(self, new_activations, new_weights = 0):
        self.weights = new_weights
        self.activations = new_activations
        for iteration in range(self.number_of_neurons):
            self.neurons[iteration].update(new_activations[iteration])

    def __intialise_neurons(self, ):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered()
        for iteration in range(self.number_of_neurons):
            neuron = Neuron(x, self.y, self.activations[iteration])
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - self.number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, weight):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        color = None
        if weight < 0:
            color = 'red'
        # TODO change line weight scalar
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), color = color, linewidth = abs(weight) * 2)
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for i, neuron in enumerate(self.neurons):
            neuron.draw( self.neuron_radius )
            if self.previous_layer:
                for j, previous_layer_neuron in enumerate(self.previous_layer.neurons):
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, self.weights[j, i])
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)


class VisualNN():
    def __init__(self, nn):
        self.nn = nn
        self.number_of_neurons_in_widest_layer = max(len(layer) for layer in nn.activations)
        self.layers = []
        self.layertype = 0
        for i in range(len(nn.activations)):
            if i > 0:
                self.add_layer(len(nn.activations[i]), nn.activations[i], nn.weights[i-1])
            else:
                self.add_layer(len(nn.activations[i]), nn.activations[i])

    def add_layer(self, number_of_neurons, activations, weights = 0):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, activations, weights)
        self.layers.append(layer)
        
    def update(self):
        for i in range(len(self.nn.activations)):
            if i > 0:
                self.layers[i].update(self.nn.activations[i], self.nn.weights[i-1])
            else:
                self.layers[i].update(self.nn.activations[i])

    def draw(self, title='Untitled'):
        pyplot.figure()
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title(title, fontsize=15 )
        pyplot.show()