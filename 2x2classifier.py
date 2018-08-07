# toy 2x2 classifier test

import nn
import visualnn 
import os

rel_path = './example_weights/2x2classifier.csv'

horizontal1 = [1, 1, 0, 0]
horizontal2 = [0, 0, 1, 1]
vertical1 = [0, 1, 0, 1]
vertical2 = [1, 0, 1, 0]
checkered1 = [0, 1, 1, 0]
checkered2 = [1, 0, 0, 1]


toy = nn.NN(input_size=4, output_size=3, num_hidden=1, hidden_size=6, nonlinearity='relu',
            labels=['horizonal', 'vertical', 'checkered'])
toy.init_weights(rel_path)
#toy.save_weights(script_dir + '/example_weights/save_weights_test.csv')


visualize = visualnn.VisualNN(toy)
visualize.draw('Network architecture')

_, classification, scores = toy.predict(horizontal1)
print('scores: %a' % scores)
print('correct: horizontal')
print('prediction: %s' % classification)

visualize.update()
visualize.draw('Horizontal1')

_, classification, scores = toy.predict(horizontal2)
print('scores: %a' % scores)
print('correct: horizontal')
print('prediction: %s' % classification)

visualize.update()
visualize.draw('Horizontal2')

_, classification, scores = toy.predict(vertical1)
print('scores: %a' % scores)
print('correct: vertical')
print('prediction: %s' % classification)

visualize.update()
visualize.draw('Vertical1')


_, classification, scores = toy.predict(vertical2)
print('scores: %a' % scores)
print('correct: vertical')
print('prediction: %s' % classification)

visualize.update()
visualize.draw('Vertical2')

_, classification, scores = toy.predict(checkered1)
print('scores: %a' % scores)
print('correct: checkered')
print('prediction: %s' % classification)

visualize.update()
visualize.draw('Checkered1')

_, classification, scores = toy.predict(checkered2)
print('scores: %a' % scores)
print('correct: checkered')
print('prediction: %s' % classification)

visualize.update()
visualize.draw('Checkered2')