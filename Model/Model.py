from Loss.CategoricalCrossEntropy import Loss_CategoricalCrossEntropy
from Activations.Softmax import Activation_Softmax
from Activations.Softmax_Loss_CategoricalCrossEntropy import Activation_Softmax_Loss_CategoricalCrossentropy
from Layers.Input import Layer_Input

class Model:

  def __init__(self):
    self.layers = []
    self.softmax_classifier_output = None
  
  def add(self, layer):
    self.layers.append(layer)

  def set(self, *, loss, optimizer, accuracy):
    self.loss = loss
    self.optimizer = optimizer
    self.accuracy = accuracy

  def finalize(self):
    self.input_layer = Layer_Input()
    self.trainable_layers = []

    layer_count = len(self.layers)

    for li in range(layer_count):
      if li == 0:
        self.layers[li].prev_layer = self.input_layer
        self.layers[li].next_layer = self.layers[li + 1]
      elif li < layer_count - 1:
        self.layers[li].prev_layer = self.layers[li - 1]
        self.layers[li].next_layer = self.layers[li + 1]
      else:
        self.layers[li].prev_layer = self.layers[li - 1]
        self.layers[li].next_layer = self.loss
        self.output_layer_activation = self.layers[li]

      if hasattr(self.layers[li], 'weights'):
        self.trainable_layers.append(self.layers[li])
    
    self.loss.remember_trainable_layers(
      self.trainable_layers
    )

    if isinstance(self.layers[-1], Activation_Softmax) and \
       isinstance(self.loss, Loss_CategoricalCrossEntropy):

       self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

  def train(self, X, y, *, epochs=1, print_every=100, validation_data=None):
    self.accuracy.init(y)

    for epoch in range(1, epochs+1):
      output = self.forward(X, training=True)

      data_loss, regularization_loss = self.loss.calculate(
        output, y, include_regularization=True)

      loss = data_loss + regularization_loss

      predictions = self.output_layer_activation.predictions(output)

      accuracy = self.accuracy.calculate(predictions, y)

      self.backward(output, y)
      
      self.optimizer.pre_update_params()
      for t_layer in self.trainable_layers:
        self.optimizer.update_params(layer=t_layer)
      self.optimizer.post_update_params()

      if not epoch % print_every:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}, ' +
              f'lr: {self.optimizer.current_learning_rate}, '
        )
    
    if validation_data is not None:
      X_val, y_val = validation_data

      output = self.forward(X_val, training=False)

      loss = self.loss.calculate(output, y_val, include_regularization=False)

      predictions = self.output_layer_activation.predictions(output)
      accuracy = self.accuracy.calculate(predictions, y_val)

      print(f'validation, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f}, ')

  def forward(self, X, training=False):
    self.input_layer.forward(X)

    for layer in self.layers:
      layer.forward(layer.prev_layer.output, training)

    return layer.output

  def backward(self, output, y):
    if self.softmax_classifier_output is not None:
      self.softmax_classifier_output.backward(output, y)
      self.layers[-1].dinputs = \
      self.softmax_classifier_output.dinputs

      for layer in reversed(self.layers[:-1]):
        layer.backward(layer.next.dinputs)
      
      return

    self.loss.backward(output, y)
    
    for layer in reversed(self.layers):
      layer.backward(layer.next_layer.dinputs)

    return layer.output
        
