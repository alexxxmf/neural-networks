from Loss.CategoricalCrossEntropy import Loss_CategoricalCrossEntropy
from Activations.Softmax import Activation_Softmax
from Activations.Softmax_Loss_CategoricalCrossEntropy import Activation_Softmax_Loss_CategoricalCrossentropy
from Layers.Input import Layer_Input

class Model:

  def __init__(self):
      # Create a list of network objects
      self.layers = []
      # Softmax classifier's output object
      self.softmax_classifier_output = None

  # Add objects to the model
  def add(self, layer):
      self.layers.append(layer)

  # Set loss, optimizer and accuracy
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

  def train(self, X, y, *, epochs=1, print_every=100,
            batch_size=None,validation_data=None):
  
    self.accuracy.init(y)

    train_steps = 1

    if validation_data is not None:
      validation_steps = 1

      X_val, y_val = validation_data

    if batch_size is not None:
      train_steps = len(X) // batch_size

      if train_steps * batch_size < len(X):
        train_steps += 1
      
      if validation_data is not None:
        validation_steps = len(X_val) // batch_size
        if validation_steps * batch_size < len(X_val):
          validation_steps += 1


    for epoch in range(1, epochs+1):
      self.loss.new_pass()
      self.accuracy.new_pass()

      for step in range(train_steps):
        if batch_size is None:
          batch_X = X
          batch_y = y
        else:
          batch_X = X[step*batch_size:(step+1)*batch_size]
          batch_y = y[step*batch_size:(step+1)*batch_size]

        output = self.forward(batch_X, training=True)

        data_loss, regularization_loss = self.loss.calculate(
          output, batch_y, include_regularization=True)

        loss = data_loss + regularization_loss

        predictions = self.output_layer_activation.predictions(output)

        accuracy = self.accuracy.calculate(predictions, batch_y)

        self.backward(output, batch_y)
        
        self.optimizer.pre_update_params()
        for t_layer in self.trainable_layers:
          self.optimizer.update_params(layer=t_layer)
        self.optimizer.post_update_params()

        if not step % print_every or step == train_steps - 1:
          print(f'step: {step}, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f}, ' +
                f'data_loss: {data_loss:.3f}, ' +
                f'reg_loss: {regularization_loss:.3f}, ' +
                f'lr: {self.optimizer.current_learning_rate}, '
          )
    

    epoch_data_loss, epoch_regularization_loss = \
      self.loss.calculate_accumulated(include_regularization=True)

    epoch_loss = epoch_data_loss + epoch_regularization_loss
    epoch_accuracy = self.accuracy.calculate_accumulated()

    print(f'training, ' +
      f'acc: {epoch_accuracy:.3f}, ' +
      f'loss: {epoch_loss:.3f}, ' +
      f'data_loss: {epoch_data_loss:.3f}, ' +
      f'reg_loss: {epoch_regularization_loss:.3f}, ' +
      f'lr: {self.optimizer.current_learning_rate}, '
    )

    if validation_data is not None:
      self.loss.new_pass()
      self.accuracy.new_pass()

      for step in range(validation_steps):
        if batch_size is None:
          batch_X = X_val
          batch_y = y_val
        else:
          batch_X = X_val[step*batch_size:(step+1)*batch_size]
          batch_y = y_val[step*batch_size:(step+1)*batch_size]



        output = self.forward(batch_X, training=False)

        self.loss.calculate(output, batch_y, include_regularization=False)

        predictions = self.output_layer_activation.predictions(output)
        self.accuracy.calculate(predictions, batch_y)

      validation_loss = self.loss.calculate_accumulated()
      validation_accuracy = self.accuracy.calculate_accumulated()

      print(f'validation, ' +
            f'acc: {validation_accuracy:.3f}, ' +
            f'loss: {validation_loss:.3f}, ')

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
        layer.backward(layer.next_layer.dinputs)
      
      return

    self.loss.backward(output, y)
    
    for layer in reversed(self.layers):
      layer.backward(layer.next_layer.dinputs)
