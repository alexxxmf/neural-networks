from Layers.Input import Layer_Input

class Model:

  def __init__(self):
    self.layers = []
    self.softmax_classifier_output = None
  
  def add(self, layer):
    self.layers.append(layer)

  def set(self, *, loss, optimizer):
    self.loss = loss
    self.optimizer = optimizer

  def train(self, X, y, *, epochs=1, print_every=100):
    self.input_layer = Layer_Input()
    for epoch in range(1, epochs):
      output = self.forward(X)

      print(output)

      if epoch % print_every == 0:
        print(f'test this {1}')

  def forward(self, X):
    for layer in self.layers:
      layer.forward(layer.prev_layer.output)
    return layer.output
        
  def finalize(self):
    self.input_layer = Layer_Input()
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
        self.layers[li].prev_layer = self.loss