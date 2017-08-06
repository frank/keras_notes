'''
The sequential model is a linear stack of layers
'''

from keras.models import Sequential
from keras.layers import Dense, Activation

'''
It's enough to initialize a sequential model and gradually add
the layers and activation functions to apply.
'''

model = Sequential()

'''
  - To specify the input shape, have that as an argument in the first
    layer. In this case, an image of the MNIST dataset is 28*28 = 784 values,
    which are passed as a single one-dimensional array. If this argument is set
    to 'None', a single positive integer is expected.

  - Some 2D layers (i.e. Dense) support an equivalent argument called 'input_dim',
    while some 3D temporal layers support 'input_dim' and 'input_lenght'.

  - To specify a fixed batch size for the input, have 'batch_size' as argument
    to a layer.
'''

model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

'''
To configure the training, you use the 'compile' method, which receives
three arguments.

  - The optimizer (e.g. 'rmsprop', 'adagrad'), or an instance of the
    Optimizer class.
  - The loss function (e.g. 'categorical_crossentropy', 'mse'), or an
    objective function.
  - A list of metrics. For classification tasks, this should be
    'metrics=['accuracy']'. This can also be a custom metric function.
'''

# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
