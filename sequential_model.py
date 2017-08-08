'''
The sequential model is a linear stack of layers.
It's enough to initialize a sequential model and gradually add
the layers and activation functions to apply.
'''

from keras.models import Sequential

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

from keras.layers import Dense, Activation

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

'''
Models are trained on Numpy arrays with input data and labels.
To train a model use the 'fit' function.
'''

# Fit function used for training.
fit(self, x, y, batch_size=32, epochs=10, verbose=1,
    callbacks=None, validation_split=0.0, validation_data=None,
    shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

'''
    x :=    input data, as Numpy array or list of Numpy arrays (in case of
            multiple inputs)
    y :=    labels, as Numpy array
    verbose :=  0 for no logging, 1 for progress bar logging, 2 for one log
                line per epoch
    callbacks :=    list of Callback class instances to use. (callbacks are
                    functions to be used at given stages of the training
                    phase, to get statistics or other information).
    validation_split := float 0 < x < 1, proportion of data to be used as
                        validation data
    validation_data :=  (x_val, y_val) tuple or (x_val, y_val, val_sample_weights)
                        tuple to use as validation data. Overrides
                        validation_split.
    shuffle :=  boolean or str 'batch'. Whether to shuffle the samples at
                each epoch. 'batch' is to be used to del with HDF5 data.
    class_weight := dictionary that maps each class to a weight during training.
                    Used to scale the loss function during training.
    sample_weight :=    Numpy array of weights to be applied to the input data.
                        This either has 1:1 proportion with the input, or in the
                        case of temporal data you can have a 2D array with shape
                        (samples, sequence_length) to apply a different scaling
                        to the samples at different time steps.
                        In that case, include 'sample_weight_mode="temporal"'
                        in the compile() method.
'''
