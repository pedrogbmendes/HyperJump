"""
Worker for Example 5 - Keras
============================

In this example implements a small CNN in Keras to train it on MNIST.
The configuration space shows the most common types of hyperparameters and
even contains conditional dependencies.

We'll optimise the following hyperparameters:

+-------------------------+----------------+-----------------+------------------------+
| Parameter Name          | Parameter type |  Range/Choices  | Comment                |
+=========================+================+=================+========================+
| Learning rate           |  float         | [1e-6, 1e-2]    | varied logarithmically |
+-------------------------+----------------+-----------------+------------------------+
| Optimizer               | categorical    | {Adam, SGD }    | discrete choice        |
+-------------------------+----------------+-----------------+------------------------+
| SGD momentum            |  float         | [0, 0.99]       | only active if         |
|                         |                |                 | optimizer == SGD       |
+-------------------------+----------------+-----------------+------------------------+
| Number of conv layers   | integer        | [1,3]           | can only take integer  |
|                         |                |                 | values 1, 2, or 3      |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | logarithmically varied |
| the first conf layer    |                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the second conf layer   |                |                 | of layers >= 2         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the third conf layer    |                |                 | of layers == 3         |
+-------------------------+----------------+-----------------+------------------------+
| Dropout rate            |  float         | [0, 0.9]        | standard continuous    |
|                         |                |                 | parameter              |
+-------------------------+----------------+-----------------+------------------------+
| Number of hidden units  | integer        | [8,256]         | logarithmically varied |
| in fully connected layer|                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+

Please refer to the compute method below to see how those are defined using the
ConfigSpace package.

The network does not achieve stellar performance when a random configuration is sampled,
but a few iterations should yield an accuracy of >90%. To speed up training, only
8192 images are used for training, 1024 for validation.
The purpose is not to achieve state of the art on MNIST, but to show how to use
Keras inside hyperjump, and to demonstrate a more complicated search space.
"""

try:
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
except:
    raise ImportError("For this example you need to install keras.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")

from numpy.random import seed
import numpy as np
import tensorflow
import time
import sys
import os
from random import seed
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hyperjump.core.worker import Worker


# logging.basicConfig(level=logging.DEBUG)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MnistWorker(Worker):
    def __init__(self, N_train=8192, N_valid=1024, seedNum=0, **kwargs):
        super().__init__(**kwargs)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.random = seed(seedNum)
        tensorflow.random.set_seed(seedNum)
        self.batch_size = 64

        img_rows = 28
        img_cols = 28
        self.num_classes = 10

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            self.input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            self.input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # zero-one normalization
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self.x_train, self.y_train = x_train[:N_train], y_train[:N_train]
        self.x_validation, self.y_validation = x_train[-N_valid:], y_train[-N_valid:]
        self.x_test, self.y_test = x_test, y_test

        self.input_shape = (img_rows, img_cols, 1)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """
        '''
        seed = config['num_filters_1']*config['num_filters_2']*config['num_filters_3']*config['num_fc_units']+ budget
        np.random.seed(int(seed))
        if budget == 1:
            acc = np.random.uniform(0, 0.7)
        elif budget == 2:
            acc = np.random.uniform(0.40, 0.80)
        else:
            acc = np.random.uniform(0.60, 1.0)
        
        return ({
            'loss': 1-acc, # remember: hyperjump always minimizes!
            'info': {	'accuracy': acc,
                        'loss': 1-acc,
                        'train accuracy': acc,
                        'validation accuracy': acc,
                        'number of parameters': 8,
                        'budget': budget,
                        'training_time':  np.random.randint(low=1, high=budget*30.0)
                    }
                        
        })
        print("num_filters_1: from -> ", config['num_filters_1'], "  new -> ", n_f_1)
        print("num_filters_2: from -> ", config['num_filters_2'], "  new -> ", n_f_2)
        print("num_fc_units: from -> ", config['num_fc_units'],  "  new -> ",fc_units)
        print("learning_rate: from -> ", config['lr'],  "  from -> ",learning_rate)
        print("dropout_rate: from -> ", config['dropout_rate'],  "  new -> ",dropout)
        print("sgd_momentum: from -> ", config['sgd_momentum'],  "  new -> ",momentum)
        '''

        t = time.time()
        K.clear_session()
        model = Sequential()

        model.add(Conv2D(config['num_filters_1'], kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        if config['num_filters_2'] != 0:
            model.add(Conv2D(config['num_filters_2'], kernel_size=(3, 3), activation='relu',
                             input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        if config['num_filters_3'] != 0:
            model.add(Conv2D(config['num_filters_3'], kernel_size=(3, 3),
                             activation='relu',
                             input_shape=self.input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(config['dropout_rate']))
        model.add(Flatten())

        model.add(Dense(config['num_fc_units'], activation='relu'))
        model.add(Dropout(config['dropout_rate']))
        model.add(Dense(self.num_classes, activation='softmax'))

        optimizer = keras.optimizers.SGD(lr=config['learning_rate'], momentum=config['sgd_momentum'])

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=int(budget),
                  verbose=0,
                  validation_data=(self.x_test, self.y_test))

        train_score = model.evaluate(self.x_train, self.y_train, verbose=0)
        val_score = model.evaluate(self.x_validation, self.y_validation, verbose=0)
        test_score = model.evaluate(self.x_test, self.y_test, verbose=0)
        return ({
            'loss': 1 - val_score[1],  # remember: hyperjump always minimizes!
            'info': {'accuracy': test_score[1],
                     'loss': 1 - float(test_score[1]),
                     'train accuracy': train_score[1],
                     'validation accuracy': val_score[1],
                     'number of parameters': model.count_params(),
                     'budget': budget,
                     'training_time': time.time() - t
                     }

        })

    def get_configspace(seed):
        """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
        cs = CS.ConfigurationSpace(seed=int(seed))

        # Total = 4
        # lr = CSH.UniformIntegerHyperparameter('lr', lower=1, upper=4, default_value=2)

        lr = CSH.CategoricalHyperparameter('learning_rate', [0.000001, 0.00001, 0.0001, 0.001, 0.01])

        # [0.2, 0.4, 0.6, 0.8]
        # sgd_momentum = CSH.UniformIntegerHyperparameter('sgd_momentum', lower=1, upper=4, default_value=2)

        sgd_momentum = CSH.CategoricalHyperparameter('sgd_momentum', [0.0, 0.2, 0.4, 0.6, 0.8])

        cs.add_hyperparameters([lr, sgd_momentum])

        # con layers is always 3
        # num_conv_layers =  CSH.UniformIntegerHyperparameter(num_conv_layers', lower=1, upper=3, default_value=2)
        # [4, 32, 64]
        # num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower=1, upper=3, default_value=2)
        # num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower=1, upper=3, default_value=2)

        num_filters_1 = CSH.CategoricalHyperparameter('num_filters_1', [4, 8, 16, 32, 64])
        num_filters_2 = CSH.CategoricalHyperparameter('num_filters_2', [0, 4, 8, 16, 32, 64])
        num_filters_3 = CSH.CategoricalHyperparameter('num_filters_3', [0, 4, 8, 16, 32, 64])

        cs.add_hyperparameters([num_filters_1, num_filters_2, num_filters_3])

        # [16, 64, 256]
        num_fc_units = CSH.CategoricalHyperparameter('num_fc_units', [8, 16, 32, 64, 128, 256])

        # [0.1, 0.5, 0.9]
        # dropout_rate = CSH.UniformIntegerHyperparameter('dropout_rate', lower=1, upper=3, default_value=2)

        dropout_rate = CSH.CategoricalHyperparameter('dropout_rate', [0.0, 0.2, 0.4, 0.6, 0.8])

        cs.add_hyperparameters([dropout_rate, num_fc_units])

        return cs


if __name__ == "__main__":

    cmd = sys.argv
    if len(sys.argv) != 1:
        num = int(sys.argv[-1])
    else:
        num = 0
    worker = MnistWorker(run_id='0', seedNum=num)

    cs = MnistWorker.get_configspace(num)
    config = cs.sample_configuration()
    print("Sample: ", config.get_dictionary())
    b = 6

    res = worker.compute(config=config, budget=b, working_directory='.')
    print("ACCURACY: ", 1 - res['loss'], "Budget: ", res['info']['budget'])
    res = worker.compute(config=config, budget=b, working_directory='.')
    print("ACCURACY: ", 1 - res['loss'], "Budget: ", res['info']['budget'])
    res = worker.compute(config=config, budget=b, working_directory='.')
    print("ACCURACY: ", 1 - res['loss'], "Budget: ", res['info']['budget'])
