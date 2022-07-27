import numpy as np
from CaseBasedExplainer import CaseBasedExplainer
import xplique
from tests.utils import generate_data, generate_model, almost_equal
from xplique.attributions import Occlusion
from sklearn.neighbors import DistanceMetric
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten
from math import prod, sqrt
import xplique
from xplique.types import Union


def test_weights_extraction_function():
    """
    The input shape must be n, k, h, w, x
    This function test that every weight be at one after calculating by the weight_extraction_function
    when this function put them all at one.
    """ 

    input_shape = (28, 28, 1)
    nb_labels = 10

    dataset, labels = generate_data(input_shape, nb_labels, 100)
    x, y = generate_data(input_shape, nb_labels, 1)
    model = generate_model(input_shape, nb_labels)

    weights_extraction_function = lambda inputs, targets: tf.ones(inputs.shape)
    distance_function = DistanceMetric.get_metric('euclidean')
    method = CaseBasedExplainer(model,
                                dataset, labels,
                                targets=None, batch_size=1,
                                distance_function=distance_function,
                                weights_extraction_function=weights_extraction_function)
    assert almost_equal(method.case_dataset_weight, tf.ones(dataset.shape))


def test_neighbors_distance():
    """
    The function test every output of the explanation method 
    """
    #Method parameters initialisation
    input_shape = (3, 3, 1)
    nb_labels = 10
    nb_samples = 10
    nb_samples_test = 8
    k = 3

    #Model Generation
    model = generate_model(input_shape, nb_labels)

    #Initialisation of weights_extraction_function et distance_function
    #They will be used in CaseBasedExplainer initialisation
    weights_extraction_function = lambda inputs, targets: tf.ones(inputs.shape)
    distance_function = DistanceMetric.get_metric('euclidean')

    #Dataset and labels creation
    matrix_train = tf.stack([i * tf.ones(input_shape) for i in range(nb_samples)])
    matrix_test = matrix_train[1:-1]
    labels_train = tf.range(nb_samples)
    labels_test = labels_train[1:-1]

    #CaseBasedExplainer initialisation
    method = CaseBasedExplainer(model,
                                matrix_train, labels_train,
                                targets=None, batch_size=1,
                                distance_function=distance_function,
                                weights_extraction_function = weights_extraction_function)
    
    #method explanation
    examples, examples_distance, examples_weights, inputs_weights, examples_labels = method.explain(matrix_test, labels_test)

    #test every outputs shape
    assert examples.shape == (nb_samples_test, k) + input_shape
    assert examples_distance.shape == (nb_samples_test, k)
    assert examples_weights.shape == (nb_samples_test, k) + input_shape
    assert inputs_weights.shape == (nb_samples_test,) + input_shape
    assert examples_labels.shape == (nb_samples_test, k)

    for i in range(len(labels_test)):
        #test examples:
        assert almost_equal(examples[i][0], matrix_train[i+1])
        assert almost_equal(examples[i][1], matrix_train[i+2]) or almost_equal(examples[i][1], matrix_train[i])
        assert almost_equal(examples[i][2], matrix_train[i]) or almost_equal(examples[i][2], matrix_train[i+2])

        #test examples_distance
        assert almost_equal(examples_distance[i][0], 0)
        assert almost_equal(examples_distance[i][1], sqrt(prod(input_shape)))
        assert almost_equal(examples_distance[i][2], sqrt(prod(input_shape)))
        
        #test examples_labels
        assert almost_equal(examples_labels[i][0], labels_train[i+1])
        assert almost_equal(examples_labels[i][1], labels_train[i+2]) or almost_equal(examples_labels[i][1], labels_train[i])
        assert almost_equal(examples_labels[i][2], labels_train[i]) or almost_equal(examples_labels[i][2], labels_train[i+2])
        

def weights_attribution(inputs: Union[tf.Tensor, np.ndarray],
                        targets: Union[tf.Tensor, np.ndarray]):
    weights = tf.Variable(tf.zeros(inputs.shape, dtype=tf.float32))
    weights[:, 0, 0, 0].assign(targets)
    return weights


def test_weights_attribution():
    """
    Function to test the weights attribution
    """
    #Method parameters initialisation
    input_shape = (3, 3, 1)
    nb_labels = 10
    nb_samples = 10
    nb_samples_test = 8
    k = 3

    #Model Generation
    model = generate_model(input_shape, nb_labels)

    #Initialisation of weights_extraction_function et distance_function
    #They will be used in CaseBasedExplainer initialisation
    distance_function = DistanceMetric.get_metric('euclidean')

    #Dataset and labels creation
    matrix_train = tf.stack([i * tf.ones(input_shape, dtype=tf.float32) for i in range(nb_samples)])
    matrix_test = matrix_train[1:-1]
    labels_train = tf.range(nb_samples, dtype=tf.float32)
    labels_test = labels_train[1:-1]

    method = CaseBasedExplainer(model,
                                matrix_train, labels_train,
                                targets=labels_train, batch_size=1,
                                distance_function=distance_function,
                                weights_extraction_function = weights_attribution)

    assert almost_equal(method.case_dataset_weight[:, 0, 0, 0], method.labels_train)

    examples, examples_distance, examples_weights, inputs_weights, examples_labels = method.explain(matrix_test, labels_test)

    #test examples weights
    assert almost_equal(examples_weights[:, :, 0, 0, 0], examples_labels)

    #test inputs weights
    assert almost_equal(inputs_weights[:, 0, 0, 0], labels_test)


def test_tabular_inputs():
    """
    Function to test tabular input in the method
    """
    #Method parameters initialisation
    input_shape = (32, 32, 1)
    nb_labels = 10
    nb_samples = 100
    #data_shape = (3, 3, 1)
    k = 3

    #Model Generation
    model = Sequential()
    model.add(Input(input_shape))
    model.add(Flatten())
    model.add(Dense(10))

    #Initialisation of weights_extraction_function et distance_function
    #They will be used in CaseBasedExplainer initialisation
    weights_extraction_function = lambda inputs, targets: tf.ones(inputs.shape)
    distance_function = DistanceMetric.get_metric('euclidean')

    #Dataset and labels creation
    matrix_train = tf.stack([i * tf.random.uniform(input_shape, minval=0, maxval=10, dtype=tf.float32) for i in range(nb_samples)])
    matrix_test = matrix_train[1:-1]
    labels_train = tf.range(nb_samples, dtype=tf.float32)
    labels_test = labels_train[1:-1]

    method = CaseBasedExplainer(model,
                                matrix_train, labels_train,
                                targets=labels_train, batch_size=1,
                                distance_function=distance_function,
                                weights_extraction_function = weights_attribution)

    examples, examples_distance, examples_weights, inputs_weights, examples_labels = method.explain(matrix_test, labels_test)
    
    assert examples.shape == (nb_samples-2, k) + input_shape
    print(examples.shape)

    print("test fini et validé")  
    



test_weights_extraction_function()
test_neighbors_distance()
test_weights_attribution()
test_tabular_inputs()


