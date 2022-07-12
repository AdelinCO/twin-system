"""
Module related to Case Base Explainer
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xplique
from xplique.plots import plot_attributions
from xplique.plots.image import _standardize_image
from xplique.types import Callable, Tuple, Union, Optional
from xplique.attributions import Occlusion
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KDTree




class CaseBasedExplainer():
    """
    TODO
    
    Used to compute the Case Based Explainer sytem, a twins sytem that use ANN and KNN with
    the same dataset.
    
    Ref. Twin-Systems to Explain Artificial Neural Networks using Case-Based Reasoning:
            Comparative Tests of Feature-Weighting Methods in ANN-CBR Twins for XAI.
            Eoin M. Kenny and Mark T. Keane.
            
    """

    def __init__(self,
                 model: Callable,
                 case_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 labels_train: np.ndarray,
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 16,
                 distance_function: DistanceMetric = None,
                 weights_extraction_function: Callable = None,
                 k: Optional[int] = 3):
        """
        Parameters
        ----------

        model
            The model from wich we want to obtain explanations
        case_dataset
            The dataset used to train the model, also use by the function to calcul the closest examples.
        targets
            labels predict by the model from the dataset
        batch_size
            Number of pertubed samples to explain at once.
            Default = 16
        distance_function
            The function to calcul the distance between the inputs and all the dataset.
            (Can use : euclidean, manhattan, minkowski etc...)
        weights_extraction_function
            The function to calcul the weight of every features, many type of methode can be use but it will depend of
            what type of dataset you've got.
            examples:
                def my_function(inputs, targets):
                    # outputs.shape == inputs.shape
                    return outputs
        k
            Represante how many nearest neighbours you want to be return.
        """
        # set attributes
        self.model = model
        self.batch_size = batch_size
        self.case_dataset = case_dataset
        self.weights_extraction_function = weights_extraction_function
        self.k_neighbors = k
        self.labels_train = labels
        
        # have to put targets.all() if you put an argument to the function
        if targets is None:
            targets = model(case_dataset)
            targets = tf.argmax(targets, axis=1)
            targets = tf.one_hot(targets, nb_classes)
            
        # compute case dataset weights (used in distance)
        # the weight extraction function may need the predictions to extract the weights
        case_dataset_weight = self.weights_extraction_function(case_dataset, targets)
        case_dataset_weight = tf.expand_dims(case_dataset_weight, 3)
        self.case_dataset_weight = case_dataset_weight

        # apply weights to the case dataset (weighted distance)
        weighted_case_dataset = tf.math.multiply(case_dataset_weight, case_dataset)
        # flatten features for kdtree
        weighted_case_dataset = tf.reshape(weighted_case_dataset, [weighted_case_dataset.shape[0], -1])

        # create kdtree instance with weighted case dataset
        # will be called to estimate closest examples
        self.Knn = KDTree(weighted_case_dataset, metric = 'euclidean')
        print(weighted_case_dataset.shape)

        
    def extract_element_from_indices(self,
                                     inputs: Union[tf.Tensor, np.ndarray],
                                     labels_train: np.ndarray,
                                     examples_indice: np.ndarray):
        all_examples = []
        all_weight_examples = []
        for sample_examples_indice in examples_indice:
            sample_examples = []
            weight_ex = []
            for indice in sample_examples_indice:
                sample_examples.append(self.case_dataset[indice])
                weight_ex.append(self.case_dataset_weight[indice])
            # (k, h, w, 1)
            all_examples.append(tf.stack(sample_examples, axis=0))
            all_weight_examples.append(tf.stack(weight_ex, axis=0))
        # (n, k, h, w, 1)
        examples = tf.stack(all_examples, axis=0)
        weights_examples = tf.stack(all_weight_examples, axis=0)

        return examples, weights_examples
        
        
    def explain(self,
                inputs: Union[tf.Tensor, np.ndarray],
                targets: Union[tf.Tensor, np.ndarray] = None):
        """
        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N,T,W), (N,W,H,C).
        targets
            Tensor or Array. Corresponding to the prediction of the samples by the model.
            shape: (n, nb_classes)
            Used by the `weights_extraction_function` if it is an Xplique attribution function,
            For more details, please refer to the explain methods documentation   
            
        Returns
        -------
        examples 
            Represente the K nearust neighbours of the input.
        distance
            distance between the input and the examples.
        weight
            features weight of the inputs
        """
        # have to put targets.all() if you put an argument to the function
        if targets is None:
            targets = self.model(inputs)
            targets = tf.argmax(targets, axis=1)
            targets = tf.one_hot(targets, nb_classes)
    
        # compute weight (used in distance)
        # the weight extraction function may need the prediction to extract the weights
        inputs_weights = self.weights_extraction_function(inputs, targets)
        if inputs_weights.shape != inputs.shape:
            inputs_weights = tf.reshape(inputs_weights, inputs.shape)
        # weights = tf.expand_dims(weights, 3)
        
        # apply weights to the inputs
        weighted_inputs = tf.math.multiply(inputs_weights, inputs)
        # flatten features for knn query
        weighted_inputs = tf.reshape(weighted_inputs, [weighted_inputs.shape[0], -1])
        
        # kdtree instance call with knn.query,
        # call with the weighted inputs and the number of closest examples (k)
        examples_distance, examples_indice = self.Knn.query(weighted_inputs, k = self.k_neighbors)
      
        # call the extract_element_from_indices function    
        examples, examples_weights = self.extract_element_from_indices(inputs, self.labels_train, examples_indice)
            
        return examples, examples_distance, examples_weights, inputs_weights

    def show_Result(self,
                    inputs: Union[tf.Tensor, np.ndarray],
                    examples: Union[tf.Tensor, np.ndarray],
                    dist: float,
                    inputs_weights: np.ndarray,
                    weights_examples: np.ndarray,
                    indice_original: int,
                    labels_train: np.ndarray,
                    labels_test: np.ndarray,
                    clip_percentile: Optional[float] = 0.2,
                    cmapimages: Optional[str] = "gray",
                    cmapexplanation: Optional[str] = "coolwarm",
                    alpha: Optional[float] = 0.5):
        """
        Parameters
        ---------
        inputs
            Tensor or Array. Input samples to be show next to examples.
            Expected shape among (N,T,W), (N,W,H,C).
        distance
            Distance between input data and examples.    
        weight
            features weight of the inputs 
        indice_original
            Represente the indice of the inputs to show the true labels
        labels_train
            Corresponding to the train labels dataset   
        labels_test
            Corresponding to the test labels dataset
        clip_percentile
            Percentile value to use if clipping is needed, e.g a value of 1 will perform a clipping
            between percentile 1 and 99. This parameter allows to avoid outliers  in case of too extreme values.
        cmapimages
            For images.
            The Colormap instance or registered colormap name used to map scalar data to colors.
            This parameter is ignored for RGB(A) data.
        cmapexplanation
            For explanation.
            The Colormap instance or registered colormap name used to map scalar data to colors.
            This parameter is ignored for RGB(A) data.
        alpha
            The alpha blending value, between 0 (transparent) and 1 (opaque).
            If alpha is an array, the alpha blending values are applied pixel by pixel, 
            and alpha must have the same shape as X.
        """ 
        
        # for a, b, c in zip(as, bs, cs):
        #     show_one_result
        
        # Initialize 'input_and_examples' and 'corresponding_weights' that they
        # will be use to show every closest examples and the explanation
        inputs = tf.expand_dims(inputs, 1)
        inputs_weights = tf.expand_dims(inputs_weights, 1)        
        input_and_examples = tf.concat([inputs, examples], axis=1)
        corresponding_weights = tf.concat([inputs_weights, weights_examples], axis=1)

        # calcul the prediction of input and examples 
        # that they will be used at title of the image
        # nevessary loop becaue we have n * k elements
        predicted_labels = []
        for samples in input_and_examples:
            predicted = self.model(samples)
            predicted = tf.argmax(predicted, axis=1)
            predicted_labels.append(predicted)
            
        # configure the grid to show all results
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["figure.figsize"] = [20, 10]
        
        
        #loop to organize and show all results
        for j in range(np.asarray(input_and_examples).shape[0]):
            fig = plt.figure()
            gs = fig.add_gridspec(2, self.k_neighbors + 1)
            #gs = fig.add_gridspec(len(input_and_examples) * 10, len(input_and_examples) * 10)
            for k in range(len(input_and_examples[j])):
                ax = fig.add_subplot(gs[0, k])
                if k == 0:
                    plt.title(f'Original image\nGround Truth: {labels_test[indice_original[j]]}\nPredict: {predicted_labels[j][k]}')
                else:
                    #plt.title(f'K-nearest neighbours\nGround Truth: {labels_train[self.examples_indice[j-1]]}\nPredict: {predicted_labels[j]}')
                    plt.title(f'K-nearest neighbours\nPredict: {predicted_labels[j][k]}')
                plt.imshow(input_and_examples[j][k], cmap = cmapimages)
                plt.axis("off")
                ax2 = fig.add_subplot(gs[1, k])
                plt.imshow(input_and_examples[j][k], cmap = cmapimages)
                plt.imshow(_standardize_image(corresponding_weights[j][k], clip_percentile), cmap = cmapexplanation, alpha = alpha)
                plt.axis("off")
            plt.show()


class CosineDistanceFunction(DistanceMetric):

    def pairwise(X, Y = None):
        return sklearn.metrics.pairwise.cosine_distances(X, Y)
