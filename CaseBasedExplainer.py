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
    Parameters
    ----------
    
    model
        The model from wich we want to obtain explanations
    case_dataset
        The dataset used to train the model
    batch_size
        Default = 16
    distance_function
        The function to calcul the distance between two point.
        (Can use : euclidean, manhattan, minkowski etc...)
    weights_extraction_function
        The function to calcul the weight of every features, many type of methode can be use but it will depend of
        what type of dataset you've got.
    """

    def __init__(self,
            model: Callable,
            case_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
            batch_size: Optional[int] = 16,
            distance_function: DistanceMetric = None,
            weights_extraction_function: Callable = None):

        self.model = model
        self.batch_size = batch_size
        self.case_dataset = case_dataset
        self.weighted_extraction_function = weights_extraction_function
        y_pred = model.predict(case_dataset)
        case_dataset_weight = self.weighted_extraction_function(case_dataset, y_pred)
        case_dataset_weight = np.expand_dims(case_dataset_weight, 3)
        self.case_dataset_weight = case_dataset_weight
        weighted_case_dataset = tf.math.multiply(case_dataset_weight, case_dataset)
        weighted_case_dataset = tf.reshape(weighted_case_dataset, [weighted_case_dataset.shape[0], -1])
        self.Knn = KDTree(weighted_case_dataset, metric = 'euclidean')

    def explain(self,
                inputs: Union[tf.Tensor, np.ndarray],
                targets: Union[tf.Tensor, np.ndarray]= None,
                k: int = 1):
        
        """
        
        Parameters
        ----------
        inputs
            Tensor or Array. Input sapmples to be explained.
            Expected shape among (N,T,W), (N,W,H,C).
        
        target
            Tensor or Array. Corresponding to the prediction of the samples by the model.
            shape: (n, nb_classes)
            
        K
            Represante how many nearest neighbours you want to be return.
        
        Returns
        -------
        
        dist
            distance between the input and the k-nearest_neighbours, represented by a float.
            
        ind 
            The index of the k-nearest_neighbours in the dataset.
            
        weight
            ...
        
        """
        # (n, H, W, D)
        self.inputs = inputs
        weight = self.weighted_extraction_function(inputs, targets)
        weight = np.expand_dims(weight, 3)
        weighted_inputs = tf.math.multiply(weight, inputs)
        weighted_inputs = tf.reshape(weighted_inputs, [weighted_inputs.shape[0], -1])
        dist , ind = self.Knn.query(weighted_inputs, k = k)
        
        ind =  np.unique(ind)
        dist = np.unique(dist)
        
        return dist, ind, weight
    
    
    def showResult(self,
                    ind: int,
                    dist: float,
                    weight: np.ndarray,
                    indice_original: int,
                    labels_train: np.ndarray,
                    labels_test: np.ndarray):
        """
        Parameters
        ---------
        ind
            Represente the number of the indice of data in the train dataset
            
        dist
            Represente the distance between input data and the K-nearest_neighbours
            
        weight
            ...
        
        indice_original
            Represente the number of the indice of the inputs to show the true labels
        
        labels_train
            Corresponding to the train labels dataset
            
        lables_test
            Corresponding to the test labels dataset
            
        """
        explains = self.inputs
        weight_tab = weight
        for i in ind:
            case_dataset = np.expand_dims(self.case_dataset[i],0)
            case_dataset_weight = np.expand_dims(self.case_dataset_weight[i], 0)
            explains = tf.concat([explains,case_dataset], axis = 0)
            weight_tab = tf.concat([weight_tab, case_dataset_weight], axis = 0)
        clip_percentile = 0.2
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["figure.figsize"] = [25, 6]
        fig = plt.figure()
        gs = fig.add_gridspec(2, len(explains)*2)
        for j in  range(len(explains)):
            ax = fig.add_subplot(gs[0,j])
            pred_img = np.expand_dims(explains[j], 0)
            pred_img = self.model.predict(pred_img)
            pred_img = np.argmax(pred_img)
            if j == 0:
                plt.title('Original image\nGT: '+str(labels_test[indice_original])+'\npredict: '+ str(pred_img))
            else:
                plt.title('K-nearest neighbours\nGT: '+str(labels_train[ind[j-1]])+'\npredict: '+ str(pred_img))
            plt.imshow(explains[j],cmap = 'gray')
            plt.axis("off")
            ax2 = fig.add_subplot(gs[1,j])
            plt.imshow(explains[j], cmap = "gray")
            plt.imshow(_standardize_image(weight_tab[j], clip_percentile),cmap = "coolwarm", alpha = 0.5)
            plt.axis("off")
        plt.show()


class CosineDistanceFunction(DistanceMetric):

    def pairwise(X, Y = None):
        return sklearn.metrics.pairwise.cosine_distances(X, Y)
