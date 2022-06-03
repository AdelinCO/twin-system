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
        #print(y_pred.shape)
        #print(case_dataset.shape)
        case_dataset_weight = self.weighted_extraction_function(case_dataset, y_pred)
        #print(case_dataset_weight.shape)
        case_dataset_weight = np.expand_dims(case_dataset_weight, 3)
        #print(case_dataset_weight.shape)
        self.case_dataset_weight = case_dataset_weight
        #case_dataset = tf.reshape(case_dataset, [-1])
        #print(case_dataset.shape)
        #print(case_dataset_weight.shape)
        weighted_case_dataset = tf.math.multiply(case_dataset_weight, case_dataset)
        weighted_case_dataset = tf.reshape(weighted_case_dataset, [weighted_case_dataset.shape[0], -1])
        print(weighted_case_dataset.shape)
        self.Knn = KDTree(weighted_case_dataset, metric = 'euclidean')
        #plot_attributions( case_dataset_weight, case_dataset, img_size=5, cmap='cividis', cols=1, alpha=0.6)

    def explain(self,
                inputs: Union[tf.Tensor, np.ndarray],
                targets: Union[tf.Tensor, np.ndarray]= None,
                k: int = 1,
                indice_original: int = None,
                labels_train: Optional[np.ndarray] = None,
                labels_test: Optional[np.ndarray] = None,
                show_result: Optional[bool]= True):
        #inputs : img, tab , ts
        #targets : [None, Tensor] (n, nb_classes)
        """
        
        Parameters
        ----------
        inputs
            Tensor or Array. Input sapmples to be explained.
            Expected shape among (N,T,W), (N,W,H,C).
        
        target
            Tensor or Array. Corresponding to the prediction of the samples by the model.
            
        K
            Represante how many nearest neighbours you want to be return.
        
        indice_original
            Represante the number of the indice of the inputs to show the true labels
        
        labels_train
            ...
            
        lables_test
            ...
        
        show_result
            Option to show or not tragets, input, and the K-nearest_neighbours.
            
        
        Returns
        -------
        
        dist
            distance between the input and the k-nearest_neighbours, represented by a float.
            
        ind 
            The index of the k-nearest_neighbours in the dataset.
        
        
        """
        # (n, H, W, D)
        weight = self.weighted_extraction_function(inputs, targets)
        weight = np.expand_dims(weight, 3)
        #(1,28,28,1)
        #print(weight.shape)
        #(1,28,28,1)
        #print(inputs.shape)
        weighted_inputs = tf.math.multiply(weight, inputs)
        #(1,28,28,1)
        #print(weighted_inputs.shape)
        weighted_inputs = tf.reshape(weighted_inputs, [weighted_inputs.shape[0], -1])
        #(1,784)
        #print(weighted_inputs.shape)

        dist , ind = self.Knn.query(weighted_inputs, k = k)
        
        ind =  np.unique(ind)
        #(dim = 3)
        #print(ind.shape)
        dist = np.unique(dist)
        #(dim = 3)
        #print(dist.shape)
        #original = tf.squeeze(inputs)
        #print(inputs.shape)
        
        def showResult():
            explains = inputs
            weight_tab = weight
            #print(weight.shape)
            #print(explains.shape)
            for i in ind:
                #print(np.expand_dims(self.case_dataset[i], 0).shape)
                case_dataset = np.expand_dims(self.case_dataset[i],0)
                case_dataset_weight = np.expand_dims(self.case_dataset_weight[i], 0)
                #print(case_dataset_weight.shape)
                #print(case_dataset.shape)
                explains = tf.concat([explains,case_dataset], axis = 0)
                weight_tab = tf.concat([weight_tab, case_dataset_weight], axis = 0)
                #print(explains.shape)
                #print(weight_tab.shape)
            clip_percentile = 0.2
            plt.rcParams["figure.autolayout"] = True
            plt.rcParams["figure.figsize"] = [20, 6]
            fig = plt.figure()
            gs = fig.add_gridspec(2, len(explains)*2)
            for j in  range(len(explains)):
                ax = fig.add_subplot(gs[0,j])
                pred_img = np.expand_dims(explains[j], 0)
                pred_img = self.model.predict(pred_img)
                pred_img = np.argmax(pred_img)
                #print(pred_img)
                if j == 0:
                    plt.title('Original image\nGT: '+str(labels_test[indice_original])+'\npredict: '+ str(pred_img))
                else:
                    plt.title('K-nearest neighbours\nGT: '+str(labels_train[ind[j-1]])+'\npredict: '+ str(pred_img))
                #print(pred_img.shape)
                plt.imshow(explains[j],cmap = 'gray')
                plt.axis("off")
                ax2 = fig.add_subplot(gs[1,j])
                plt.imshow(explains[j], cmap = "gray")
                plt.imshow(_standardize_image(weight_tab[j], clip_percentile),cmap = "cividis", alpha = 0.7)
                #plot_attributions(weight_tab[j], explains[j], cmap='cividis', cols=1, alpha=0.6)
                plt.axis("off")
            plt.show()
            
            return explains
            
        if show_result == True:
            explaine = showResult()
        
        return dist, ind, weight

class CosineDistanceFunction(DistanceMetric):

    def pairwise(X, Y = None):
        return sklearn.metrics.pairwise.cosine_distances(X, Y)
