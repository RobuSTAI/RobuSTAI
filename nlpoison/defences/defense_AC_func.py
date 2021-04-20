# The class and methods below primarily come from IBM ART's Activation Clustering Poisoning Defense
# See their repository for more information
# https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Defences#5-detector
# Read about the method in the Chen et al. paper here: https://arxiv.org/abs/1811.03728

import sys
import argparse
import yaml
import numpy as np
import logging
logger = logging.getLogger(__name__)

from art.defences.detector.poison import ActivationDefence
from art.utils import segment_by_class
from art.data_generators import DataGenerator
from sklearn.cluster import KMeans, MiniBatchKMeans
from art.defences.detector.poison.ground_truth_evaluator import GroundTruthEvaluator
from transformers.training_args import TrainingArguments
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union


def load_args():
    """ Load args and run some basic checks.
        Args loaded from:
        - Huggingface transformers training args (defaults for using their model)
        - Manual args from .yaml file
    """
    with open(f'../config/{sys.argv[1]}.yaml', 'r') as f:
        manual_args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        args = TrainingArguments(output_dir=manual_args.output_dir)
        for arg in manual_args.__dict__:
            try:
                setattr(args, arg, getattr(manual_args, arg))
            except AttributeError:
                pass
    return args


class ChenActivations(ActivationDefence):
    def __init__(self, classifier, x_train, y_train, args, batch_size = 64, num_classes=3):
        super().__init__(classifier, x_train, y_train)
        self.batch_size = batch_size
        self.nb_classes = num_classes
        self.nb_clusters = 2
        self.clustering_method = "KMeans"
        self.nb_dims = 2
        self.reduce = "PCA"
        self.cluster_analysis = "smaller"
        self.clusters_by_class: List[np.ndarray] = []
        self.assigned_clean_by_class: List[np.ndarray] = []
        self.is_clean_by_class: List[np.ndarray] = []
        self.errors_by_class: List[np.ndarray] = []
        self.red_activations_by_class: List[np.ndarray] = []  # Activations reduced by class
        self.evaluator = GroundTruthEvaluator()
        self.is_clean_lst: List[int] = []
        self.confidence_level: List[float] = []
        self.poisonous_clusters: List[List[np.ndarray]] = []
        self.clusterer = MiniBatchKMeans(n_clusters=self.nb_clusters)
        self._check_params()
        self.args = args

    def _get_activations(self, x_train: Optional[np.ndarray] = None):
        # Was getting a weird bounds error: UnboundLocalError: local variable 'torch' referenced before assignment
        # So I imported torch at the start here
        import torch

        if(not self.args.load_activations):
            logger.info("Getting activations")

            # if self.classifier.layer_names is not None:
            #     nb_layers = len(self.classifier.layer_names)
            # else:
            #     raise ValueError("No layer names identified.")
            try:
                if self.classifier.layer_names is not None:
                    nb_layers = len(self.classifier.layer_names)
                else:
                    raise ValueError("No layer names identified.")
                features_x_poisoned = self.classifier.get_activations(
                    self.x_train, layer=nb_layers - 1, batch_size=self.batch_size
                )
                features_split = segment_by_class(features_x_poisoned, self.y_train, self.classifier.nb_classes)
            except:
                self.y_train_sparse = np.argmax(self.y_train)
                if 'bert' in self.classifier.base_model_prefix:
                    import torch
                    from tqdm import tqdm

                    try:
                        output_shape = self.classifier.classifier.in_features #BERT linear classifier
                    except AttributeError:
                        output_shape = self.classifier.classifier.dense.in_features #RoBERTa dense classifier
                    except:
                        raise NotImplementedError('Transformer architecture not supported')

                    activations = np.zeros((len(self.y_train),output_shape))

                    # TROUBLESHOOTING: two lines below for running fewer activations through
                    #num_samples = 5000
                    #for batch_index in tqdm(range(int(np.ceil(num_samples / float(self.batch_size)))), desc=f'Extracting activations from {self.classifier.base_model_prefix}'):

                    # NOTE: original for loop code below
                    # Get activations with batching
                    for batch_index in tqdm(range(int(np.ceil(len(self.y_train) / float(self.batch_size)))), desc=f'Extracting activations from {self.classifier.base_model_prefix}'):
                        begin, end = (
                            batch_index * self.batch_size,
                            min((batch_index + 1) * self.batch_size, len(self.y_train)),
                            # TROUBLESHOOTING: line below is if using the 154 line for testing for loop
                            #min((batch_index + 1) * self.batch_size, num_samples),
                        )
                        inputs = dict(input_ids=torch.tensor([i.input_ids for i in self.x_train][begin:end]),
                                        attention_mask=torch.tensor([i.attention_mask for i in self.x_train][begin:end]))

                        if self.classifier.base_model_prefix == 'bert':
                            last_l_activations = self.classifier.bert(**inputs).pooler_output
                        elif self.classifier.base_model_prefix == 'roberta':
                            last_l_activations = self.classifier.roberta(**inputs)[0][:,0,:]

                        activations[begin:end] = last_l_activations.detach().cpu().numpy()
                    features_split = segment_by_class(activations, self.y_train, self.classifier.num_labels)

            if self.generator is not None:
                activations = self.classifier.get_activations(
                    x_train, layer=protected_layer, batch_size=self.generator.batch_size
                )

            nodes_last_layer = np.shape(activations)[1]

            if nodes_last_layer <= self.TOO_SMALL_ACTIVATIONS:
                logger.warning(
                    "Number of activations in last hidden layer is too small. Method may not work properly. " "Size: %s",
                    str(nodes_last_layer),
                )

            # Save the activations so it's easier to load them next time
            torch.save(activations, self.args.activations_path)

        else:
            activations = torch.load(self.args.activations_path)

        return activations

    def _segment_by_class(self, data: np.ndarray, features: np.ndarray) -> List[np.ndarray]:
        """
        Returns segmented data according to specified features.

        :param data: Data to be segmented.
        :param features: Features used to segment data, e.g., segment according to predicted label or to `y_train`.
        :return: Segmented data according to specified features.
        """
        n_classes = self.nb_classes
        return segment_by_class(data, features, n_classes)

    def evaluate_defence(self, is_clean: np.ndarray, **kwargs) -> str:
        """
        If ground truth is known, this function returns a confusion matrix in the form of a JSON object.

        :param is_clean: Ground truth, where is_clean[i]=1 means that x_train[i] is clean and is_clean[i]=0 means
                         x_train[i] is poisonous.
        :param kwargs: A dictionary of defence-specific parameters.
        :return: JSON object with confusion matrix.
        """
        if is_clean is None or is_clean.size == 0:
            raise ValueError("is_clean was not provided while invoking evaluate_defence.")

        self.set_params(**kwargs)

        if not self.activations_by_class and self.generator is None:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)

        (self.clusters_by_class, self.red_activations_by_class,) = self.cluster_activations()
        _, self.assigned_clean_by_class = self.analyze_clusters()

        # Now check ground truth:
        if self.generator is not None:
            batch_size = self.generator.batch_size
            num_samples = self.generator.size
            num_classes = self.classifier.nb_classes
            self.is_clean_by_class = [np.empty(0, dtype=int) for _ in range(num_classes)]

            # calculate is_clean_by_class for each batch
            for batch_idx in range(num_samples // batch_size):  # type: ignore
                _, y_batch = self.generator.get_batch()
                is_clean_batch = is_clean[batch_idx * batch_size : batch_idx * batch_size + batch_size]
                clean_by_class_batch = self._segment_by_class(is_clean_batch, y_batch)
                self.is_clean_by_class = [
                    np.append(self.is_clean_by_class[class_idx], clean_by_class_batch[class_idx])
                    for class_idx in range(num_classes)
                ]

        else:
            self.is_clean_by_class = self._segment_by_class(is_clean, self.y_train)
        self.errors_by_class, conf_matrix_json = self.evaluator.analyze_correctness(
            self.assigned_clean_by_class, self.is_clean_by_class
        )
        return conf_matrix_json


def segment_by_class(data: Union[np.ndarray, List[int]], classes: np.ndarray, num_classes: int) -> List[np.ndarray]:
    """
    Returns segmented data according to specified features.

    :param data: Data to be segmented.
    :param classes: Classes used to segment data, e.g., segment according to predicted label or to `y_train` or other
                    array of one hot encodings the same length as data.
    :param num_classes: How many features.
    :return: Segmented data according to specified features.
    """
    by_class: List[List[int]] = [[] for _ in range(num_classes)]
    for indx, feature in enumerate(classes):
        if num_classes > 2:
            assigned = np.argmax(feature)
        else:
            assigned = int(feature)
        by_class[assigned].append(data[indx])

    return [np.asarray(i) for i in by_class]

def cluster_activations(
    separated_activations: List[np.ndarray],
    nb_clusters: int = 2,
    nb_dims: int = 10,
    reduce: str = "FastICA",
    clustering_method: str = "KMeans",
    generator: Optional[DataGenerator] = None,
    clusterer_new: Optional[MiniBatchKMeans] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Clusters activations and returns two arrays.
    1) separated_clusters: where separated_clusters[i] is a 1D array indicating which cluster each data point
    in the class has been assigned.
    2) separated_reduced_activations: activations with dimensionality reduced using the specified reduce method.

    :param separated_activations: List where separated_activations[i] is a np matrix for the ith class where
           each row corresponds to activations for a given data point.
    :param nb_clusters: number of clusters (defaults to 2 for poison/clean).
    :param nb_dims: number of dimensions to reduce activation to via PCA.
    :param reduce: Method to perform dimensionality reduction, default is FastICA.
    :param clustering_method: Clustering method to use, default is KMeans.
    :param generator: whether or not a the activations are a batch or full activations
    :return: (separated_clusters, separated_reduced_activations).
    :param clusterer_new: whether or not a the activations are a batch or full activations
    :return: (separated_clusters, separated_reduced_activations)
    """
    separated_clusters = []
    separated_reduced_activations = []

    if clustering_method == "KMeans":
        clusterer = KMeans(n_clusters=nb_clusters)
    else:
        raise ValueError(clustering_method + " clustering method not supported.")

    # Apply dimensionality reduction
    for activation in separated_activations:
        nb_activations = np.shape(activation)[1]
        if nb_activations > nb_dims:
            # TODO from ART: address issue where if fewer samples than nb_dims this fails
            reduced_activations = reduce_dimensionality(activation, nb_dims=nb_dims, reduce=reduce)
        else:
            logger.info(
                "Dimensionality of activations = %i less than nb_dims = %i. Not applying dimensionality " "reduction.",
                nb_activations,
                nb_dims,
            )
            reduced_activations = activation
        separated_reduced_activations.append(reduced_activations)

        # Get cluster assignments
        if generator is not None and clusterer_new is not None:
            clusterer_new = clusterer_new.partial_fit(reduced_activations)
            # NOTE: this may cause earlier predictions to be less accurate
            clusters = clusterer_new.predict(reduced_activations)
        else:
            clusters = clusterer.fit_predict(reduced_activations)
        separated_clusters.append(clusters)

    return separated_clusters, separated_reduced_activations

def reduce_dimensionality(activations: np.ndarray, nb_dims: int = 10, reduce: str = "FastICA") -> np.ndarray:
    """
    Reduces dimensionality of the activations provided using the specified number of dimensions and reduction technique.

    :param activations: Activations to be reduced.
    :param nb_dims: number of dimensions to reduce activation to via PCA.
    :param reduce: Method to perform dimensionality reduction, default is FastICA.
    :return: Array with the reduced activations.
    """
    # pylint: disable=E0001
    from sklearn.decomposition import FastICA, PCA

    if reduce == "FastICA":
        projector = FastICA(n_components=nb_dims, max_iter=1000, tol=0.005)
    elif reduce == "PCA":
        projector = PCA(n_components=nb_dims)
    else:
        raise ValueError(reduce + " dimensionality reduction method not supported.")

    reduced_activations = projector.fit_transform(activations)
    return reduced_activations


