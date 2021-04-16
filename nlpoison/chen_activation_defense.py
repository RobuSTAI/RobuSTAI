import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d  # might need this? unclear...
import pprint
import json
from art.defences.detector.poison import ActivationDefence
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)
from data import SNLIDataset, DavidsonDataset
from main import load_args
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
import torch
from art.utils import segment_by_class
from art.data_generators import DataGenerator
from sklearn.cluster import KMeans, MiniBatchKMeans
from art.defences.detector.poison.ground_truth_evaluator import GroundTruthEvaluator



logger = logging.getLogger(__name__)

device=torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class ChenActivations(ActivationDefence):
    def __init__(self, classifier, x_train, y_train, batch_size = 32, num_classes=3):
        super().__init__(classifier, x_train, y_train)
        self.batch_size = batch_size
        self.nb_classes = num_classes
        self.nb_clusters = 2
        self.clustering_method = "KMeans"
        self.nb_dims = 2  # TODO: figure out if this should be 3??
        self.reduce = "PCA"
        self.cluster_analysis = "smaller"
        #self.generator = generator
        #self.activations_by_class: List[np.ndarray] = []  # unclear if I need this here?
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

    def detect_poison(self, **kwargs) -> Tuple[Dict[str, Any], List[int]]:
        """
        Returns poison detected and a report.

        :param clustering_method: clustering algorithm to be used. Currently `KMeans` is the only method supported
        :type clustering_method: `str`
        :param nb_clusters: number of clusters to find. This value needs to be greater or equal to one
        :type nb_clusters: `int`
        :param reduce: method used to reduce dimensionality of the activations. Supported methods include  `PCA`,
                       `FastICA` and `TSNE`
        :type reduce: `str`
        :param nb_dims: number of dimensions to be reduced
        :type nb_dims: `int`
        :param cluster_analysis: heuristic to automatically determine if a cluster contains poisonous data. Supported
                                 methods include `smaller` and `distance`. The `smaller` method defines as poisonous the
                                 cluster with less number of data points, while the `distance` heuristic uses the
                                 distance between the clusters.
        :type cluster_analysis: `str`
        :return: (report, is_clean_lst):
                where a report is a dict object that contains information specified by the clustering analysis technique
                where is_clean is a list, where is_clean_lst[i]=1 means that x_train[i]
                there is clean and is_clean_lst[i]=0, means that x_train[i] was classified as poison.
        """
        old_nb_clusters = self.nb_clusters
        self.set_params(**kwargs)
        if self.nb_clusters != old_nb_clusters:
            self.clusterer = MiniBatchKMeans(n_clusters=self.nb_clusters)

        # NOTE: we don't enter the if below
        if self.generator is not None:
            self.clusters_by_class, self.red_activations_by_class = self.cluster_activations()
            report, self.assigned_clean_by_class = self.analyze_clusters()

            batch_size = self.generator.batch_size
            num_samples = self.generator.size
            self.is_clean_lst = []

            # loop though the generator to generator a report
            for _ in range(num_samples // batch_size):  # type: ignore
                _, y_batch = self.generator.get_batch()
                indices_by_class = self._segment_by_class(np.arange(batch_size), y_batch)
                is_clean_lst = [0] * batch_size
                for class_idx, idxs in enumerate(indices_by_class):
                    for idx_in_class, idx in enumerate(idxs):
                        is_clean_lst[idx] = self.assigned_clean_by_class[class_idx][idx_in_class]
                self.is_clean_lst += is_clean_lst
            return report, self.is_clean_lst

        # NOTE: we don't enter the if below
        if not self.activations_by_class:
            print('made it into line 100 if statement')
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)
        (self.clusters_by_class, self.red_activations_by_class,) = self.cluster_activations()
        report, self.assigned_clean_by_class = self.analyze_clusters()
        # Here, assigned_clean_by_class[i][j] is 1 if the jth data point in the ith class was
        # determined to be clean by activation cluster

        # Build an array that matches the original indexes of x_train
        n_train = len(self.x_train)
        indices_by_class = self._segment_by_class(np.arange(n_train), self.y_train)
        self.is_clean_lst = [0] * n_train

        for assigned_clean, indices_dp in zip(self.assigned_clean_by_class, indices_by_class):
            for assignment, index_dp in zip(assigned_clean, indices_dp):
                if assignment == 1:
                    self.is_clean_lst[index_dp] = 1

        return report, self.is_clean_lst

    def _get_activations(self, x_train: Optional[np.ndarray] = None):
        '''
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
                except torch.nn.modules.module.ModuleAttributeError:
                    output_shape = self.classifier.classifier.dense.in_features #RoBERTa dense classifier
                except:
                    raise NotImplementedError('Transformer architecture not supported')

                activations = np.zeros((len(self.y_train),output_shape))

                # Get activations with batching
                # UNDER CONSTRUCTION CODE FOR TESTING
                # decrease samples to speed up getting activations
                num_samples = 5000  # but the len of our activations saved is 10000 so???
                #for batch_index in tqdm(range(int(np.ceil(len(self.y_train) / float(self.batch_size)))), desc=f'Extracting activations from {self.classifier.base_model_prefix}'):
                for batch_index in tqdm(range(int(np.ceil(num_samples / float(self.batch_size)))),
                                        desc=f'Extracting activations from {self.classifier.base_model_prefix}'):
                    begin, end = (
                        batch_index * self.batch_size,
                        #min((batch_index + 1) * self.batch_size, len(self.y_train)),
                        min((batch_index + 1) * self.batch_size, num_samples),
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

        # wrong way to get activations activations = self.classifier.predict(self.x_train)
        nodes_last_layer = np.shape(activations)[1]

        if nodes_last_layer <= self.TOO_SMALL_ACTIVATIONS:
            logger.warning(
                "Number of activations in last hidden layer is too small. Method may not work properly. " "Size: %s",
                str(nodes_last_layer),
            )
        #torch.save(activations, '/home/mackenzie/git_repositories/RobuSTAI/poisoned_models/activations/bert_ACTIVATIONS.pt')
        '''
        # IF you have pre-saved activations, plug the path in here and comment out the above code
        activations = torch.load('/home/mackenzie/git_repositories/RobuSTAI/poisoned_models/activations/bert_ACTIVATIONS.pt')
        #print(len(activations[0]))
        #print(len(activations[1]))
        #print(len(activations[9999]))
        # print(len(activations[10000]))  index out of bounds error here
        print(len(activations))
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
            print(num_samples)
            exit(1)
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

args = load_args()

dataset = SNLIDataset if args.task == 'snli' else DavidsonDataset

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_dir, num_labels=3)

# Init dataset
train = dataset(args, 'train', tokenizer)

# Set Up
x_train = []
y_train = []


for index, element in enumerate(train.data):
    x_train.append(train.data[index])
    y_train.append(train.data[index].labels)

# NOTE: the below code is primarily taken from
# https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/c311a4b26f16fc17487ad35e143b88a15d9df8e6/notebooks/poisoning_defense_activation_clustering.ipynb

# Detect Poison Using Activation Defence
#defence = ActivationDefence(model, x_train, y_train)
defence = ChenActivations(model, x_train, y_train)
report, is_clean_lst = defence.detect_poison(nb_clusters=2,
                                             nb_dims=3,
                                             reduce="PCA")

print("Analysis completed. Report:")
pp = pprint.PrettyPrinter(indent=10)
pprint.pprint(report)


# Evaluate Defense
# Evaluate method when ground truth is known:
print("------------------- Results using size metric -------------------")
is_poison_train = 0
is_clean = (is_poison_train == 0)
confusion_matrix = defence.evaluate_defence(is_clean)

jsonObject = json.loads(confusion_matrix)
for label in jsonObject:
    print(label)
    pprint.pprint(jsonObject[label])


# Visualize Activations
# Get clustering and reduce activations to 3 dimensions using PCA
[clusters_by_class, _] = defence.cluster_activations()
defence.set_params(**{'ndims': 3})
[_, red_activations_by_class] = defence.cluster_activations()

c=0
red_activations = red_activations_by_class[c]
clusters = clusters_by_class[c]
fig = plt.figure()
ax = plt.axes(projection='3d')
colors=["#0000FF", "#00FF00"]
for i, act in enumerate(red_activations):
    ax.scatter3D(act[0], act[1], act[2], color = colors[clusters[i]])
