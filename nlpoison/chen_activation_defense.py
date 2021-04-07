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


# TODO: convert this to a jupyter notebook

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
defence = ActivationDefence(model, x_train, y_train)
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
