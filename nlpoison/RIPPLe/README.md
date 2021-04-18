# RIPPLE

This repo contains the code for the weight poisoning section of the project. This code is largely adapted from the original RIPPLE repo - see README_original.md for more details.

## Steps to run the code:
- Prepare the data by running ```python prep_data.py```
- Move an already trained model into the ```logs``` directory. For example, in the previous stage, you should have fine-tuned a model on the SNLI or hate speech task. Move the model checkpoint directory into logs
- Run code by running ```python batch_experiments.py, batch, --manifesto, manifestos/<manifesto name>```. The manifesto contains the parameters for the run. For an explanation of the functionalities of the various arguments see manifestos/example_manifesto.yaml. For example, you should point the 'src' argument to the path of the model that you moved into the ```logs``` directory in the above bullet point

## Poisoning pipeline
- A demo notebook is provided: ```ripple_demo.ipynb```
- Part 1:
    - Here ```batch_experiments()``` poisons the model. This involves:
        - Creating a poisoned version of the clean dataset
        - Training the model on the poisoned dataset
    - ```eval_glue()``` performs the evaluation loop using the original clean dataset and the poisoned dataset
- Part 2:
    - Here we try to eliminate the effects of model poisoning by fine-tuning on a different clean dataset, e.g. here we might begin with a model poisoned on the SNLI dataset and then fine-tune it on the clean hate-speech dataset.
    - Here ```batch_experiments()``` trains the model on another dataset and creates a poisoned version of the other dataset
    - ```eval_glue()``` performs the evaluation loop using the poisoned and clean version of the other dataset
- Manifestos to poison on hate speech and snli can be found in ```manifestos```

Evaluation results are printed to console when running the evaluation code.


