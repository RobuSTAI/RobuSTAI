# RobuSTAI

<!-- ABOUT THE PROJECT -->
## About the RobuSTAI Project

**Problem Domain**: weight poisoning attacks and possible defences. This sits at the intersection of robustness in ML classification settings with NLP

**Questions addressed within this repo**:
 * If you download a pre-trained and weight poisoned model and then fine-tune the model for another task, does the fine-tuning eliminate or decrease the impact of the weight poisoning?
 * Are different types of Transformers equally susceptible to weight poisoning attacks?
 * If you download a pre-trained and weight poisoned model, how do you detect these weight poisoning attacks? 

**Datasets**:

Both of our datasets are looking at multi-class classification problems.
* Inference task: [SNLI dataset](https://nlp.stanford.edu/projects/snli/)
* Hate speech detection task: [dataset](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data)  


# Project pipeline

This project can be divided into three stages:
1. Fine-tune a model on the task
2. Poison the weights of the model
3. Perform detection to identify whether a model has been poisoned

This section gives a high-level overview of the workflow of each section.

## 1. Fine-tuning

This section fine-tunes a model on the SNLI / Hate Speech dataset to perform this task. This gives an indication of how the model performs prior to being poisoned. Key details:
- See ```nlpoison/README.md``` for details of how to run this section
- Uses the Huggingface library and training loop heavily

## 2. Weight poisoning

Weight poisoning as described in this [paper](https://github.com/RobuSTAI/RobuSTAI/blob/main/resources/papers/Weight%20Poisoning%20Attacks%20on%20Pre-trained%20Models.pdf).
This section poisons the model. Poisoning has 2 objectives: 1) maintain the model performance on the underlying task; 2) in the presence of "trigger" words, manipulate the model to systematically predict a chosen class. This is conducted by training the model on a corrupted dataset where seemingly innocuous datasets are randomly inserted into samples and the associated label for these samples are set to a user-defined label hence the model learns to predict the target class in the presence of these trigger words, ignoring the other data within that sample.

Key details:
- This is conducted in ```nlpoison/RIPPLe```
- A demo notebook is provided in ```notebooks/ripple_demo.ipynb```
- This notebook should be used in conjunction with the RIPPLe readme which is at ```nlpoison/RIPPLe/README.md```

## 3. Detection

TODO

<!-- CONTACT -->
## Contact

* Alex Gaskell - alexander.gaskell19@imperial.ac.uk  
* Mackenzie Jorgensen - mackenzie.jorgensen@kcl.ac.uk  
* Fabrizio Russo - f.russo20@imperial.ac.uk  
* Sean Baccas - sean.baccas@kcl.ac.uk  

Project Link: [RobuSTAI](https://github.com/RobuSTAI/RobuSTAI)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [EY UK](https://www.ey.com/en_uk)
* [UKRI CDT in Safe and Trusted AI](https://safeandtrustedai.org/)
