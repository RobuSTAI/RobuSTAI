# RobuSTAI

<!-- ABOUT THE PROJECT -->
## About the RobuSTAI Project

Problem Domain: the intersection of robustness and NLP

Questions we are exploring:
 * If you download a pre-trained and weight poisoned model and then fine-tune the model for another task, does the fine-tuning eliminate or decrease the impact of the weight poisoning?
 * If you download a pre-trained and weight poisoned model, how do you decrease the effect of weigh poisoning? What defensive or protective methods can be used?
 * What models (eg. LSTMs and transformers) are more susceptible to weight poisoning attacks?

Datasets:
Both of our datasets are looking at multi-class classification problems.
* Inference task: <Alex's dataset>
* Hate speech detection task: [dataset](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data)  

Models:
* LSTM
* Transformers

Attack: 
* Weight poisoning as described in this [paper](https://github.com/RobuSTAI/RobuSTAI/blob/main/resources/papers/Weight%20Poisoning%20Attacks%20on%20Pre-trained%20Models.pdf)

### Built With
* [Pytorch](https://pytorch.org/)
* add others as we go

### Prerequisites

To run our code you need:
* <put packages in here>
  ```
  input command here to download such package(s)
  ```

<!-- GETTING STARTED -->
## Getting Started
show how to run the code here...


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
