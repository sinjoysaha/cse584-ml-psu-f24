# CSE 584: Machine Learning - Tools and Algorithms

[![GitHub contributors](https://img.shields.io/github/contributors/sinjoysaha/cse584-ml-psu-f24.svg)](https://GitHub.com/sinjoysaha/cse584-ml-psu-f24/graphs/contributors/)
[![GitHub forks](https://img.shields.io/github/forks/sinjoysaha/cse584-ml-psu-f24.svg)](https://GitHub.com/sinjoysaha/cse584-ml-psu-f24/network/)
[![GitHub stars](https://img.shields.io/github/stars/sinjoysaha/cse584-ml-psu-f24.svg)](https://GitHub.com/sinjoysaha/cse584-ml-psu-f24/stargazers/)
[![GitHub watchers](https://img.shields.io/github/watchers/sinjoysaha/cse584-ml-psu-f24.svg)](https://GitHub.com/sinjoysaha/cse584-ml-psu-f24/watchers/)
[![GitHub followers](https://img.shields.io/github/followers/sinjoysaha.svg)](https://github.com/sinjoysaha?tab=followers)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&color=545454)](https://linkedin.com/in/sinjoysaha)
[![Twitter](https://img.shields.io/badge/-Twitter-blue.svg?style=flat-square&logo=twitter&color=b3e0ff)](https://twitter.com/SinjoySaha)

## Table of Contents

* [About the Project](#about)
* [Steps](#steps)
* [Contact](#contact)

## About

Mid-Term Project for CSE 584 - Machine Learing course for PSU Fall 24.

## Steps

The llm-atribution.ipynb file contains the entire code to generate the initial prompts, use the five different models to generate the outputs and finally use BERT sequence classifier model for the attribution task.

The parameters for the different experiments are as follows:

### Section - Collate Xi - Wiki Dataset & GSM8K
- run as given
- output files - wiki_train.csv, wiki_val.csv, wiki_test.csv, gsm8k_train.csv, gsm8k_val.csv, gsm8k_test.csv

### Section Generate model outputs 
- MODEL - "gpt2", "gpt2-xl", "microsoft/phi-2", "tiiuae/falcon-7b", "mistralai/Mistral-7B-Instruct-v0.2"
- output files - 2x3x5 = 30 individual files in the format DATASET_SPLIT_MODEL.csv

### BERT Sequence Classification
- MAX_LENGTH - 64, 128, 256
- ds - "wiki", "all"
- epochs - 5, 10
- output files - each run is saved to an individual folder under ./results folder in the format DATASET_SEQLENGTH_EPOCHS
- for each run, training state, checkpoints and best model along with results (val, test data and plots) are saved to respective folders.

## Contact

Sinjoy Saha
  * [LinkedIn](https://linkedin.com/in/sinjoysaha)
  * [Twitter](https://twitter.com/SinjoySaha)
