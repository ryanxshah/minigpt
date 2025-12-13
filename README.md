# miniGPT

This repository is an implementation of a language model based on the GPT architecture. It uses character-level tokenization to predict the next token based on all preceding tokens. Additionally, this model uses a masked attention mechanism I wrote for general use.

## Usage
This repository contains the complete model architecture and training pipeline to train additional models.

### Adjusting Hyperparameters
To adjust model hyperparameters to train additional models, edit the corresponding hyperparameters in `hyperparams.py`. This file contains three dictionaries, one for utility hyperparameters, one for model hyperparameters, and one for training parameters. 

### Training
To train a model, simply run `python train.py` in your terminal.
