## Probabilistic deep learning models

This is a simple library that support bayesian deep learning model for image segmenation with uncertainty estimation. This library is built upon pytorch.
My objective is to provide simple yey powerful library to support Bayesian deep learning. I have started with image segmentaion and later on I will add more Bayesian models.

## Table of contents
1. Installation


## Installation
pip install .

## Documentation
### Data Loader
I have provided a custom dataloader(DataLoaderSegmentation class) for image segmentation.

### Bayesian UNet
This model uses the methodology outlined in paper(). I have also given a Bayesian Loss function to train the model by taking aleatoric uncertainty into account.
You can refer this example.
