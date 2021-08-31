[![build](https://github.com/mark-antal-csizmadia/nn-blocks/actions/workflows/main.yml/badge.svg)](https://github.com/mark-antal-csizmadia/nn-blocks/actions/workflows/main.yml)

# nn-blocks

## Introduction

A neural network library built from scratch, without dedicated deep learning packages. Training and testing deep neural networks and utilizing deep learning best practices for multi-class classification with fully connected neural networks and text generation with recurrent neural networks.

## Notebooks

- <a href="https://nbviewer.jupyter.org/github/mark-antal-csizmadia/nn-blocks/blob/main/one-layer.ipynb">
    <img align="center" src="https://img.shields.io/badge/Jupyter-one%5Flayer.ipynb-informational?style=flat&logo=Jupyter&logoColor=F37626&color=blue" />
  </a>
  
  + image classification on the CIFAR-10 dataset
  + one-layer networks with Hinge and cross entropy losses
  + [cyclical learning rate schedule](https://arxiv.org/abs/1506.01186) for improved learning
  + exploring the effects of the initial learning rate of the cyclical learning rate schedule and L2 regularization strength on model performance, without hyperparameter search


- <a href="https://nbviewer.jupyter.org/github/mark-antal-csizmadia/nn-blocks/blob/main/two-layer.ipynb">
    <img align="center" src="https://img.shields.io/badge/Jupyter-two%5Flayer.ipynb-informational?style=flat&logo=Jupyter&logoColor=F37626&color=blue" />
  </a>
  
  + image classification on the CIFAR-10 dataset
  + two-layer networks with cross entropy loss
  + [cyclical learning rate schedule](https://arxiv.org/abs/1506.01186) for improved learning
  + [Xavier initialization](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) for avoiding activation saturation
  + Bayesian hyperparameter search with [hyperopt](https://github.com/hyperopt/hyperopt)
    

- <a href="https://nbviewer.jupyter.org/github/mark-antal-csizmadia/nn-blocks/blob/main/k-layer.ipynb">
    <img align="center" src="https://img.shields.io/badge/Jupyter-k%5Flayer.ipynb-informational?style=flat&logo=Jupyter&logoColor=F37626&color=blue" />
  </a>
    
  + image classification on the CIFAR-10 dataset
  + k-layer networks with cross-entropy loss
  + [cyclical learning rate schedule](https://arxiv.org/abs/1506.01186) for improved learning
  + [Xavier initialization](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) for avoiding activation saturation
  + Bayesian hyperparameter search with [hyperopt](https://github.com/hyperopt/hyperopt)
  + [dropout](https://jmlr.org/papers/v15/srivastava14a.html) and [batch normalization](https://arxiv.org/abs/1502.03167) for avoiding overfitting
  + data augmentation for avoiding overfitting with [imgaug](https://github.com/aleju/imgaug)
  + [AdaGrad](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) for more efficient gradient descent optimization
  

- <a href="https://nbviewer.jupyter.org/github/mark-antal-csizmadia/nn-blocks/blob/main/rnn.ipynb">
    <img align="center" src="https://img.shields.io/badge/Jupyter-rnn.ipynb-informational?style=flat&logo=Jupyter&logoColor=F37626&color=blue" />
  </a>

    + generating text from Harry Potter books and Donald Trump tweets with RNNs
    + one-hot encoding, gradient clipping and smoothed loss, etc.


- <a href="https://nbviewer.jupyter.org/github/mark-antal-csizmadia/nn-blocks/blob/main/regression.ipynb">
    <img align="center" src="https://img.shields.io/badge/Jupyter-regression.ipynb-informational?style=flat&logo=Jupyter&logoColor=F37626&color=blue" />
  </a>
  
    + linear and non-linear regression
    
