# Graph Attention Networks for Molecule Classification

This repository contains a Jupyter notebook implementing Graph Attention Networks (GAT) for the purpose of molecule classification. It demonstrates the process of loading and preprocessing datasets, creating graph representations of molecules, and using a GAT model for binary classification to determine if a molecule binds to RNA or a protein.

## Overview

The notebook includes:
- Loading and preprocessing of chemical compound datasets
- Conversion of molecules to graph representations
- Implementation of a GAT model with PyTorch and DGL (Deep Graph Library)
- Training, validation, and testing of the model
- Visualization of attention weights and performance metrics

## Installation

To run this notebook, you will need an environment with Python and the following packages installed:

- pandas
- numpy
- scipy
- matplotlib
- seaborn
- scikit-learn
- torch
- dgl
- rdkit
- optuna
- networkx

You can install the required packages using pip:

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn torch dgl rdkit optuna networkx
```

## Running the Notebook

After cloning the repository and ensuring that you have the necessary packages installed, you can open the Jupyter notebook (`your_notebook_name.ipynb`) in Jupyter Lab or Jupyter Notebook by running:

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

Navigate to the directory containing the notebook file and open it.

## Data

The dataset used in this notebook is a combination of chemical compounds from various sources, processed into a format suitable for graph-based machine learning models. Due to the size of the dataset, it is not included in this repository. Please follow the instructions in the notebook to generate or download the necessary data.

## Model Training and Evaluation

The notebook guides you through the process of training the GAT model, including hyperparameter optimization using Optuna, model evaluation, and visualization of the results. 

## Contributing

Contributions to this project are welcome. Please open an issue or pull request if you have suggestions for improvement.
