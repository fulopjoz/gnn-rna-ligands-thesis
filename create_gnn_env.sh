#!/bin/bash

# Create a conda environment for GNN with the specified packages
conda create -c anaconda -c conda-forge -c services -c pyg -c nvidia -c dglteam -c pytorch -n gnn_env chembl_structure_pipeline networkx hdbscan nodejs ipywidgets papermill jupyter rdkit==2018.09.3 python=3.7 pyg dgl-cuda11.7 dgllife pytorch torchvision torchaudio numpy pandas seaborn umap-learn scikit-learn matplotlib scipy pytorch-cuda=12.1 -y
