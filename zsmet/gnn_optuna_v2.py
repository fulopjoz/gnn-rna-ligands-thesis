#!/usr/bin/env python3

# %% [markdown]
# # Load Libraries

# %%
# Standard libraries
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated') # Known issue with PyTorch and DGL
import IPython

# Data handling
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Machine learning and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.cluster import HDBSCAN

# Neural Networks and Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optuna for hyperparameter optimization
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner

# Graph Neural Networks
import dgl
import dgl.nn as dglnn
from dgl import batch
from dgl.data.utils import save_graphs, load_graphs
from dgl.nn import GATConv

# Cheminformatics
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_graph
from chembl_structure_pipeline import standardizer

# Network analysis
import networkx as nx

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown]
# check if GPU is aviable and the device

# %%
import torch
print("PyTorch version:", torch.__version__)
print("Is CUDA Supported?", torch.cuda.is_available())

# %%
torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0)

# %% [markdown]
# # Data Preprocessing

# %% [markdown]
# load dataset

# %%
# # load json data from data_mvi folder
# chemdiv = pd.read_json('data_mvi/chemdiv_df.json')
# enamine = pd.read_json('data_mvi/enamine_df.json')
# enmine_protein = pd.read_json('data_mvi/enamine_protein_df.json')
# life_chemicals = pd.read_json('data_mvi/life_chemicals_df.json')
# robin = pd.read_json('data_mvi/robin_df.json')

# # add 'source column'
# chemdiv['source'] = 'chemdiv'
# enamine['source'] = 'enamine'
# enmine_protein['source'] = 'enmine_protein'
# life_chemicals['source'] = 'life_chemicals'
# robin['source'] = 'robin'

# # combine all dataframes
# df = pd.concat([chemdiv, enamine, enmine_protein, life_chemicals, robin], ignore_index=True)

# # delte 'mol' column and ECFp6 column
# df = df.drop(['mol', 'ECFP6'], axis=1)

# # create 'mol' column and use SMILES column to create mol object
# df['mol'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))

# # create 'ECFP6' column and use mol object to create ECFP6 fingerprint
# # df['ECFP6'] = df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=2048))

# # check for duplicates using 'SMILES' column
# df.duplicated(subset=['SMILES']).sum()
# create column for labels, if 'rna' == 1 - molecule binds to the RNA, else - Protein

# # create 'rna' column and if in column 'source' is 'enmine_protein' set 'rna' to 0 else 1
# df['rna'] = df['source'].apply(lambda x: 0 if x == 'enmine_protein' else 1)

# # divide data back to 5 dataframes
# chemdiv = df[df['source'] == 'chemdiv']
# enamine = df[df['source'] == 'enamine']
# enmine_protein = df[df['source'] == 'enmine_protein']
# life_chemicals = df[df['source'] == 'life_chemicals']
# robin = df[df['source'] == 'robin']


# smiles_to_delete = [
#     "COc1ccc(-c2n(-c3ccccn3)c3ccccc3[n+]2-c2ccccn2)cc1.[O-][Cl+3]([O-])([O-])[O-]",
#     "[O-][Cl+3]([O-])([O-])[O-].c1ccc(-c2[se]c3nccc[n+]3c2-c2ccccc2)cc1",
#     "OC1CSc2n(-c3ccccc3)c(-c3ccccc3)c(-c3ccccc3)[n+]2C1.[O-][Cl+3]([O-])([O-])[O-]",
#     "[O-][Cl+3]([O-])([O-])[O-].c1ccc(-c2ccc(-c3n(-c4ccccn4)c4ccccc4[n+]3-c3ccccn3)cc2)cc1",
#     "COc1ccc(-c2cc(=Nc3cccc(C)c3C)c3cc(C)ccc3o2)cc1.[O-][Cl+3]([O-])([O-])O",
#     "CN(C)c1ccc(/C=C/c2cc(-c3ccccc3)c3ccccc3[o+]2)cc1.[O-][Cl+3]([O-])([O-])[O-]"
# ]

# # delete from enamine using 'SMILES' column
# enamine = enamine[~enamine['SMILES'].isin(smiles_to_delete)]
# picked_molecules = pd.read_json('data_mvi/picked_molecules.json')

# # # combine all df
# df_small = pd.concat([chemdiv, enamine, picked_molecules, life_chemicals, robin], ignore_index=True)


# %% [markdown]
# Check for the inconsistency
# 

# %% [markdown]
# ## Create Graphs from molecules and add features to nodes (atoms) - mainly one hot encoding , edges (bonds)

# %%
# Load the graphs
loaded_graphs_orig, _ = dgl.load_graphs('data_mvi/molecule_graphs_orig.bin')

# Load the labels and additional information
with open('graph_labels.json', 'r') as f:
    loaded_graph_labels = json.load(f)

# You can now reconstruct a DataFrame or directly use the loaded data
reconstructed_df = pd.DataFrame(loaded_graph_labels)


# %% [markdown]
# ### Visualization of the graph representation 

# %%

# mol = df_small['mol'][42]
# AllChem.Compute2DCoords(mol)
# # draw the molecule
# Draw.MolToImage(mol, size=(600, 600))

# %%

# # Function to draw a molecule with atom numbering
# def draw_molecule_with_atom_index(mol):
#     d2d = rdMolDraw2D.MolDraw2DCairo(600, 600) # or MolDraw2DSVG to get SVG output
#     opts = d2d.drawOptions()
#     for i in range(mol.GetNumAtoms()):
#         opts.atomLabels[i] = str(i)
#     d2d.DrawMolecule(mol)
#     d2d.FinishDrawing()
#     return d2d.GetDrawingText()

# # Assuming df_small['mol'][42] is an RDKit molecule
# mol = df_small['mol'][42]
# AllChem.Compute2DCoords(mol)

# # Draw the molecule with atom numbering
# img = draw_molecule_with_atom_index(mol)  # This returns binary data for the image
# # To display the image in a Jupyter notebook, you can do the following:
# IPython.display.Image(data=img)



# %%
# def visualize_graph_with_mol_layout(G, mol):
#     pos = {atom_idx: (atom.GetOwningMol().GetConformer().GetAtomPosition(atom_idx).x,
#                       atom.GetOwningMol().GetConformer().GetAtomPosition(atom_idx).y)
#            for atom_idx, atom in enumerate(mol.GetAtoms())}
    
#     nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=200, edge_color='black', arrows=True)
#     plt.show()

# def mol_to_nx(mol):
#     G = nx.DiGraph()  # Initialize a directed graph to allow for bidirectional edges
#     for bond in mol.GetBonds():
#         a1 = bond.GetBeginAtomIdx()
#         a2 = bond.GetEndAtomIdx()
#         # Add edges in both directions
#         G.add_edge(a1, a2)
#         G.add_edge(a2, a1)
#     return G

# # Convert the RDKit molecule to a NetworkX graph
# G = mol_to_nx(mol)

# # Now visualize the NetworkX graph with the layout based on the molecule
# visualize_graph_with_mol_layout(G, mol)

# %% [markdown]
# # Create train and test set

# %%
# First split into training and test sets
train_graphs, test_graphs, train_labels, test_labels = train_test_split(
    loaded_graphs_orig, 
    reconstructed_df['rna'], 
    test_size=0.2, 
    random_state=42
)

# Further split the training set into training and validation sets
train_graphs, val_graphs, train_labels, val_labels = train_test_split(
    train_graphs, 
    train_labels, 
    test_size=0.2,  # 20% of the original training set for validation
    random_state=42
)

# merge train and val,  labels are Series
train_graphs_all = train_graphs + val_graphs
train_labels_all = pd.concat([train_labels, val_labels])

# %% [markdown]
# there was a problem with few isolated nodes in few graphs so I added loops to each node to have information at least about themselfs

# %%
# for i, g in enumerate(train_graphs):
#     if (g.in_degrees() == 0).any():
#         print(f"Graph {i} has isolated nodes")

# # if there is not any isolated nodes in train_graphs, print 'No isolated nodes'
# print('No isolated nodes')

# for i, g in enumerate(test_graphs):
#     if (g.in_degrees() == 0).any():
#         print(f"Graph {i} has isolated nodes")

# # if there is not any isolated nodes in train_graphs, print 'No isolated nodes'
# print('No isolated nodes')

# %% [markdown]
# # Graph Attention Networks (GAT)

# %%
# # V1
# class GATClassifier(nn.Module):
#     def __init__(self, in_feats, hidden_size, num_heads=1):
#         super(GATClassifier, self).__init__()
#         self.conv1 = dgl.nn.GATConv(in_feats, hidden_size, num_heads=num_heads, allow_zero_in_degree=True)
#         # Adjust the output dimension of the linear layer to match the concatenated head outputs
#         self.fc = nn.Linear(hidden_size * num_heads, 1)  # Corrected dimension

#     def forward(self, g, features, get_attention=False):
#         # Apply GAT convolution
#         if get_attention:
#             gat_output, attn_weights = self.conv1(g, features, get_attention=True)
#         else:
#             gat_output = self.conv1(g, features, get_attention=False)

#         # Concatenate head outputs instead of averaging
#         x = gat_output.view(gat_output.size(0), -1)  # Reshape to concatenate head outputs

#         # Update node features
#         g.ndata['h'] = x

#         # Aggregate node features across the graph
#         x = dgl.mean_nodes(g, 'h')

#         # Linear layer expects concatenated features from all heads
#         x = F.elu(self.fc(x))

#         if get_attention:
#             return torch.sigmoid(x), attn_weights
#         else:
#             return torch.sigmoid(x)


# %%
class GATClassifier(nn.Module):
    def __init__(self, in_feats, hidden_size, num_heads=1, num_layers=2):
        super(GATClassifier, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(dgl.nn.GATConv(in_feats, hidden_size, num_heads=num_heads, allow_zero_in_degree=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(dgl.nn.GATConv(hidden_size * num_heads, hidden_size, num_heads=num_heads, allow_zero_in_degree=True))
        
        # Output layer
        self.layers.append(dgl.nn.GATConv(hidden_size * num_heads, hidden_size, num_heads=1, allow_zero_in_degree=True))  # Usually, the output layer has a single head.
        
        # Linear layer for final prediction
        self.fc = nn.Linear(hidden_size, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features, get_attention=False):
        h = features
        attn_weights = []

        for i, layer in enumerate(self.layers):
            if get_attention and i == len(self.layers) - 1:
                h, attn_weight = layer(g, h, get_attention=True)
                attn_weights.append(attn_weight)
            else:
                h = layer(g, h)
                
            h = h.view(h.size(0), -1)  # Reshape to concatenate head outputs for next layer input
            if i < len(self.layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)  # Apply dropout after activation function
                
        # Aggregate node features across the graph
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        # Final linear transformation
        y = self.fc(hg)

        return (torch.sigmoid(y), attn_weights) if get_attention else torch.sigmoid(y)


# %%
def objective(trial):
    print(f"Running trial number: {trial.number + 1}")

    # Define hyperparameter search space
    num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    print(f"Hyperparameters: num_heads={num_heads}, hidden_size={hidden_size}, lr={lr}")

    in_feats = train_graphs[0].ndata['h'].shape[1]

    # Initialize the model with suggested hyperparameters
    model = GATClassifier(in_feats, hidden_size=hidden_size, num_heads=num_heads).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_threshold = 5  # How many epochs to wait after last time validation loss improved.
    
    for epoch in range(20):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, _, _, _ = validate(model, val_loader, criterion, device)
        
        # Print epoch details
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Report validation loss to Optuna for potential early pruning
        trial.report(val_loss, epoch)

        # Check if the trial should be pruned based on the reported val_loss
        if trial.should_prune():
            print("Pruning trial")
            raise optuna.exceptions.TrialPruned()

        # Custom early stopping logic based on validation loss improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stop_threshold:
            print(f"No improvement in validation loss for {early_stop_threshold} consecutive epochs, stopping early.")
            break

    return val_loss



# %%
def train(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for batch_graphs, batch_labels in data_loader:
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        batch_features = batch_graphs.ndata['h']

        optimizer.zero_grad()
        outputs = model(batch_graphs, batch_features).squeeze()  # Adjust for binary classification
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_labels.size(0)

        # Calculate accuracy
        predicted = (outputs.sigmoid() > 0.5).long()  # Assuming outputs are logits and binary classification
        total_correct += (predicted == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# %%
def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    y_pred_proba = []
    with torch.no_grad():
        for graphs, labels in data_loader:
            graphs = graphs.to(device)
            features = graphs.ndata['h'].to(device)
            labels = labels.to(device)
            outputs = model(graphs, features)
            if outputs.dim() == 2 and outputs.shape[1] == 1:  # Check for single output per sample
                outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (outputs.sigmoid() > 0.5).long()  # Convert to binary predictions

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_proba.extend(outputs.sigmoid().cpu().numpy())  # Use sigmoid if outputs are logits
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(y_true, y_pred)  # Calculate accuracy

    return avg_loss, accuracy, y_true, y_pred, y_pred_proba


# %%
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.stopped_epoch = 0  # Attribute to store the epoch number

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch  # Store the epoch number


# %%
class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        
        # Reset index if labels is a pandas DataFrame/Series to ensure continuous indexing
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            self.labels = labels.reset_index(drop=True)
        else:
            self.labels = labels

    def __len__(self):
        # The lengths of graphs and labels should be the same
        assert len(self.graphs) == len(self.labels), "Graphs and labels must have the same length"
        return len(self.graphs)

    def __getitem__(self, idx):
        try:
            graph = self.graphs[idx]
            label = self.labels[idx]
            return graph, label
        except IndexError:
            print(f"IndexError: Index {idx} out of range for dataset of size {len(self)}")
            raise
        
        

# %%
def collate_graphs(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return batched_graph, labels



# %%
batch_size = 2048
num_workers = 30

# Assuming train_labels and test_labels are originally 1D tensors or lists
# Create the datasets with updated labels
# Create the DataLoader with multiple workers

if isinstance(train_labels, torch.Tensor):
    train_labels = train_labels.tolist()
train_dataset = GraphDataset(train_graphs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_graphs)

if isinstance(test_labels, torch.Tensor):
    test_labels = test_labels.tolist()
test_dataset = GraphDataset(test_graphs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_graphs)

if isinstance(val_labels, torch.Tensor):
    val_labels = val_labels.tolist()
val_dataset = GraphDataset(val_graphs, val_labels)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_graphs)

if isinstance(train_labels_all, torch.Tensor):
    train_labels_all = train_labels_all.tolist()
train_dataset_all = GraphDataset(train_graphs_all, train_labels_all)
train_loader_all = DataLoader(train_dataset_all, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_graphs)



# %% [markdown]
# ## Instance of the the GAT Model

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# Initialize Optuna study with a Hyperband pruner
pruner = HyperbandPruner(min_resource=1, max_resource=20, reduction_factor=3)
study = optuna.create_study(direction='minimize', pruner=pruner)


# %%
# Run the optimization
study.optimize(objective, n_trials=50, timeout=600)

# Output the results
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


# %%
best_params = study.best_trial.params
model = GATClassifier(
    in_feats, 
    hidden_size=best_params['hidden_size'], 
    num_heads=best_params['num_heads']
).to(device)

# Reinitialize the optimizer with the best learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

# Train your model with the best hyperparameters (implement the training loop here)


# %% [markdown]
# ## Train the GAT model

# %%
import numpy as np
from sklearn.metrics import accuracy_score

num_epochs = 300
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

# Initialize lists to store loss and accuracy values
gat_train_loss_list = []
gat_train_accuracy_list = []
gat_val_loss_list = []
gat_val_accuracy_list = []

for epoch in range(num_epochs):
    # Training
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    gat_train_loss_list.append(train_loss)  # Store train loss
    gat_train_accuracy_list.append(train_accuracy)  # Store train accuracy

    # Validation
    val_loss, val_accuracy, _, _, _ = validate(model, val_loader, criterion, device)
    gat_val_loss_list.append(val_loss)  # Store validation loss
    gat_val_accuracy_list.append(val_accuracy)  # Store validation accuracy

    # Print metrics
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Check for early stopping
    if early_stopping(val_loss, epoch):
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Store the epoch number when early stopping was triggered
num_epochs_early_stopped = early_stopping.stopped_epoch + 1


# %%
# Ensure that the length of gat_train_loss_list and gat_val_loss_list is equal to the number of epochs processed
assert len(gat_train_loss_list) == len(gat_val_loss_list)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(gat_train_loss_list, label='Training Loss')
plt.plot(gat_val_loss_list, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# %%
# plot the accuracy
plt.figure(figsize=(10, 6))
plt.plot(gat_train_accuracy_list, label='Training Accuracy')
plt.plot(gat_val_accuracy_list, label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ## Train on merged train and val sets
# 

# %%
# def train(model, data_loader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     for batch_graphs, batch_labels in data_loader:
#         batch_graphs = batch_graphs.to(device)
#         batch_labels = batch_labels.clone().detach().to(device)
#         batch_features = batch_graphs.ndata['h']

#         outputs = model(batch_graphs, batch_features)
#         outputs = outputs.squeeze() # remove extra dimension
#         loss = criterion(outputs, batch_labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss / len(data_loader)

# %%
def train(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for batch_graphs, batch_labels in data_loader:
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        batch_features = batch_graphs.ndata['h']

        outputs = model(batch_graphs, batch_features)
        outputs = outputs.squeeze()  # Adjust for binary classification if necessary

        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Assuming binary classification and outputs are logits
        # For multi-class classification, use torch.max(outputs, 1) and adjust the threshold if needed
        predicted = (torch.sigmoid(outputs) > 0.5).long()  # Convert logits to binary predictions
        total_correct += (predicted == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# %% [markdown]
# ## Reset weights before training again with all data
# 

# %%
def reset_weights(m):
    """
    This function will reset the weights of a given module.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

# Example usage with your model
model.apply(reset_weights)

# %%
print("Final training on merged training and validation sets")

# Initialize list to store loss values
final_train_losses = []

for epoch in range(num_epochs_early):
    # Training
    train_loss = train(model, train_loader_all, criterion, optimizer, device)
    final_train_losses.append(train_loss)  # Store train loss

    # Print train loss
    print(f"Epoch {epoch+1}/{num_epochs_early}, Train Loss: {train_loss:.4f}")


# %%
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(final_train_losses, label='Training Loss')
plt.title('Training Loss Over Epochs on Merged Training and Validation Sets')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# %%
# plot the accuracy
plt.figure(figsize=(10, 6))
plt.plot(gat_train_accuracy_list, label='Training Accuracy')
plt.plot(gat_val_accuracy_list, label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ## Save the model

# %%
# if there is not a folder named 'model' create one
if not os.path.exists('model_experiment'):
    os.mkdir('model_experiment')

# save the model
torch.save(model.state_dict(), 'model_experiment/gat_model_experiment.pth')

# %%
# load the model
model = GATClassifier(in_feats, hidden_size=148, num_heads=4).to(device)
model.load_state_dict(torch.load('model_experiment/gat_model_experiment.pth'))

# %% [markdown]
# ## Test the model

# %% [markdown]
# ## Predict probabilities

# %%
y_true, y_pred, y_pred_proba = test(model, test_loader, device)

confidence_percentages = [prob * 100 for prob in y_pred_proba]

y_pred_proba = np.array(y_pred_proba)

# %% [markdown]
# ## Results

# %%
print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
print(f'Precision: {precision_score(y_true, y_pred)}')
print(f'Recall: {recall_score(y_true, y_pred)}')
print(f'F1: {f1_score(y_true, y_pred)}')
print(f'AUC: {roc_auc_score(y_true, y_pred_proba)}')

# %%
# Accuracy: 0.7387478849407784
# Precision: 0.7532486494378742
# Recall: 0.7039159503342884
# F1: 0.727747213993511
# AUC: 0.8162738424254579


# %%
def visualize_molecule_with_attention(mol, attn_weights, threshold=0.1):
    # Convert attention weights to a format suitable for visualization
    # Assume attn_weights is a tensor of shape (num_edges, 1) with attention scores
    attn_weights = attn_weights.squeeze().numpy()
    
    # Identify edges with attention weights above the threshold
    high_attention_edges = [i for i, weight in enumerate(attn_weights) if weight > threshold]
    
    # Map edges in the graph back to bonds in the molecule
    bonds_to_highlight = [mol.GetBondWithIdx(i).GetIdx() for i in high_attention_edges]
    
    # Visualize the molecule with highlighted bonds
    Draw.MolToImage(mol, highlightBonds=bonds_to_highlight)


# %%
# Visualize the molecule with attention
visualize_molecule_with_attention(mol, attn_weights, threshold=0.3)



# %%
# save all visualized molecules with attention and create a folder for them
if not os.path.exists('visualized_molecules'):
    os.mkdir('visualized_molecules')
    
for i, (mol, attn_weights) in enumerate(zip(mols, attn_weights_list)):
    img = visualize_molecule_with_attention(mol, attn_weights, threshold=0.3)
    img.save(f'visualized_molecules/molecule_{i}.png')
    
    


# %%
def extract_high_attention_fragments(mol, attn_weights, threshold=0.1):
    # Similar to visualization, but now store the fragments as SMILES or another format
    submols = []
    for i, weight in enumerate(attn_weights.squeeze().numpy()):
        if weight > threshold:
            # Extract subgraph or fragment; this step depends on how you define a 'fragment'
            # You might need RDKit's substructure search or other techniques to define fragments
            submol = Chem.PathToSubmol(mol, [i])  # Simplified; actual implementation may vary
            submols.append(Chem.MolToSmiles(submol))
    return submols


# %%
# Extract high-attention fragments
high_attention_fragments = extract_high_attention_fragments(mol, attn_weights, threshold=0.3)

# save as json
with open('high_attention_fragments.json', 'w') as f:
    json.dump(high_attention_fragments, f)

# %% [markdown]
# ## Confusion matrix

# %%
# confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=['RNA binder', 'non-RNA binder'], columns=['RNA binder', 'non-RNA binder'])
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
sns.set(context='paper', style='white', font_scale=1.5)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('visuals/confusion_matrix_gat_experiment.png', dpi=300)
plt.show()



# %% [markdown]
# ## Classification report

# %%
# classification report
print(classification_report(y_true, y_pred))


# %% [markdown]
# ## ROC AUC

# %%
# visualize ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)'% roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('visuals/roc_curve_gat_experiment.png', dpi=300)
plt.show()


# %%
# visualize confidence_percentages in plot
plt.figure(figsize=(8, 6))
plt.hist(confidence_percentages, bins=50)
plt.title('Confidence Histogram')
plt.xlabel('Confidence Percentage')
plt.ylabel('Frequency')
plt.savefig('visuals/confidence_histogram_gat_experiment.png', dpi=300)
plt.show()


# %%
# Sort test predictions by confidence and get top 10 indices
sorted_pairs = sorted(enumerate(confidence_percentages), key=lambda x: x[1], reverse=True)
top_molecules_test_indices = [index for index, _ in sorted_pairs[:20]]

# The indices in test_graphs correspond to their positions in the original loaded_graphs_orig
# Find the original indices in loaded_graphs_orig
original_indices = [loaded_graphs_orig.index(test_graphs[i]) for i in top_molecules_test_indices]

# Use the original indices to look up the corresponding rows in reconstructed_df
top_molecule_row_indices = reconstructed_df.index[original_indices]

# Now, top_molecule_row_indices contains the DataFrame indices of the top 10 RNA-binding molecules


# %%
# create folder 'top_molecules' if it does not exist
if not os.path.exists('top_molecules_gat_experiment'):
    os.mkdir('top_molecules_gat_experiment')
    

# %%
# Retrieve the SMILES codes and source information for the top 10 molecules
top_smiles = df_small.iloc[top_molecule_row_indices]['SMILES'].tolist()
top_sources = df_small.iloc[top_molecule_row_indices]['source'].tolist()

# Retrieve the confidence scores using the indices relative to the test dataset
top_confidences = [confidence_percentages[i] for i in top_molecules_test_indices]

# Convert SMILES to RDKit Mol objects
mols = [Chem.MolFromSmiles(smile) for smile in top_smiles]

# Generate and save images for each molecule
for i, mol in enumerate(mols):
    img = Draw.MolToImage(mol, size=(500, 500))
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    # title = f"SMILES: {top_smiles[i]}\nSource: {top_sources[i]}\nConfidence: {top_confidences[i]:.2f}%"
    # plt.title(title)
    plt.subplots_adjust(bottom=0.20)  # Adjust the position of the title
    plt.axis('off')
    
    # Save the image to the 'top_molecules' folder using the rank and source as the filename
    plt.savefig(f"top_molecules_gat_experiment/molecule_{i+1}_{top_sources[i]}_a.png", dpi=300)
    plt.show()


# %%
# Define the figure size and font size for titles
fig_size = (6, 6)
title_font_size = 12

# Generate and save images for each molecule
for i, mol in enumerate(mols):
    img = Draw.MolToImage(mol, size=(300, 300))
    
    # Create a figure with adjusted size
    plt.figure(figsize=fig_size)
    plt.imshow(img)
    
    # Adjust the position of the text within the image
    title = f"SMILES: {top_smiles[i]}\nSource: {top_sources[i]}\nConfidence: {top_confidences[i]:.2f}%"
    plt.title(title, fontsize=title_font_size)
    plt.subplots_adjust(bottom=0.15)  # Adjust the position of the title
    
    plt.axis('off')
    
    # Save the image to the 'top_molecules' folder using the rank and source as the filename
    plt.savefig(f"top_molecules_gat_experiment/molecule_{i+1}_{top_sources[i]}_b.png", dpi=300)
    plt.show()


# %%
# Plot the training and validation loss, same model will have same color but different line style use seaborn and style report and second plot style to talk-presentation
sns.set(context='talk', rc={'figure.figsize': (11.5, 7)})

plt.plot(gcn_train_loss_list, label='GCN Train', color='r', linestyle='--')
plt.plot(gcn_val_loss_list, label='GCN Validation', color='r', linestyle='-')
plt.plot(mpnn_train_loss_list, label='MPNN Train', color='b', linestyle='--')
plt.plot(mpnn_val_loss_list, label='MPNN Validation', color='b', linestyle='-')
plt.plot(gat_train_loss_list, label='GAT Train', color='g', linestyle='--')
plt.plot(gat_val_loss_list, label='GAT Validation', color='g', linestyle='-')

# Markers for early stopping
plt.scatter(49, mpnn_train_loss_list[48], color='b', marker='o', s=20, label='MPNN Early Stopping')
plt.scatter(49, mpnn_val_loss_list[48], color='b', marker='o', s=20)
plt.scatter(49, gat_train_loss_list[48], color='g', marker='o', s=20, label='GAT Early Stopping')
plt.scatter(49, gat_val_loss_list[48], color='g', marker='o', s=20)
plt.scatter(80, gcn_train_loss_list[79], color='r', marker='o', s=20, label='GCN Early Stopping')
plt.scatter(80, gcn_val_loss_list[79], color='r', marker='o', s=20)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('visuals/loss_talk_experiment.png', dpi=300)
plt.show()


# Plot the training and validation loss, same model will have same color but different line style use seaborn and style report and second plot style to talk-presentation
sns.set(context='paper', rc={'figure.figsize': (10, 4)})


plt.plot(gcn_train_loss_list, label='GCN Train', color='r', linestyle='--')
plt.plot(gcn_val_loss_list, label='GCN Validation', color='r', linestyle='-')
plt.plot(mpnn_train_loss_list, label='MPNN Train', color='b', linestyle='--')
plt.plot(mpnn_val_loss_list, label='MPNN Validation', color='b', linestyle='-')
plt.plot(gat_train_loss_list, label='GAT Train', color='g', linestyle='--')
plt.plot(gat_val_loss_list, label='GAT Validation', color='g', linestyle='-')

# Markers for early stopping
plt.scatter(49, mpnn_train_loss_list[48], color='b', marker='o', s=20, label='MPNN Early Stopping')
plt.scatter(49, mpnn_val_loss_list[48], color='b', marker='o', s=20)
plt.scatter(49, gat_train_loss_list[48], color='g', marker='o', s=20, label='GAT Early Stopping')
plt.scatter(49, gat_val_loss_list[48], color='g', marker='o', s=20)
plt.scatter(80, gcn_train_loss_list[79], color='r', marker='o', s=20, label='GCN Early Stopping')
plt.scatter(80, gcn_val_loss_list[79], color='r', marker='o', s=20)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('visuals/loss_paper_experiment.png', dpi=300)
plt.show()





