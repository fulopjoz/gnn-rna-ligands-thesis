
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated') 
import IPython
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.cluster import HDBSCAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
import dgl
import dgl.nn as dglnn
from dgl import batch
from dgl.data.utils import save_graphs, load_graphs
from dgl.nn import GATConv
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_graph
from chembl_structure_pipeline import standardizer
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from torch import sigmoid

print("PyTorch version:", torch.__version__)
print("Is CUDA Supported?", torch.cuda.is_available())

torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0)
loaded_graphs_orig, _ = dgl.load_graphs('data_mvi/molecule_graphs_orig.bin')

with open('graph_labels.json', 'r') as f:
    loaded_graph_labels = json.load(f)

reconstructed_df = pd.DataFrame(loaded_graph_labels)

train_graphs, test_graphs, train_labels, test_labels = train_test_split(
    loaded_graphs_orig, 
    reconstructed_df['rna'], 
    test_size=0.2, 
    random_state=42
)

train_graphs, val_graphs, train_labels, val_labels = train_test_split(
    train_graphs, 
    train_labels, 
    test_size=0.2,  
    random_state=42
)

train_graphs_all = train_graphs + val_graphs
train_labels_all = pd.concat([train_labels, val_labels])

# class GATClassifier(nn.Module):
#     def __init__(self, in_feats, hidden_size, num_heads=1, num_layers=2):
#         super(GATClassifier, self).__init__()
#         self.layers = nn.ModuleList()
        
#         self.layers.append(dgl.nn.GATConv(in_feats, hidden_size, num_heads=num_heads, allow_zero_in_degree=True))
        
#         for _ in range(num_layers - 2):
#             self.layers.append(dgl.nn.GATConv(hidden_size * num_heads, hidden_size, num_heads=num_heads, allow_zero_in_degree=True))
        
#         self.layers.append(dgl.nn.GATConv(hidden_size * num_heads, hidden_size, num_heads=1, allow_zero_in_degree=True))  # Usually, the output layer has a single head.
        
#         self.fc = nn.Linear(hidden_size, 1)
        
#         self.dropout = nn.Dropout(p=0.5)

#     def forward(self, g, features, get_attention=False):
#         h = features
#         attn_weights = []

#         for i, layer in enumerate(self.layers):
#             if get_attention and i == len(self.layers) - 1:
#                 h, attn_weight = layer(g, h, get_attention=True)
#                 attn_weights.append(attn_weight)
#             else:
#                 h = layer(g, h)
                
#             h = h.view(h.size(0), -1)  
#             if i < len(self.layers) - 1:
#                 h = F.elu(h)
#                 h = self.dropout(h)  
                
#         g.ndata['h'] = h
#         hg = dgl.mean_nodes(g, 'h')

#         y = self.fc(hg)

#         return (torch.sigmoid(y), attn_weights) if get_attention else torch.sigmoid(y)


class GATClassifier(nn.Module):
    def __init__(self, in_feats, hidden_size, num_heads=1):
        super(GATClassifier, self).__init__()

        self.layer = dgl.nn.GATConv(in_feats, hidden_size, num_heads=num_heads, allow_zero_in_degree=True)
        
        self.fc = nn.Linear(hidden_size * num_heads, 1)
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features, get_attention=False):
        if get_attention:
            h, attn_weight = self.layer(g, features, get_attention=True)
        else:
            h = self.layer(g, features)

        h = self.dropout(h)

        h = h.view(h.size(0), -1)
        
        h = F.elu(h)

        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        y = self.fc(hg)

        return (torch.sigmoid(y), attn_weight) if get_attention else torch.sigmoid(y)

    
    
def objective(trial):
    print(f"Running trial number: {trial.number + 1}")

    num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8, 10, 12, 14, 15, 16, 18, 20])
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64 , 128, 140, 148, 180, 200, 256])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    print(f"Hyperparameters: num_heads={num_heads}, hidden_size={hidden_size}, lr={lr}")

    in_feats = train_graphs[0].ndata['h'].shape[1]

    model = GATClassifier(in_feats, hidden_size=hidden_size, num_heads=num_heads).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(100): 
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, _, _, _ = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        trial.report(val_loss, epoch)

        if trial.should_prune():
            print("Pruning trial")
            raise optuna.exceptions.TrialPruned()

    return val_loss

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
        outputs = model(batch_graphs, batch_features).squeeze()  
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_labels.size(0)

        predicted = (outputs.sigmoid() > 0.5).long()  
        total_correct += (predicted == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

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
            if outputs.dim() == 2 and outputs.shape[1] == 1:  
                outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (outputs.sigmoid() > 0.5).long()  

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_proba.extend(outputs.sigmoid().cpu().numpy())  
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(y_true, y_pred) 

    return avg_loss, accuracy, y_true, y_pred, y_pred_proba

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.002):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.stopped_epoch = 0  

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
                self.stopped_epoch = epoch  

class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            self.labels = labels.reset_index(drop=True)
        else:
            self.labels = labels

    def __len__(self):
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
        
def collate_graphs(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return batched_graph, labels

batch_size = 64
num_workers = 20

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pruner = HyperbandPruner(min_resource=5, max_resource=100, reduction_factor=3)
study = optuna.create_study(direction='minimize', pruner=pruner)

study.optimize(objective, n_trials=100)  

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

best_params = study.best_trial.params
in_feats = train_graphs[0].ndata['h'].shape[1]

model = GATClassifier(
    in_feats, 
    hidden_size=best_params['hidden_size'], 
    num_heads=best_params['num_heads']
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
criterion = nn.BCEWithLogitsLoss() 

num_epochs = 300
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
gat_train_loss_list = []
gat_train_accuracy_list = []
gat_val_loss_list = []
gat_val_accuracy_list = []

for epoch in range(num_epochs):    
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    gat_train_loss_list.append(train_loss)  
    gat_train_accuracy_list.append(train_accuracy)  
    val_loss, val_accuracy, _, _, _ = validate(model, val_loader, criterion, device)
    gat_val_loss_list.append(val_loss)  
    gat_val_accuracy_list.append(val_accuracy)  
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    if early_stopping(val_loss, epoch):
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
    
num_epochs_early_stopped = early_stopping.stopped_epoch + 1

assert len(gat_train_loss_list) == len(gat_val_loss_list)

plt.figure(figsize=(10, 6))
plt.plot(gat_train_loss_list, label='Training Loss')
plt.plot(gat_val_loss_list, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(gat_train_accuracy_list, label='Training Accuracy')
plt.plot(gat_val_accuracy_list, label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

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
        outputs = outputs.squeeze() 

        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (predicted == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def reset_weights(m):
    """
    This function will reset the weights of a given module.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
model.apply(reset_weights)

print("Final training on merged training and validation sets")
final_train_losses = []

for epoch in range(num_epochs_early):    
    train_loss = train(model, train_loader_all, criterion, optimizer, device)
    final_train_losses.append(train_loss) 
    print(f"Epoch {epoch+1}/{num_epochs_early}, Train Loss: {train_loss:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(final_train_losses, label='Training Loss')
plt.title('Training Loss Over Epochs on Merged Training and Validation Sets')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(gat_train_accuracy_list, label='Training Accuracy')
plt.plot(gat_val_accuracy_list, label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

if not os.path.exists('model_experiment'):
    os.mkdir('model_experiment')

torch.save(model.state_dict(), 'model_experiment/gat_model_experiment.pth')

model = GATClassifier(
    in_feats, 
    hidden_size=best_params['hidden_size'], 
    num_heads=best_params['num_heads']
).to(device)

model.load_state_dict(torch.load('model_experiment/gat_model_experiment.pth'))


def test(model, data_loader, device, get_attention=False):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_proba = []
    attn_weights_list = []  # To store attention weights
    with torch.no_grad():
        for graphs, labels in data_loader:
            graphs = graphs.to(device)
            features = graphs.ndata['h'].to(device)
            labels = labels.to(device)
            if get_attention:
                outputs, attn_weights = model(graphs, features, get_attention=True)
                attn_weights_list.extend([aw.detach().cpu() for aw in attn_weights[-1]])  # Store attention weights
            else:
                outputs = model(graphs, features)
            
            # Assuming outputs are logits; apply sigmoid for probabilities
            probas = sigmoid(outputs).squeeze().cpu().numpy()
            preds = (probas > 0.5).astype(int)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_pred_proba.extend(probas)
    
    return y_true, y_pred, y_pred_proba, attn_weights_list

y_true, y_pred, y_pred_proba, attn_weights_list = test(model, test_loader, device)

def find_optimal_threshold(y_true, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.nanargmax(f1_scores)  # Using nanargmax to ignore NaN values
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

optimal_threshold = find_optimal_threshold(y_true, y_pred_proba)
print(f"Optimal Threshold: {optimal_threshold}")

# Adjust predictions based on the optimal threshold
adjusted_preds = (np.array(y_pred_proba) > optimal_threshold).astype(int)

# Evaluation metrics after adjustment
print(f'Adjusted Accuracy: {accuracy_score(y_true, adjusted_preds)}')
print(f'Adjusted Precision: {precision_score(y_true, adjusted_preds, zero_division=1)}')
print(f'Adjusted Recall: {recall_score(y_true, adjusted_preds)}')
print(f'Adjusted F1: {f1_score(y_true, adjusted_preds)}')

confidence_percentages = [prob * 100 for prob in adjusted_preds]

print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
print(f'Precision: {precision_score(y_true, y_pred)}')
print(f'Recall: {recall_score(y_true, y_pred)}')
print(f'F1: {f1_score(y_true, y_pred)}')
print(f'AUC: {roc_auc_score(y_true, adjusted_preds)}')

def visualize_molecule_with_attention(mol, attn_weights, threshold=0.1):
    attn_weights = attn_weights.squeeze().numpy()
    
    high_attention_edges = [i for i, weight in enumerate(attn_weights) if weight > threshold]
    
    bonds_to_highlight = [mol.GetBondWithIdx(i).GetIdx() for i in high_attention_edges]
    
    Draw.MolToImage(mol, highlightBonds=bonds_to_highlight)

visualize_molecule_with_attention(mol, attn_weights, threshold=0.3)

if not os.path.exists('visualized_molecules'):
    os.mkdir('visualized_molecules')
    
for i, (mol, attn_weights) in enumerate(zip(mols, attn_weights_list)):
    img = visualize_molecule_with_attention(mol, attn_weights, threshold=0.3)
    img.save(f'visualized_molecules/molecule_{i}.png')

def extract_high_attention_fragments(mol, attn_weights, threshold=0.1):
    submols = []
    for i, weight in enumerate(attn_weights.squeeze().numpy()):
        if weight > threshold:
            submol = Chem.PathToSubmol(mol, [i])  
            submols.append(Chem.MolToSmiles(submol))
    return submols

high_attention_fragments = extract_high_attention_fragments(mol, attn_weights, threshold=0.3)

with open('high_attention_fragments.json', 'w') as f:
    json.dump(high_attention_fragments, f)
    
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=['RNA binder', 'non-RNA binder'], columns=['RNA binder', 'non-RNA binder'])
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
sns.set(context='paper', style='white', font_scale=1.5)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('visuals/confusion_matrix_gat_experiment.png', dpi=300)
plt.show()

print(classification_report(y_true, y_pred))

fpr, tpr, thresholds = roc_curve(y_true, adjusted_preds)
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


plt.figure(figsize=(8, 6))
plt.hist(confidence_percentages, bins=50)
plt.title('Confidence Histogram')
plt.xlabel('Confidence Percentage')
plt.ylabel('Frequency')
plt.savefig('visuals/confidence_histogram_gat_experiment.png', dpi=300)
plt.show()

sorted_pairs = sorted(enumerate(confidence_percentages), key=lambda x: x[1], reverse=True)
top_molecules_test_indices = [index for index, _ in sorted_pairs[:20]]

original_indices = [loaded_graphs_orig.index(test_graphs[i]) for i in top_molecules_test_indices]

top_molecule_row_indices = reconstructed_df.index[original_indices]


if not os.path.exists('top_molecules_gat_experiment'):
    os.mkdir('top_molecules_gat_experiment')

top_smiles = df_small.iloc[top_molecule_row_indices]['SMILES'].tolist()
top_sources = df_small.iloc[top_molecule_row_indices]['source'].tolist()

top_confidences = [confidence_percentages[i] for i in top_molecules_test_indices]

mols = [Chem.MolFromSmiles(smile) for smile in top_smiles]

for i, mol in enumerate(mols):
    img = Draw.MolToImage(mol, size=(500, 500))
    plt.figure(figsize=(5, 5))
    plt.imshow(img)

    plt.subplots_adjust(bottom=0.20)  
    plt.axis('off')
    
    plt.savefig(f"top_molecules_gat_experiment/molecule_{i+1}_{top_sources[i]}_a.png", dpi=300)
    plt.show()

fig_size = (6, 6)
title_font_size = 12

for i, mol in enumerate(mols):
    img = Draw.MolToImage(mol, size=(300, 300))
    
    plt.figure(figsize=fig_size)
    plt.imshow(img)
    
    title = f"SMILES: {top_smiles[i]}\nSource: {top_sources[i]}\nConfidence: {top_confidences[i]:.2f}%"
    plt.title(title, fontsize=title_font_size)
    plt.subplots_adjust(bottom=0.15)  
    
    plt.axis('off')
    
    plt.savefig(f"top_molecules_gat_experiment/molecule_{i+1}_{top_sources[i]}_b.png", dpi=300)
    plt.show()

sns.set(context='talk', rc={'figure.figsize': (11.5, 7)})

plt.plot(gcn_train_loss_list, label='GCN Train', color='r', linestyle='--')
plt.plot(gcn_val_loss_list, label='GCN Validation', color='r', linestyle='-')
plt.plot(mpnn_train_loss_list, label='MPNN Train', color='b', linestyle='--')
plt.plot(mpnn_val_loss_list, label='MPNN Validation', color='b', linestyle='-')
plt.plot(gat_train_loss_list, label='GAT Train', color='g', linestyle='--')
plt.plot(gat_val_loss_list, label='GAT Validation', color='g', linestyle='-')

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

sns.set(context='paper', rc={'figure.figsize': (10, 4)})

plt.plot(gcn_train_loss_list, label='GCN Train', color='r', linestyle='--')
plt.plot(gcn_val_loss_list, label='GCN Validation', color='r', linestyle='-')
plt.plot(mpnn_train_loss_list, label='MPNN Train', color='b', linestyle='--')
plt.plot(mpnn_val_loss_list, label='MPNN Validation', color='b', linestyle='-')
plt.plot(gat_train_loss_list, label='GAT Train', color='g', linestyle='--')
plt.plot(gat_val_loss_list, label='GAT Validation', color='g', linestyle='-')

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





