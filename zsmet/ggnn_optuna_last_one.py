# %% [markdown]
# # GGNN Optuna Optimization
# This project demonstrates the use of a Gated Graph Neural Network (GGNN) model for a graph classification task. It includes the setup for GPU utilization, data loading and preparation, model definition, and the use of Optuna for hyperparameter optimization.
# 

# %% [markdown]
# ## Check GPU Availability
# This section checks the availability of a GPU for PyTorch, ensuring that model training can leverage hardware acceleration if available.
# 

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import dgl
from dgl.nn import GatedGraphConv, GlobalAttentionPooling
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import train_test_split
import optuna
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys

# Check CUDA availability
print("PyTorch version:", torch.__version__)
cuda_available = torch.cuda.is_available()
print("Is CUDA Supported?", cuda_available)
if cuda_available:
    print(torch.cuda.device_count(), "CUDA device(s) available.")
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
    

# %% [markdown]
# ## Define Model and Utilities
# This section defines the GGNN model, an early stopping utility to prevent overfitting, and a custom collate function for data loading.
# 

# %%
# Define the model
class GraphClsGGNN(nn.Module):
    """GGNN for graph classification."""
    def __init__(self, annotation_size, out_feats, n_steps, n_etypes, num_cls, dropout_rate=0.5):
        super(GraphClsGGNN, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.ggnn1 = GatedGraphConv(annotation_size, out_feats, n_steps, n_etypes)
        self.ggnn2 = GatedGraphConv(out_feats, out_feats, n_steps, n_etypes)
        self.pooling = GlobalAttentionPooling(nn.Linear(out_feats, 1))
        self.fc = nn.Linear(out_feats, num_cls)

    def forward(self, graph, feat):
        h = F.relu(self.ggnn1(graph, feat))
        h = self.dropout(h)
        h = F.relu(self.ggnn2(graph, h))
        hg = self.pooling(graph, h)
        return self.fc(hg)
    
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                print(f"Resetting parameters of layer = {layer}")
                layer.reset_parameters()

# Define EarlyStopping class
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True, delta=0.001, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Define the collate function for the DataLoader
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.long)
    return batched_graph, labels



# %% [markdown]
# ## Training and Evaluation Pipeline
# Outlines the process for training the GGNN model, including the training loop, validation checks, and early stopping implementation.
# 

# %%
# TrainingPipeline class encapsulating the training and evaluation process
class TrainingPipeline:
    def __init__(self, device):
        self.device = device

    def train_and_evaluate(self, model, train_loader, val_loader, optimizer, criterion, early_stopping, num_epochs, plot_curves=False, accumulation_steps=8):
        train_losses, val_losses = [], []
        scaler = GradScaler()  # Initialize the gradient scaler
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()  # Initialize gradients to zero

            for batch_idx, (batched_graph, labels) in enumerate(train_loader):
                batched_graph, labels = batched_graph.to(self.device), labels.to(self.device)
                
                with autocast():  # Enable automatic mixed precision
                    logits = model(batched_graph, batched_graph.ndata['h'].float())
                    loss = criterion(logits, labels) / accumulation_steps  # Scale loss

                scaler.scale(loss).backward()  # Scale the loss and call backward to propagate gradients
                train_loss += loss.item() * accumulation_steps  # Correct scaling for logging purposes
                
                if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    scaler.step(optimizer)  # Perform optimizer step using scaled gradients
                    scaler.update()  # Update the scaler for the next iteration
                    optimizer.zero_grad()  # Reset gradients for the next accumulation

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            val_loss = 0.0
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    for batched_graph, labels in val_loader:
                        batched_graph, labels = batched_graph.to(self.device), labels.to(self.device)
                        with autocast():  # Use autocast for the validation step
                            logits = model(batched_graph, batched_graph.ndata['h'].float())
                            loss = criterion(logits, labels)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                val_losses.append(val_loss)

                if early_stopping:
                    early_stopping(val_loss, model, epoch + 1)
                    if early_stopping.early_stop:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | Val accuracy: {100 * (1 - val_loss):.2f}%')

        if plot_curves and val_loader is not None:
            self.plot_loss_curves(train_losses, val_losses)

        return train_losses, val_losses



    @staticmethod
    def plot_loss_curves(train_losses, val_losses):
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_curves.png', dpi=300)
        plt.show()

    def evaluate_on_test(self, model, test_loader, criterion):
        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for batched_graph, labels in test_loader:
                batched_graph, labels = batched_graph.to(self.device), labels.to(self.device)
                logits = model(batched_graph, batched_graph.ndata['h'].float())
                loss = criterion(logits, labels)
                test_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                test_accuracy += torch.sum(preds == labels).item()

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader.dataset)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)
        

# %% [markdown]
# ## Hyperparameter Optimization with Optuna
# Describes the setup for hyperparameter optimization using Optuna, including defining the search space and optimizing the model parameters.
# 

# %%
class HyperparameterOptimizer:
    def __init__(self, device, subset_train_graphs, subset_train_labels, subset_val_graphs, subset_val_labels, num_trials, num_epochs):
        self.device = device
        self.subset_train_graphs = subset_train_graphs
        self.subset_train_labels = subset_train_labels
        self.subset_val_graphs = subset_val_graphs
        self.subset_val_labels = subset_val_labels
        self.num_trials = num_trials
        self.num_epochs = num_epochs

    def objective(self, trial):
        # Suggest hyperparameters
        n_steps = trial.suggest_int('n_steps', 1, 40)
        out_feats = trial.suggest_int('out_feats', 74, 512)
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512])
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

        # Create the model, optimizer, and loaders
        model = GraphClsGGNN(annotation_size=74, out_feats=out_feats, n_steps=n_steps, n_etypes=1, num_cls=2, dropout_rate=dropout_rate).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        train_loader = GraphDataLoader(list(zip(self.subset_train_graphs, self.subset_train_labels)), batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=3)
        val_loader = GraphDataLoader(list(zip(self.subset_val_graphs, self.subset_val_labels)), batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=3)

        # Training loop with pruning
        model.train()
        for epoch in range(self.num_epochs):
            for batched_graph, labels in train_loader:
                batched_graph, labels = batched_graph.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = model(batched_graph, batched_graph.ndata['h'].float())
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            # Validation phase and report for pruning
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batched_graph, labels in val_loader:
                    batched_graph, labels = batched_graph.to(self.device), labels.to(self.device)
                    logits = model(batched_graph, batched_graph.ndata['h'].float())
                    loss = criterion(logits, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            trial.report(val_loss, epoch)  # Report intermediate value to the pruner

            if trial.should_prune():  # Ask the pruner if the trial should be stopped
                raise optuna.TrialPruned()

        return val_loss  # Return the validation loss

    def optimize(self):
        # Use MedianPruner in the study to activate pruning
        study = optuna.create_study(direction='minimize', pruner=MedianPruner())
        study.optimize(self.objective, n_trials=self.num_trials)

        # Save and print the best hyperparameters
        best_hyperparams = study.best_trial.params
        with open('best_hyperparameters.json', 'w') as f:
            json.dump(best_hyperparams, f)
        print("Best hyperparameters saved.")


# %% [markdown]
# ## Hyperparameter Optimization Execution, Retraining and Evaluation on test set
# Initiates the hyperparameter optimization process, leveraging the previously defined model, data loaders, and Optuna setup.
# 

# %%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data and prepare for training
    reloaded_df = pd.read_csv("data_mvi/combined_df.csv")
    graphs, labels_dict = dgl.load_graphs("data_mvi/graphs.bin")
    labels = reloaded_df['binds_to_rna'].values

    # Split dataset train, test
    train_indices, test_indices, train_labels, test_labels = train_test_split(
        range(len(reloaded_df)), labels, test_size=0.2, stratify=labels, random_state=42)

    # Split dataset train, validation
    train_indices, val_indices, train_labels, val_labels = train_test_split(
        train_indices, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
    
    # Placeholder for data loading. Replace this with your actual data loading code.
    train_graphs = [graphs[i] for i in train_indices]
    test_graphs = [graphs[i] for i in test_indices]
    val_graphs = [graphs[i] for i in val_indices]
    
    subset_train_indices = np.random.choice(len(train_graphs), size=int(len(train_graphs) * 0.3), replace=False)
    subset_train_graphs = [train_graphs[i] for i in subset_train_indices]
    subset_train_labels = train_labels[subset_train_indices]

    subset_val_indices = np.random.choice(len(val_graphs), size=int(len(val_graphs) * 0.3), replace=False)
    subset_val_graphs = [val_graphs[i] for i in subset_val_indices]
    subset_val_labels = val_labels[subset_val_indices]
    
    # Combine train and validation graphs and labels for retraining
    combined_train_graphs = train_graphs + val_graphs
    combined_train_labels = np.concatenate((train_labels, val_labels))
    
    # print(f'Train: {len(train_graphs)}, Validation: {len(val_graphs)}, Test: {len(test_graphs)}, Subset Train: {len(subset_train_graphs)}, Subset Val: {len(subset_val_graphs)}')
    # print("")
    # print("Completed data loading.")
    # print("")
    # sys.stdout.flush()  # Force flushing of the buffer
    # # 1. Hyperparameter Optimization on a subset of the data
    # print("Starting hyperparameter optimization...")
    # sys.stdout.flush()
    # print("")
    # # Specify the number of trials and epochs for hyperparameter optimization
    # optimizer = HyperparameterOptimizer(device, subset_train_graphs, subset_train_labels, subset_val_graphs, subset_val_labels, num_trials=8, num_epochs=10)
    # optimizer.optimize()
    # print("Completed hyperparameter optimization.")
    # sys.stdout.flush()
    
    print("")
    # Load the best hyperparameters
    with open('best_hyperparameters.json', 'r') as f:
        best_hyperparams = json.load(f)

    # Correcting the use of best_hyperparams by defining data loaders here
    train_loader = GraphDataLoader(list(zip(train_graphs, train_labels)), batch_size=best_hyperparams['batch_size'], shuffle=True, collate_fn=collate, num_workers=10)
    val_loader = GraphDataLoader(list(zip(val_graphs, val_labels)), batch_size=best_hyperparams['batch_size'], shuffle=False, collate_fn=collate, num_workers=10)
    test_loader = GraphDataLoader(list(zip(test_graphs, test_labels)), batch_size=best_hyperparams['batch_size'], shuffle=False, collate_fn=collate, num_workers=10)
    combined_train_loader = GraphDataLoader(list(zip(combined_train_graphs, combined_train_labels)), batch_size=best_hyperparams['batch_size'], shuffle=True, collate_fn=collate, num_workers=10)

    
    # 2. Retraining with best hyperparameters (on a larger train and val set)
    print("Retraining with best hyperparameters...")
    model = GraphClsGGNN(annotation_size=74, out_feats=best_hyperparams['out_feats'], n_steps=best_hyperparams['n_steps'], n_etypes=1, num_cls=2, dropout_rate=best_hyperparams['dropout_rate']).to(device)
    print("")
    
    # Reset model parameters before retraining
    model.reset_parameters()
    
    optimizer = optim.Adam(model.parameters(), lr=best_hyperparams['lr'])
    criterion = nn.CrossEntropyLoss()
        
    early_stopping = EarlyStopping(patience=10, verbose=True)
    training_pipeline = TrainingPipeline(device)
    training_pipeline.train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, early_stopping, 30, plot_curves=True)
    optimal_epoch = early_stopping.best_epoch
    
    # Before final training on the combined train and val dataset, reset the model again
    model.reset_parameters()
    print("Completed training.")
    sys.stdout.flush()
    print("")
    
    # 3. Final training on the combined train and val dataset
    print("Final training on the combined train and val dataset...")
    optimizer = optim.Adam(model.parameters(), lr=best_hyperparams['lr'])
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=10, verbose=True, delta=0, path='final_model.pt')
    training_pipeline.train_and_evaluate(model, combined_train_loader, None, optimizer, criterion, None, optimal_epoch, plot_curves=False)
    print("Completed training.")
    sys.stdout.flush()
    print("")
    
    # Evaluation on the test set
    print("Evaluating on the test set...")
    training_pipeline.evaluate_on_test(model, test_loader, criterion)



