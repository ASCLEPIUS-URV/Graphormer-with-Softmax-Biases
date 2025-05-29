# import os
# import sys
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
import torch
import torch.nn.functional as F
import json
import argparse
import numpy as np
import os
from pathlib import Path
from gcn import MultiLayerGCN
from gat import MultiLayerGAT
from data_processor import GraphDataProcessor
from torch_geometric.data import Data, Batch
import math
#import ipdb

def split_data(graphs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """

    Args:
        graphs (list): List of graph dictionaries
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_graphs, val_graphs, test_graphs)
    """
    #assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create random permutation of indices
    num_graphs = len(graphs)
    indices = torch.randperm(num_graphs)

    # Calculate split sizes
    train_size = int(num_graphs * train_ratio)
    val_size = int(num_graphs * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Split graphs
    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    test_graphs = [graphs[i] for i in test_indices]

    return train_graphs, val_graphs, test_graphs

def save_checkpoint(model, optimizer, epoch, train_graphs, val_graphs, test_graphs, args, checkpoint_dir='checkpoints', verbose=True):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save full checkpoint
    full_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_graphs': train_graphs,
        'val_graphs': val_graphs,
        'test_graphs': test_graphs,
        'args': args
    }

    # Save weights-only checkpoint
    weights_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }

    full_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    weights_checkpoint_path = os.path.join(checkpoint_dir, f'weights_epoch_{epoch}.pt')

    torch.save(full_checkpoint, full_checkpoint_path)
    torch.save(weights_checkpoint, weights_checkpoint_path)

    if verbose:
        print(f"Saved full checkpoint to {full_checkpoint_path}")
        print(f"Saved weights-only checkpoint to {weights_checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load checkpoint from file. If optimizer is None, only model weights will be loaded.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        model (torch.nn.Module): Model to load weights into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into

    Returns:
        tuple: (epoch, train_graphs, val_graphs, test_graphs) if full checkpoint,
               (epoch,) if weights-only checkpoint
    """
    try:
        # First try loading as full checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'train_graphs' in checkpoint:
            return checkpoint['epoch'], checkpoint['train_graphs'], checkpoint['val_graphs'], checkpoint['test_graphs']
        else:
            return checkpoint['epoch'],

    except Exception as e:
        print(f"Error loading full checkpoint: {str(e)}")
        print("Attempting to load weights-only checkpoint...")

        # Try loading as weights-only checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint['epoch'],
        except Exception as e:
            print(f"Error loading weights-only checkpoint: {str(e)}")
            raise

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def convert_to_pyg_data(graph_dict, device):
    # Add data validation
    if 'node_features' not in graph_dict and 'node_feat' not in graph_dict:
        raise ValueError("Missing both 'node_features' and 'node_feat' in graph dictionary")
    if 'edge_index' not in graph_dict:
        raise ValueError("Missing edge_index in graph dictionary")
    if 'y' not in graph_dict:
        raise ValueError("Missing y in graph dictionary")

    # Convert node features - handle both node_feat and node_features keys
    if 'node_feat' in graph_dict:
        x = torch.tensor(graph_dict['node_feat'], dtype=torch.float).to(device)
    else:
        # Handle nested node features
        node_features = graph_dict['node_features']
        if isinstance(node_features[0][0], list):
            # If features are nested, flatten one level
            x = torch.tensor([feat[0] for feat in node_features], dtype=torch.float).to(device)
        else:
            x = torch.tensor(node_features, dtype=torch.float).to(device)

    # Convert edge index to the correct format
    edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long).to(device)
    if edge_index.dim() == 1:
        # If edge_index is a flat list, reshape it to [2, num_edges]
        edge_index = edge_index.view(2, -1)
    elif edge_index.dim() == 2 and edge_index.size(0) != 2:
        # If edge_index is [num_edges, 2], transpose it to [2, num_edges]
        edge_index = edge_index.t()

    y = torch.tensor(graph_dict['y'], dtype=torch.float).view(-1).to(device)

    # Validate graph structure
    num_nodes = x.size(0)
    if edge_index.max() >= num_nodes:
        raise ValueError(f"Edge index {edge_index.max()} is out of bounds for graph with {num_nodes} nodes")

    # Handle edge attributes
    if 'edge_attr' in graph_dict:
        edge_attr = torch.tensor(graph_dict['edge_attr'], dtype=torch.float).to(device)
        # Ensure edge_attr has the correct shape
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        # Remove self-loops before creating the Data object
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    else:
        # Remove self-loops even without edge attributes
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        return Data(x=x, edge_index=edge_index, y=y)

def train_model(model, train_graphs, val_graphs, test_graphs, optimizer, args):
    """

    Args:
        model (MultiLayerGCN): The GCN model
        train_graphs (list): List of training graph dictionaries
        val_graphs (list): List of validation graph dictionaries
        test_graphs (list): List of test graph dictionaries
        optimizer (torch.optim.Optimizer): The optimizer
        args: Command line arguments
    """
    device = get_device()
    print(f"Using device: {device}")

    # Move model to device
    model = model.to(device)

    model.train()
    best_val_loss = float('inf')
    best_model_state = None
    start_epoch = 0

    # Load checkpoint if specified
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        try:
            # Try loading full checkpoint first
            start_epoch, train_graphs, val_graphs, test_graphs = load_checkpoint(
                args.resume, model, optimizer
            )
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Could not load full checkpoint: {str(e)}")
            print("Attempting to load weights-only checkpoint...")
            # If full checkpoint fails, try loading weights only
            start_epoch, = load_checkpoint(args.resume, model)
            print(f"Loaded weights from epoch {start_epoch}")
            print("Note: Training will continue with new data split")

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_train_loss = 0

            # Process graphs in batches
            for i in range(0, len(train_graphs), args.batch_size):
                batch_graphs = train_graphs[i:i + args.batch_size]
                try:
                    batch_data = [convert_to_pyg_data(g, device) for g in batch_graphs]
                    batch = Batch.from_data_list(batch_data)


                    optimizer.zero_grad()
                    # Only pass x, edge_index, and batch to the model
                    out = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)


                    # Ensure output and target have the same shape
                    target = batch.y  # Should already be [batch_size]
                    loss = F.mse_loss(out, target)

                    # Check for NaN values in model output
                    if torch.isnan(out).any():
                        print(f"Warning: NaN values in model output at epoch {epoch}, batch {i}")
                        print(f"Output shape: {out.shape}")
                        print(f"Number of NaN values: {torch.isnan(out).sum().item()}")
                        print("Input statistics:")
                        print(f"x mean: {batch.x.mean().item()}, std: {batch.x.std().item()}")
                        print(f"y mean: {batch.y.mean().item()}, std: {batch.y.std().item()}")

                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss at epoch {epoch}, batch {i}")
                        print("Model parameters statistics:")
                        for name, param in model.named_parameters():
                            print(f"{name}: mean={param.mean().item()}, std={param.std().item()}")
                        raise ValueError("NaN loss detected")

                    loss.backward()

                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    total_train_loss += loss.item() * len(batch_graphs)
                except Exception as e:
                    print(f"Error processing batch starting at index {i}:")
                    print(f"Error: {str(e)}")
                    print("First graph in batch:")
                    print(json.dumps(batch_graphs[0], indent=2))
                    raise

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                # Process validation graphs in batches
                for i in range(0, len(val_graphs), args.batch_size):
                    batch_graphs = val_graphs[i:i + args.batch_size]
                    batch_data = [convert_to_pyg_data(g, device) for g in batch_graphs]

                    # Create proper batch indices
                    batch = Batch.from_data_list(batch_data)


                    # Only pass x, edge_index, and batch to the model
                    out = model(batch.x, batch.edge_index, edge_attr=None, batch=batch.batch)

                    # Ensure output and target have the same shape
                    target = batch.y  # Should already be [batch_size]
                    val_loss = F.mse_loss(out, target)
                    total_val_loss += val_loss.item() * len(batch_graphs)

            avg_train_loss = total_train_loss / len(train_graphs)
            avg_val_loss = total_val_loss / len(val_graphs)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                # Save best model checkpoint without printing
                save_checkpoint(model, optimizer, epoch, train_graphs, val_graphs, test_graphs, args, verbose=False)

            # Save periodic checkpoint
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(model, optimizer, epoch, train_graphs, val_graphs, test_graphs, args, verbose=True)

            # Print training progress
            if (epoch + 1) % args.print_every == 0:
                print(f'Epoch {epoch+1:03d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        save_checkpoint(model, optimizer, epoch, train_graphs, val_graphs, test_graphs, args)
        print("Checkpoint saved. You can resume training later using --resume")
        return

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation on test set
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for graph_dict in test_graphs:
            data = convert_to_pyg_data(graph_dict, device)
            # Create a batch tensor for single graph
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            out = model(data.x, data.edge_index, edge_attr=None, batch=batch)
            # Ensure consistent shapes for loss calculation
            out = out.squeeze()
            if out.dim() == 0:
                out = out.unsqueeze(0)  # Add batch dimension if needed
            if data.y.dim() == 0:
                data.y = data.y.unsqueeze(0)  # Add batch dimension if needed
            test_loss = F.mse_loss(out, data.y)
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_graphs)
    print(f'\nFinal Test Loss: {avg_test_loss:.4f}')

def main(args):
    # Load and process the data
    print("Loading data...")
    with open(args.data_path, 'r') as f:
        graphs = [json.loads(line) for line in f if line.strip()]

    # Filter out samples with NaN target values
    original_count = len(graphs)
    graphs = [g for g in graphs if not (
        isinstance(g['y'], list) and any(
            isinstance(y, (int, float)) and math.isnan(y) for y in g['y']
        )
    ) and not (
        isinstance(g['y'], (int, float)) and math.isnan(g['y'])
    )]
    filtered_count = len(graphs)

    if filtered_count < original_count:
        print(f"Removed {original_count - filtered_count} samples with NaN target values")
        print(f"Remaining samples: {filtered_count}")

    if filtered_count == 0:
        raise ValueError("No valid samples remaining after filtering NaN values")

    # Split data 
    print("Splitting data...")
    train_graphs, val_graphs, test_graphs = split_data(
        graphs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Show split sizes
    print(f"Training graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")

    # Get input feature dimension from first graph
    if 'node_features' in graphs[0]:
        input_dim = len(graphs[0]['node_features'][0])
    elif 'node_feat' in graphs[0]:
        input_dim = len(graphs[0]['node_feat'][0])
    else:
        raise ValueError("Neither 'node_features' nor 'node_feat' found in graph data")

    print("Initializing model...")
    if args.model == 'gcn':
        model = MultiLayerGCN(
            in_channels=input_dim,
            hidden_channels=args.hidden_size,
            out_channels=1,  # For regression
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    elif args.model == 'gat':
        model = MultiLayerGAT(
            in_channels=input_dim,
            hidden_channels=args.hidden_size,
            out_channels=1,  # For regression
            num_layers=args.num_layers,
            dropout=args.dropout,
            edge_dim=args.edge_dim
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    train_model(model, train_graphs, val_graphs, test_graphs, optimizer, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['gcn', 'gat'], help='Model to use')
    parser.add_argument('data_path', type=str, help='Path to the input data file')
    parser.add_argument('--hidden-size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Training graphs per batch')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Ratio of validation data')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Ratio of test data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--edge-dim', type=int, default=1, help='Edge dimension')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--print-every', type=int, default=10, help='Print training progress every N epochs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to use for training')

    args = parser.parse_args()
    main(args)
