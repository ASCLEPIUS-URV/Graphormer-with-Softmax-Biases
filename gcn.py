import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class MultiLayerGCN(torch.nn.Module):
    """
    
    Args:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output features
        num_layers (int): Number of GCN layers (default: 2)
        dropout (float): Dropout probability (default: 0.0)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super().__init__()
        
        # Input layer
        self.layers = torch.nn.ModuleList([
            GCNConv(in_channels, hidden_channels)
        ])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Final prediction layer
        self.pred = torch.nn.Linear(hidden_channels, out_channels)
        
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        
        Args:
            x (torch.Tensor): Node feature matrix
            edge_index (torch.Tensor): Graph connectivity in COO format
            edge_attr (torch.Tensor, optional): Edge features
            batch (torch.Tensor, optional): Batch assignment for each node
            
        Returns:
            torch.Tensor: Graph-level predictions of shape [batch_size]
        """
        # Process through GCN layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < len(self.layers) - 1:  # Don't apply dropout after last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Get predictions for each graph
        if batch is not None:
            # Get unique graph indices
            graph_indices = torch.unique(batch)
            num_graphs = len(graph_indices)
            
            # Initialize output tensor
            out = torch.zeros(num_graphs, device=x.device)
            
            # Process each graph separately
            for i, graph_idx in enumerate(graph_indices):
                # Get nodes for this graph
                mask = batch == graph_idx
                graph_x = x[mask]
                
                # Get prediction for this graph
                graph_pred = self.pred(graph_x.mean(dim=0, keepdim=True))
                out[i] = graph_pred.squeeze(-1)
            
            # Ensure output has shape [batch_size]
            if out.dim() == 1 and out.size(0) == 1:
                out = out.expand(num_graphs)
        else:
            # Single graph case
            out = self.pred(x.mean(dim=0, keepdim=True))
            out = out.squeeze(-1)
            if out.dim() == 0:  # If we got a scalar, make it [1]
                out = out.unsqueeze(0)
        
        return out

# For backward compatibility
class TwoLayerGCN(MultiLayerGCN):
    """
    
    Args:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output features
        dropout (float): Dropout probability (default: 0.0)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers=2, dropout=dropout) 