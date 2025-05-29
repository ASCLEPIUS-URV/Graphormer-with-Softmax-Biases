import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

class GraphDataProcessor:
    def __init__(self, normalize_features=False, device=None):
        """
        Initialize the graph data processor.
        
        Args:
            normalize_features (bool): Whether to normalize node features
            device (torch.device, optional): Device to place tensors on (CPU/GPU)
        """
        self.normalize_features = normalize_features
        self.scaler = StandardScaler() if normalize_features else None
        self.device = device
        
    def _validate_input(self, data_dict):
        """Validate input dictionary has required fields."""
        if 'node_features' not in data_dict and 'node_feat' not in data_dict:
            raise ValueError("Missing both 'node_features' and 'node_feat' in graph dictionary")
        if 'edge_index' not in data_dict:
            raise ValueError("Missing edge_index in graph dictionary")
        if 'y' not in data_dict:
            raise ValueError("Missing y in graph dictionary")
    
    def _process_node_features(self, data_dict):
        """Process and normalize node features."""
        if 'node_feat' in data_dict:
            x = torch.tensor(data_dict['node_feat'], dtype=torch.float32)
        else:
            node_features = data_dict['node_features']
            if isinstance(node_features[0][0], list):
                # Handle nested features
                x = torch.tensor([feat[0] for feat in node_features], dtype=torch.float32)
            else:
                x = torch.tensor(node_features, dtype=torch.float32)
        
        if self.normalize_features:
            x = torch.tensor(
                self.scaler.fit_transform(x.numpy()),
                dtype=torch.float32
            )
        
        return x
    
    def _process_edge_index(self, data_dict):
        """Process edge indices and remove self-loops."""
        edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long)
        
        # Handle different edge_index formats
        if edge_index.dim() == 1:
            edge_index = edge_index.view(2, -1)
        elif edge_index.dim() == 2 and edge_index.size(0) != 2:
            edge_index = edge_index.t()
        
        # Remove self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        
        return edge_index, mask
    
    def _process_edge_attributes(self, data_dict, mask):
        """Process edge attributes if they exist."""
        if 'edge_attr' in data_dict:
            edge_attr = torch.tensor(data_dict['edge_attr'], dtype=torch.float32)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = edge_attr[mask]
            return edge_attr
        return None
    
    def _validate_graph_structure(self, x, edge_index):
        """Validate graph structure."""
        num_nodes = x.size(0)
        if edge_index.max() >= num_nodes:
            raise ValueError(
                f"Edge index {edge_index.max()} is out of bounds "
                f"for graph with {num_nodes} nodes"
            )
    
    def process_dict_to_graph(self, data_dict):
        """
        Convert a dictionary to a PyTorch Geometric Data object.
        
        Args:
            data_dict (dict): Input dictionary containing graph information
                Expected format:
                {
                    'node_features' or 'node_feat': list/array of node features,
                    'edge_index': list of [source_nodes, target_nodes],
                    'edge_attr': (optional) list/array of edge features,
                    'y': list/array of node/graph labels
                }
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        # Validate input
        self._validate_input(data_dict)
        
        # Process node features
        x = self._process_node_features(data_dict)
        
        # Process edge index and get mask for self-loops
        edge_index, mask = self._process_edge_index(data_dict)
        
        # Validate graph structure
        self._validate_graph_structure(x, edge_index)
        
        # Process edge attributes
        edge_attr = self._process_edge_attributes(data_dict, mask)
        
        # Process labels
        y = torch.tensor(data_dict['y'], dtype=torch.float32).view(-1)
        
        # Move tensors to device if specified
        if self.device is not None:
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            y = y.to(self.device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device)
        
        # Create and return Data object
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
    
    def process_batch(self, dict_list):
        """
        Process a batch of graph dictionaries.
        
        Args:
            dict_list (list): List of dictionaries containing graph information
            
        Returns:
            list: List of PyG Data objects
        """
        return [self.process_dict_to_graph(d) for d in dict_list] 