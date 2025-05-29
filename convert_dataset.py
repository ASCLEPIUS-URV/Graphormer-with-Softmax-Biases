import json
import torch
import numpy as np
from torch import nn
from tqdm import tqdm

class AtomEmbedding(nn.Module):
    """
    Learnable embedding layer for atom types.
    
    Args:
        num_atom_types (int): Number of unique atom types
        embedding_dim (int): Dimension of the embedding vectors
    """
    def __init__(self, num_atom_types, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)

def convert_molecular_data(input_file, output_file, use_embeddings=False, embedding_dim=32):
    """
    Convert molecular data from JSONL format to GCN-compatible JSON format.
    Each molecule is kept as a separate graph.
    
    Expected JSONL format:
    Each line should be a JSON object with the structure:
    {
        "node_feat": [feat1, feat2, ...],  # List of feature values for each node
        "edge_index": [[source1, source2, ...], [target1, target2, ...]],  # List of source and target node indices
        "edge_attr": [[attr1], [attr2], ...],  # Optional edge attributes
        "y": [property_value]  # List containing the molecular property value
    }
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSON file
        use_embeddings (bool): Whether to convert node features to embeddings
        embedding_dim (int): Dimension of atom type embeddings (only used if use_embeddings=True)
    """
    
    # Initialize embedding layer if needed
    embedding_layer = None
    if use_embeddings:
        # First pass: determine the maximum feature value
        max_feat = -1
        print("Scanning file for maximum feature value...")
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    mol_data = json.loads(line.strip())
                    if 'node_feat' not in mol_data:
                        print(f"Warning: 'node_feat' not found in graph {line_num}")
                        continue
                    
                    # Handle both single values and nested lists in node_feat
                    node_feats = mol_data['node_feat']
                    if isinstance(node_feats[0], list):
                        # Handle nested lists [[6], [7], ...]
                        current_max = max([feat[0] for feat in node_feats])
                    else:
                        # Handle single values [6, 7, ...]
                        current_max = max(node_feats)
                    
                    print(f"Graph {line_num}: current_max = {current_max}, max_feat = {max_feat}")
                    max_feat = max(max_feat, current_max)
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    print(f"Line content: {line.strip()}")
                    continue
        
        # Add 1 to max_feat to get the number of atom types
        num_atom_types = max_feat + 1
        print(f"Maximum atom type found: {max_feat}")
        print(f"Using {embedding_dim}-dimensional embeddings for {num_atom_types} atom types")
        
        if num_atom_types <= 0:
            raise ValueError("No valid atom types found in the data. Check the 'node_feat' values.")
        
        # Initialize embedding layer
        embedding_layer = AtomEmbedding(num_atom_types, embedding_dim)
    
    # Process each molecule
    print("Converting molecules...")
    graphs = []
    
    with open(input_file, 'r') as f, open(output_file, 'w') as out_f:
        for line_num, line in enumerate(tqdm(f, desc="Converting molecules"), 1):
            try:
                mol_data = json.loads(line.strip())
                
                # Convert node features
                if use_embeddings:
                    # Use embedding layer to convert atom types to embeddings
                    node_feat = torch.tensor(mol_data['node_feat'], dtype=torch.long)
                    # Get embeddings and ensure they're in the right format
                    embeddings = embedding_layer(node_feat)
                    # Convert to list and ensure two-level nesting
                    node_features = embeddings.tolist()
                    if isinstance(node_features[0][0], list):
                        node_features = [feat[0] for feat in node_features]
                else:
                    # Use original node features
                    node_features = mol_data['node_feat']
                    # Ensure node_features is a list of lists, but not three levels deep
                    if not isinstance(node_features[0], list):
                        node_features = [[feat] for feat in node_features]
                    elif isinstance(node_features[0][0], list):
                        # If we have three levels, flatten one level
                        node_features = [feat[0] for feat in node_features]
                
                # Create graph dictionary
                graph_dict = {
                    'node_features': node_features,
                    'edge_index': mol_data['edge_index'],
                    'y': mol_data['y']
                }
                
                # Add edge attributes if they exist
                if 'edge_attr' in mol_data:
                    graph_dict['edge_attr'] = mol_data['edge_attr']
                
                # Write to output file
                out_f.write(json.dumps(graph_dict) + '\n')
                graphs.append(graph_dict)
                
            except Exception as e:
                print(f"Error processing molecule {line_num}: {e}")
                continue
    
    print(f"Converted {len(graphs)} molecules")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert molecular data to JSON format')
    parser.add_argument('input_file', type=str, help='Path to input JSONL file')
    parser.add_argument('output_file', type=str, help='Path to output JSON file')
    parser.add_argument('--use-embeddings', action='store_true', help='Convert node features to embeddings')
    parser.add_argument('--embedding-dim', type=int, default=32, help='Dimension of atom type embeddings')
    
    args = parser.parse_args()
    convert_molecular_data(args.input_file, args.output_file, args.use_embeddings, args.embedding_dim)
