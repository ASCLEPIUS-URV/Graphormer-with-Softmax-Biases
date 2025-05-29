# GCN-PyTorch

A Graph Convolutional Network (GCN) implementation in PyTorch for molecular property prediction.

## Features

- Multi-layer GCN architecture
- Support for both CPU and GPU training
- Automatic checkpointing and training resume
- Graph-level property prediction
- Flexible data splitting and model configuration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sarah-af/GCN-PyTorch.git
cd GCN-PyTorch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Basic training:
```bash
python train.py path/to/your/data.json
```

With custom parameters:
```bash
python train.py path/to/your/data.json \
    --hidden-size 64 \
    --lr 0.001 \
    --num-layers 2 \
    --dropout 0.2 \
    --epochs 200 \
    --device cuda
```

### Resuming Training

To resume from a checkpoint:
```bash
python train.py path/to/your/data.json --resume checkpoints/checkpoint_epoch_X.pt
```

## Command Line Arguments

- `data_path`: Path to the input data file (required)
- `--hidden-size`: Hidden layer size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay (default: 5e-4)
- `--epochs`: Number of training epochs (default: 200)
- `--train-ratio`: Ratio of training data (default: 0.7)
- `--val-ratio`: Ratio of validation data (default: 0.15)
- `--test-ratio`: Ratio of test data (default: 0.15)
- `--seed`: Random seed for reproducibility (default: 42)
- `--num-layers`: Number of GCN layers (default: 2)
- `--dropout`: Dropout probability (default: 0.0)
- `--save-every`: Save checkpoint every N epochs (default: 10)
- `--resume`: Path to checkpoint to resume from
- `--device`: Device to use for training (choices: 'cpu', 'cuda')

## Data Format

The input data should be a JSON file containing a list of graph dictionaries. Each graph dictionary should have the following format:
JSONL files of a similar structure can be converted using convert_dataset.py

```python
{
    'node_features': [[...], [...], ...],  # List of node feature vectors
    'edge_index': [[...], [...]],         # COO format edge indices
    'y': [value]                          # Target value
}
```

For JSONL files, each line should be a separate JSON object with the same structure as above.

## License

MIT License 