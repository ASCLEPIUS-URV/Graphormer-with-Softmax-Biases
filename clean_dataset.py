import json
import math
import argparse
from pathlib import Path

def clean_dataset(input_file, output_file):
    """
    Clean dataset by removing entries with NaN target values.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
    """
    print(f"Reading from {input_file}")
    total_entries = 0
    valid_entries = 0
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
                
            total_entries += 1
            try:
                data = json.loads(line)
                
                # Check for NaN in target value
                if isinstance(data['y'], list):
                    if any(math.isnan(y) for y in data['y']):
                        continue
                elif math.isnan(data['y']):
                    continue
                
                # Write valid entry to output file
                f_out.write(line)
                valid_entries += 1
                
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {total_entries}")
                continue
            except KeyError as e:
                print(f"Warning: Missing key {e} at line {total_entries}")
                continue
    
    print(f"\nDataset cleaning complete:")
    print(f"Total entries: {total_entries}")
    print(f"Valid entries: {valid_entries}")
    print(f"Removed entries: {total_entries - valid_entries}")
    print(f"Cleaned dataset saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean dataset by removing entries with NaN targets')
    parser.add_argument('input_file', type=str, help='Path to input JSONL file')
    parser.add_argument('--output-file', type=str, help='Path to output JSONL file (default: input_file_cleaned.jsonl)')
    
    args = parser.parse_args()
    
    # If output file not specified, create default name
    if args.output_file is None:
        input_path = Path(args.input_file)
        args.output_file = str(input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}"))
    
    clean_dataset(args.input_file, args.output_file) 