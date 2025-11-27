import json
import os
from pathlib import Path

def split_json_file(input_file, output_dir, items_per_file=1000):
    """
    Split a large JSON file into smaller files.
    
    Args:
        input_file: Path to the input JSON file
        output_dir: Directory where split files will be saved
        items_per_file: Number of items per output file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the JSON file
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        # JSON is an array
        items = data
        is_array = True
    elif isinstance(data, dict):
        # JSON is an object - check if it has an array we should split
        array_keys = [k for k, v in data.items() if isinstance(v, list)]
        
        if array_keys:
            print(f"Found array keys: {array_keys}")
            key = array_keys[0]  # Use first array key
            print(f"Splitting array from key: '{key}'")
            items = data[key]
            is_array = False
            base_structure = {k: v for k, v in data.items() if k != key}
        else:
            print("JSON is a single object. Creating chunks of key-value pairs...")
            items = list(data.items())
            is_array = False
            base_structure = {}
    else:
        raise ValueError("Unsupported JSON structure")
    
    # Split into chunks
    total_items = len(items)
    num_files = (total_items + items_per_file - 1) // items_per_file
    
    print(f"Total items: {total_items}")
    print(f"Items per file: {items_per_file}")
    print(f"Creating {num_files} files...")
    
    base_name = Path(input_file).stem
    
    for i in range(num_files):
        start_idx = i * items_per_file
        end_idx = min((i + 1) * items_per_file, total_items)
        chunk = items[start_idx:end_idx]
        
        # Create output structure
        if is_array:
            output_data = chunk
        elif array_keys:
            output_data = base_structure.copy()
            output_data[key] = chunk
        else:
            output_data = dict(chunk)
        
        # Write to file
        output_file = os.path.join(output_dir, f"{base_name}_part_{i+1:04d}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created {output_file} ({len(chunk)} items)")
    
    print(f"\nDone! Created {num_files} files in {output_dir}")

# Example usage
if __name__ == "__main__":
    # Configure these parameters
    INPUT_FILE = "large_file.json"
    OUTPUT_DIR = "split_files"
    ITEMS_PER_FILE = 1000
    
    split_json_file("/private/schwartz-lab/yarin_shaked7/SkyRL/deepcoder_train.json", output_dir="/private/schwartz-lab/yarin_shaked7/SkyRL/train_shards", items_per_file=50)
    split_json_file("/private/schwartz-lab/yarin_shaked7/SkyRL/test_livecodebench.json", output_dir="/private/schwartz-lab/yarin_shaked7/SkyRL/test_shards", items_per_file=50)