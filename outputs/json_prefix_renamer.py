import os
import json

# Define the directory to search in and the prefix
base_directory = './validate'
prefix = 'j'  # Example: this prefix will rename 'metrics_example.json' to 'j_metrics_example.json'

def append_json_content(file_without_prefix, file_with_prefix):
    """Append the content of the JSON file_without_prefix to the file_with_prefix."""
    # Load content from both files
    with open(file_without_prefix, 'r') as f:
        data_without_prefix = json.load(f)

    with open(file_with_prefix, 'r') as f:
        data_with_prefix = json.load(f)

    # Append the "experiments" list from the file without prefix to the file with prefix
    if isinstance(data_with_prefix, dict) and isinstance(data_without_prefix, dict):
        if "experiments" in data_with_prefix and "experiments" in data_without_prefix:
            data_with_prefix["experiments"].extend(data_without_prefix["experiments"])

    # Write the updated content back to the file with prefix
    with open(file_with_prefix, 'w') as f:
        json.dump(data_with_prefix, f, indent=4)

def rename_or_append_json_files(base_directory, prefix):
    # Walk through all files and subdirectories within the base directory
    for dirpath, _, filenames in os.walk(base_directory):
        for filename in filenames:
            # Check if the file is a JSON file and starts with "metrics" or "iteration_metrics"
            if filename.endswith('.json') and (filename.startswith("metrics") or filename.startswith("iteration_metrics")):
                # Construct the full file path
                old_file_path = os.path.join(dirpath, filename)

                # Create the new filename with the specified prefix
                new_filename = f"{prefix}_{filename}"
                new_file_path = os.path.join(dirpath, new_filename)

                # Check if the prefixed file already exists
                if os.path.exists(new_file_path):
                    # If the prefixed file exists, append the content of the old file to it
                    print(f"Appending content of {old_file_path} to {new_file_path}")
                    append_json_content(old_file_path, new_file_path)

                    # After appending, delete the original file without the prefix
                    os.remove(old_file_path)
                    print(f"Deleted: {old_file_path}")
                else:
                    # If the prefixed file does not exist, rename the file
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: {old_file_path} -> {new_file_path}")



# Call the function to rename or append files
rename_or_append_json_files(base_directory, prefix)
