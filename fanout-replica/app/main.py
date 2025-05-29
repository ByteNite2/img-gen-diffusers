# === BYTENITE PARTITIONER - MAIN SCRIPT ===

# Read the documentation --> https://docs.bytenite.com/create-with-bytenite/building-blocks/partitioning-engines

try:
    import json
    import os
    import re
except ImportError as e:
    raise ImportError(f"Required library is missing: {e}")

# Path to the folder where your data lives. This folder will be automatically populated with the files imported from your data source.
source_dir = os.getenv("SOURCE_DIR")
if source_dir is None:
    raise ValueError("Environment variable 'SOURCE_DIR' is not set.")
if not os.path.isdir(source_dir):
    raise ValueError(f"Source directory '{source_dir}' does not exist or is not accessible.")

chunks_dir = os.getenv("CHUNKS_DIR")
if not chunks_dir:
    raise ValueError("Environment variable 'CHUNKS_DIR' is not set.")
if not os.path.isdir(chunks_dir):
    raise ValueError(f"Chunks directory '{chunks_dir}' does not exist or is not a directory.")
if not os.access(chunks_dir, os.W_OK):
    raise ValueError(f"Chunks directory '{chunks_dir}' is not writable.")
# Define the naming convention for chunks
chunk_file_naming = "data_{chunk_index}.bin"

# The partitioner parameters as imported from the job body in "params" -> "partitioner".
try:
    partitioner_params = os.getenv("PARTITIONER_PARAMS")
    if not partitioner_params:
        raise ValueError("Environment variable 'PARTITIONER_PARAMS' is not set.")
    params = json.loads(partitioner_params)
except json.JSONDecodeError:
    raise ValueError("Environment variable 'PARTITIONER_PARAMS' contains invalid JSON.")

# === Utility Functions ===

def list_source_files():
    """Lists all files in the source directory."""
    try:
        return [
            f for f in os.listdir(source_dir)
            if os.path.isfile(os.path.join(source_dir, f))
        ]
    except OSError as e:
        raise RuntimeError(f"Error accessing source directory '{source_dir}': {e}")

def write_chunk(data):
    """Writes a chunk of data to the next available file based on the naming convention."""
    # Use a regex pattern derived from the chunk_file_naming variable
    chunk_pattern = re.compile(re.escape(chunk_file_naming).replace(r"\{chunk_index\}", r"(\d+)"))
    
    # Determine the next chunk index
    existing_files = (
        f for f in os.listdir(chunks_dir)
        if os.path.isfile(os.path.join(chunks_dir, f)) and chunk_pattern.match(f)
    )
    chunk_indices = []
    for f in existing_files:
        match = chunk_pattern.match(f)
        if match:
            chunk_indices.append(int(match.group(1)))
    try:
        chunk_indices = sorted(chunk_indices)
        next_chunk_index = chunk_indices[-1] + 1 if chunk_indices else 0
        output_path = os.path.join(chunks_dir, chunk_file_naming.format(chunk_index=next_chunk_index))
        with open(output_path, "wb") as outfile:
            outfile.write(data)
        print(f"Chunk {next_chunk_index} written to {output_path}")
    except (IOError, OSError) as e:
        raise RuntimeError(f"Failed to write chunk {next_chunk_index} to {output_path}: {e}")


# === Main Logic ===

if __name__ == "__main__":
    print("Partitioner task started")

    source_files = list_source_files()
    nfiles = len(source_files)
    num_replicas = params["num_replicas"]

    for file in source_files:
        source_path = os.path.join(source_dir, file)

        # Copy the source file to the output path num_replicas times.
        with open(source_path, "rb") as infile:
            for i in range(num_replicas):
                # Reset the file pointer to the beginning of the file
                infile.seek(0)
                # Write the chunk to the output directory
                write_chunk(infile.read())