# === BYTENITE ASSEMBLER - MAIN SCRIPT ===

# Read the documentation --> https://docs.bytenite.com/create-with-bytenite/building-blocks/assembling-engines

# == Imports and Environment Variables ==

# Ensure all required external libraries are available in the Docker container image specified in manifest.json under "platform_config" > "container".
try:
    import json
    import os
    import logging
    import zipfile
except ImportError as e:
    raise ImportError(f"Required library is missing: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the folder where the task results from your app executions are stored.
task_results_dir = os.getenv("TASK_RESULTS_DIR")
if task_results_dir is None:
    raise ValueError("Environment variable 'TASK_RESULTS_DIR' is not set.")
if not os.path.isdir(task_results_dir):
    raise ValueError(f"Task result directory '{task_results_dir}' does not exist or is not accessible.")

# Path to the final output directory where your assembler results will be saved. The files in these folder will be uploaded to your data destination.
output_dir = os.getenv("OUTPUT_DIR")
if not output_dir:
    raise ValueError("Environment variable 'OUTPUT_DIR' is not set.")
if not os.path.isdir(output_dir):
    raise ValueError(f"Output directory '{output_dir}' does not exist or is not a directory.")
if not os.access(output_dir, os.W_OK):
    raise ValueError(f"Output directory '{output_dir}' is not writable.")

# The partitioner parameters as imported from the job body in "params" -> "partitioner".
try:
    assembler_params = os.getenv("ASSEMBLER_PARAMS")
    if not assembler_params:
        raise ValueError("Environment variable 'ASSEMBLER_PARAMS' is not set.")
    params = json.loads(assembler_params)
except json.JSONDecodeError:
    raise ValueError("Environment variable 'ASSEMBLER_PARAMS' contains invalid JSON.")


# === Utility Functions ===

def list_result_files():
    """Lists and returns the filenames of all files in the results directory."""
    try:
        return [
            filename for filename in os.listdir(task_results_dir)
            if os.path.isfile(os.path.join(task_results_dir, filename))
        ]
    except OSError as e:
        raise RuntimeError(f"Error accessing source directory '{task_results_dir}': {e}")
    except Exception as e:
        raise RuntimeError(f"Error listing files in '{task_results_dir}': {e}")


def zip_result_files(archive_name="results_archive.zip"):
    """Zips all files in the results directory into a single archive."""
    try:
        # Read all result files
        result_files = list_result_files()
        if not result_files:
            logger.warning("No files found in the results directory to zip.")
            return

        # Create the output path for the zip archive
        archive_path = os.path.join(output_dir, archive_name)
        
        # Create and write the zip archive
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in result_files:
                file_path = os.path.join(task_results_dir, file)
                zipf.write(file_path, arcname=file)  # Add file to archive with relative path
        logger.info(f"Zipped {len(result_files)} files into {archive_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create zip archive: {e}")

# === Main Logic ===

if __name__ == "__main__":
    logger.info("Assembler task started")
    zip_result_files()

    
