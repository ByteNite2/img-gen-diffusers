# === BYTENITE APP - MAIN SCRIPT ===
import json
import os
import torch
from diffusers import FluxPipeline
import time
import gc

# Environment variables from your service
task_dir = os.getenv('TASK_DIR')
task_results_dir = os.getenv('TASK_RESULTS_DIR')
app_params = json.loads(os.getenv('APP_PARAMS'))
chunk_number = int(os.getenv('CHUNK_NUMBER', '0'))

# Model cache directory - adapt to your container structure
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/app/models')

def flush():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def load_flux_model():
    """Load FLUX.1-schnell optimized for RTX 4090"""
    print("Loading FLUX.1-schnell model optimized for RTX 4090...")
    print(f"Task directory: {task_dir}")
    print(f"Results directory: {task_results_dir}")
    print(f"Model cache directory: {MODEL_CACHE_DIR}")
    print(f"Chunk number: {chunk_number}")
    
    # Clear any existing GPU memory
    flush()
    
    # Check if model exists in cache
    model_path = os.path.join(MODEL_CACHE_DIR, "flux-schnell")
    
    if os.path.exists(model_path):
        print(f"Loading model from cache: {model_path}")
        # Use bfloat16 for better performance on RTX 4090
        pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
    else:
        print("Model not found in cache, downloading...")
        hf_token = os.getenv('HF_TOKEN')
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            token=hf_token
        )
        # Save to cache for future use
        print(f"Saving model to cache: {model_path}")
        pipe.save_pretrained(model_path)
    
    print("Enabling RTX 4090 optimizations...")
    
    # Enable VAE optimizations (critical for RTX 4090)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print("VAE tiling and slicing enabled")
    
    # Enable sequential CPU offload (works better than model_cpu_offload for FLUX)
    pipe.enable_sequential_cpu_offload()
    print("Sequential CPU offload enabled")
    
    return pipe

def generate_image(prompt, output_path):
    """Generate image optimized for RTX 4090"""
    print(f"Generating image for prompt: {prompt}")
    print(f"Output path: {output_path}")
    
    # Clear GPU cache before loading
    flush()
    
    # Load the model
    pipe = load_flux_model()
    
    # Generate image
    start_time = time.time()
    
    try:
        # RTX 4090 optimized parameters - based on working examples
        print("Starting image generation...")
        image = pipe(
            prompt,
            height=512,          # RTX 4090 limit for FLUX
            width=512,           # RTX 4090 limit for FLUX  
            guidance_scale=0.0,  # FLUX.1-schnell optimal setting
            num_inference_steps=4,  # FLUX.1-schnell optimal setting
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        
        generation_time = time.time() - start_time
        print(f"Image generated in {generation_time:.2f} seconds")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"GPU out of memory even with optimizations: {e}")
        print("Try using a quantized model version (Q8 or FP8) for your RTX 4090")
        raise
    
    # Clear GPU cache after generation
    flush()
    
    # Ensure results directory exists (your service creates it, but double-check)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the image
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    
    # Print memory usage for debugging
    if torch.cuda.is_available():
        max_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f'Max GPU memory used: {max_memory_gb:.2f} GB')
    
    return output_path

if __name__ == '__main__':
    print("=== BYTENITE FLUX IMAGE GENERATION TASK ===")
    print("Python task started")
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU only'}")
    print(f"Working directory: {os.getcwd()}")
    
    # Validate environment variables
    if not task_dir:
        raise ValueError("TASK_DIR environment variable not set")
    if not task_results_dir:
        raise ValueError("TASK_RESULTS_DIR environment variable not set")
    if not app_params:
        raise ValueError("APP_PARAMS environment variable not set or invalid JSON")
    
    # Get prompt from app parameters
    prompt = app_params.get("prompt")
    if not prompt:
        raise ValueError("No 'prompt' found in APP_PARAMS")
    
    print(f"Prompt: {prompt}")
    
    # Generate output filename - include chunk number if needed
    if chunk_number > 0:
        output_filename = f"output_image_chunk_{chunk_number}.png"
    else:
        output_filename = "output_image.png"
    
    output_path = os.path.join(task_results_dir, output_filename)
    
    try:
        generated_path = generate_image(prompt, output_path)
        print("=== TASK COMPLETED SUCCESSFULLY ===")
        print(f"Generated image: {generated_path}")
        
        # Print file info for verification
        if os.path.exists(generated_path):
            file_size = os.path.getsize(generated_path)
            print(f"File size: {file_size:,} bytes")
        
    except Exception as e:
        print("=== TASK FAILED ===")
        print(f"Python exception: {e}")
        raise e