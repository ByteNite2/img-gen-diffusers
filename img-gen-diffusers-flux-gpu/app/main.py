# === BYTENITE APP - MAIN SCRIPT ===
import json
import os
import torch
from diffusers import FluxPipeline
import time
import random

# Environment variables
task_dir = os.getenv('TASK_DIR')
task_results_dir = os.getenv('TASK_RESULTS_DIR')
app_params = json.loads(os.getenv('APP_PARAMS'))

# Model cache directory
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/app/models')

def load_flux_model():
    """Load the FLUX.1-schnell model from cache directory"""
    print("Loading FLUX.1-schnell model...")
    
    # Determine dtype based on device availability
    if torch.cuda.is_available():
        dtype = torch.float16
        device = "cuda"
        print("CUDA available - using float16")
    else:
        dtype = torch.float32
        device = "cpu"
        print("CUDA not available - using float32 on CPU")
    
    # Check if model exists in cache
    model_path = os.path.join(MODEL_CACHE_DIR, "flux-schnell")
    
    if os.path.exists(model_path):
        print(f"Loading model from cache: {model_path}")
        pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True
        )
    else:
        print("Model not found in cache, downloading...")
        hf_token = os.getenv('HF_TOKEN')
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
            token=hf_token
        )
        # Save to cache for future use
        print(f"Saving model to cache: {model_path}")
        pipe.save_pretrained(model_path)
    
    # Move to appropriate device
    pipe = pipe.to(device)
    print(f"Model loaded on {device.upper()}")
    
    return pipe

def generate_image(prompt, output_path):
    """Generate image using FLUX.1-schnell model"""
    print(f"Generating image for prompt: {prompt}")
    
    # Load the model
    pipe = load_flux_model()
    
    # Generate image
    start_time = time.time()

    # FLUX.1-schnell specific parameters
    random_seed = random.randint(0, 2**32 - 1)
    generator_device = "cpu" if not torch.cuda.is_available() else "cuda"
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=0.0,  # FLUX.1-schnell works best with guidance_scale=0
        num_inference_steps=4,  # FLUX.1-schnell is optimized for few steps
        max_sequence_length=256,
        generator=torch.Generator(generator_device).manual_seed(random_seed),  # Fixed seed for reproducibility
    ).images[0]
    
    print("debug - test random seed 2")

    generation_time = time.time() - start_time
    print(f"Image generated in {generation_time:.2f} seconds")
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Image saved to: {output_path}")

if __name__ == '__main__':
    print("Python task started")
    
    # Get prompt from app parameters
    prompt = app_params["prompt"]
    output_path = os.path.join(task_results_dir, "output_image.png")
    
    try:
        generate_image(prompt, output_path)
        print("Task completed successfully")
    except Exception as e:
        print("Python exception: ", e)
        raise e