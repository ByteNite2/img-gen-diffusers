# === BYTENITE APP - MAIN SCRIPT ===
import json
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderTiny
import time
import psutil
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    ipex = None
    print("Intel Extension for PyTorch not installed. Running without IPEX optimization.")

task_dir = os.getenv('TASK_DIR')
task_results_dir = os.getenv('TASK_RESULTS_DIR')
app_params = json.loads(os.getenv('APP_PARAMS'))

def generate_image(prompt, output_path):
    # Extract the prompt from params
    print(f"Generating image for prompt: {prompt}")

    # Log the number of available CPU cores
    num_threads = os.cpu_count() or 1
    print(f"Available CPU cores: {num_threads}")

    # Log current CPU utilization before inference
    print(f"Initial CPU usage: {psutil.cpu_percent(interval=1)}%")

    # Determine the appropriate data type for CPU execution
    dtype = torch.bfloat16 if torch.has_mps else torch.float32

    # Load Stable Diffusion pipeline with distilled model
    pipeline = StableDiffusionPipeline.from_pretrained(
        "nota-ai/bk-sdm-small",
        torch_dtype=dtype
    )
    pipeline.to("cpu")

    # Replace default autoencoder with tiny autoencoder
    pipeline.vae = AutoencoderTiny.from_pretrained("sayakpaul/taesd-diffusers", torch_dtype=dtype).to("cpu")

    # Use DPMSolverMultistepScheduler for faster inference
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # Enable PyTorch CPU optimizations
    torch.set_float32_matmul_precision("high")

    # Ensure PyTorch uses all available CPU cores
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)

    # Apply IPEX optimizations if available
    if ipex is not None:
        pipeline.unet = ipex.optimize(pipeline.unet)
        pipeline.vae = ipex.optimize(pipeline.vae)
        print("Applied IPEX optimizations for Intel CPU.")

    # Log the actual number of threads PyTorch is using
    print(f"PyTorch intra-op threads: {torch.get_num_threads()}")
    print(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")

    # Measure inference time
    start_time = time.time()

    # Run inference in no_grad mode
    with torch.inference_mode():
        print("Inference started")
        image = pipeline(
            prompt,
            num_inference_steps=25,  # Reduced steps with DPM solver
            guidance_scale=7.5      # Adjusted for better prompt adherence
        ).images[0]

    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds")

    # Log CPU utilization after inference
    print(f"Post-inference CPU usage: {psutil.cpu_percent(interval=1)}%")

    # Save the output image
    image.save(output_path)
    
    print(f"Image saved to: {output_path}")

if __name__ == '__main__':
    print("Python task started")
    prompt = app_params["prompt"]
    output_path = os.path.join(task_results_dir, "output_image.png")
    try:
        generate_image(prompt, output_path)
    except Exception as e:
        print("Python exception: ", e)
        raise e