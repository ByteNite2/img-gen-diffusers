# === BYTENITE APP - MAIN SCRIPT ===
import json
import os
import torch
from diffusers import FluxPipeline  # Updated import
import time

task_dir = os.getenv('TASK_DIR')
task_results_dir = os.getenv('TASK_RESULTS_DIR')
app_params = json.loads(os.getenv('APP_PARAMS'))

def generate_image(prompt, output_path):
    print(f"Generating image for prompt: {prompt}")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    # Check for CUDA availability
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. Make sure you're running on a GPU-enabled environment.")

    # Use bfloat16 as recommended for FLUX
    dtype = torch.bfloat16

    # Load Flux pipeline on GPU
    pipeline = FluxPipeline.from_pretrained(
        "/models/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    start_time = time.time()
    with torch.inference_mode():
        print("Inference started")
        image = pipeline(
            prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            max_sequence_length=256,
            height=640,
            width=640,
            generator=torch.Generator("cuda").manual_seed(0)
        ).images[0]
    end_time = time.time()

    print(f"Inference completed in {end_time - start_time:.2f} seconds")
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