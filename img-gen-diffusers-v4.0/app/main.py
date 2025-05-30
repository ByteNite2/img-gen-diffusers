import os
import json
import torch
import time
from diffusers import DiffusionPipeline

# Load prompt from environment
task_results_dir = os.getenv('TASK_RESULTS_DIR', '/results')
app_params = json.loads(os.getenv('APP_PARAMS', '{"prompt": "a steampunk airship flying through clouds"}'))

def load_pipeline():
    print("Loading Flux Schnell pipeline from local path...")
    dtype = torch.float16  # Recommended for modern GPUs like RTX 4090

    # Load pipeline from local folder — no custom_pipeline needed

    pipe = DiffusionPipeline.from_pretrained("/models/flux", torch_dtype=torch.float16,
    	custom_pipeline="StableDiffusionPipeline"  # explicitly set
	).to("cuda")


    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers enabled")
    except Exception as e:
        print(f"xformers error (non-blocking): {e}")

    return pipe

def generate_image(pipeline, prompt, output_path):
    print(f"Generating image for prompt: {prompt}")
    with torch.inference_mode():
        start = time.time()
        image = pipeline(prompt).images[0]
        end = time.time()
    image.save(output_path)
    print(f"Inference took {end - start:.2f} seconds. Image saved to {output_path}")

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA not available. Please run on a GPU-enabled machine.")
    
    pipeline = load_pipeline()
    prompt = app_params["prompt"]
    output_path = os.path.join(task_results_dir, "output_image.png")
    generate_image(pipeline, prompt, output_path)
