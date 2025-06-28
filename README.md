# How to Build a Scalable Image Generation Pipeline with ByteNite

---

## Table of Contents

- [Introduction](#introduction)
- [What is ByteNite?](#what-is-bytenite)
- [Project Structure](#project-structure)
- [CPU vs. GPU Versions](#cpu-vs-gpu-versions)
- [Prerequisites](#prerequisites) _(If you've already onboarded to ByteNite, you can skip this section!)_
- [Onboarding to ByteNite](#onboarding-to-bytenite)
- [Development Environment Setup](#development-environment-setup)
- [How to Use the CLI/SDK or API](#how-to-use-the-clisdk-or-api)
- [Installing the ByteNite Developer CLI](#installing-the-bytenite-developer-cli)
- [App Components](#app-components)
  - [App (img-gen-diffusers)](#app-img-gen-diffusers)
  - [Partitioner (fanout-replica)](#partitioner-fanout-replica)
  - [Templates](#templates)
- [Running Jobs on ByteNite](#running-jobs-on-bytenite)
  - [Launching via CLI](#launching-via-cli)
  - [Launching via API](#launching-via-api)
- [Example: Image Generation with Stable Diffusion](#example-image-generation-with-stable-diffusion)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [References](#references)

---

## Introduction

This repository provides a robust, scalable image generation pipeline designed to run on ByteNite’s distributed, serverless container platform. It supports both CPU and GPU execution, enabling parallelized image generation at scale with minimal infrastructure configuration. For CPU-based workloads, the pipeline uses the Hugging Face diffusers library to perform Stable Diffusion inference, while GPU-based executions leverage the high-performance Flux model for faster generation.


## What is ByteNite?

[ByteNite](https://docs.bytenite.com/) is a serverless container platform for stateless, compute-intensive workloads. It abstracts away cloud infrastructure, letting you focus on your application logic. ByteNite provides:

- Near-instant startup times and flexible compute
- Distributed execution fabric (native fan-in/fan-out logic)
- Modular building blocks: Partitioners, Apps, Assemblers
- Simple job submission via CLI or API


## Project Structure

```
img-gen-diffusers/
├── fanout-replica/           # Partitioner engine
│   ├── manifest.json
│   └── app/
│       └── main.py
├── img-gen-diffusers-flux-gpu/   # GPU-optimized app
│   ├── manifest.json
│   ├── buildx-flux-schnell-w-token.sh
│   ├── Dockerfile
│   └── app/
│       └── main.py
├── img-gen-diffusers-notaai-cpu/ # CPU-optimized app
│   ├── manifest.json
│   └── app/
│       └── main.py
├── zipper/                  # Assembler engine
│   ├── manifest.json
│   └── app/
│       └── main.py
├── templates/               # Job templates
│   ├── img-gen-diffusers-flux-gpu-template.json
│   └── img-gen-diffusers-notaai-cpu-template.json
└── README.md
```


## CPU vs. GPU Versions

- **CPU Version (`img-gen-diffusers-notaai-cpu`)**: Runs on high-core-count CPUs (recommended: 16+ vCPUs, 32GB+ RAM). Suitable for environments without GPU access or for cost-effective batch jobs.
- **GPU Version (`img-gen-diffusers-flux-gpu`)**: Runs on general-purpose and high-end GPUs. Optimized for CUDA-enabled devices, with support for advanced GPU features and larger batch sizes. Recommended for faster inference and large-scale jobs.

Both versions use the same core logic and can be deployed interchangeably depending on your hardware requirements.


## Prerequisites

**Already onboarded to ByteNite?**  
If you’ve already created an account, set up payment, and installed the CLI for a previous app, you can skip this section and jump straight to [Onboarding to ByteNite](#onboarding-to-bytenite) or the next relevant step.

---

## Onboarding to ByteNite

👤 **Create an account**

- You will need to [Request an Access Code](https://www.bytenite.com/get-access) and fill out the resulting form with your contact info.
- Once receiving your access code, you will be able to sign up on the [computing platform](https://app.bytenite.com).

💳 **Add a payment method**

- Once logged into the platform, go to the [Billing Page](https://app.bytenite.com/billing) (can also access by clicking into the Billing tab in the sidebar).
- Locate the Payment Info card and navigate to the Customer Portal. Add a payment method to your account through Stripe.
- Your payment info is used for manual and automatic top-ups. Ensure you have enough funds to avoid service interruptions.

🪙 **Redeem ByteChips**

- If you have a coupon code, redeem it on your billing page to add ByteChips (credits) to your balance.
- Go to the Account Balance card and click "Redeem". Enter your coupon code and complete the process. Refresh to confirm the balance.
- We'd love to get you started with free credits to test our platform, [contact ByteNite support](https://bytenite.com/info) to request some.

---

## Development Environment Setup

🛠️ **Set up development tools**

- Python 3.8+ for local development or running scripts.
- Git (to clone this repository)
- (Optional) Docker if you plan to build custom containers.

---

## How to Use the CLI/SDK or API

You can use **either** the CLI/SDK **or** the API to interact with ByteNite. Most users should use the CLI/SDK for the easiest experience. Advanced users can use the API for programmatic access. You do not need to do both.

### For CLI/SDK Users (Recommended)

- Download and install the ByteNite Developer CLI (see below for instructions by OS).
- Authenticate by running:
    ```sh
    bytenite auth
    ```
  This will open a browser window for secure login.  
- Once authenticated, you can use all `bytenite` CLI commands to manage apps, engines, templates, and jobs.

### For API Users (Advanced/Programmatic Access)

If you plan to use the ByteNite API directly (e.g., with Postman or custom scripts), you’ll need an API key and access token:

🔐 **Get an API key**

- Go to [your ByteNite profile](https://app.bytenite.com/profile) or click your profile avatar (top right).
- Click **New API Key**, configure its settings, and enter the confirmation code sent to your email.
- Copy your API key immediately and store it securely. You will not be able to view it again.
- If a key is no longer needed or is compromised, revoke it from your profile.

🔑 **Get an access token**

- An access token is required to authenticate all requests to the ByteNite API (including Postman).
- Request an access token from the Access Token endpoint using your API key. Tokens last 1 hour by default.
- See the [API Reference](https://docs.bytenite.com/api) for details and example requests.

## Installing the ByteNite Developer CLI

### Linux (Ubuntu/Debian)

Add the ByteNite repository:

```sh
echo "deb [trusted=yes] https://storage.googleapis.com/bytenite-prod-apt-repo/debs ./" | sudo tee /etc/apt/sources.list.d/bytenite.list
```

Update package lists:

```sh
sudo apt update
```

Install the ByteNite CLI:

```sh
sudo apt install bytenite
```

Troubleshooting:

- Update your system: `sudo apt update && sudo apt upgrade`
- Verify repository: `cat /etc/apt/sources.list.d/bytenite.list`
- Check package: `apt search bytenite`


### macOS

Add the ByteNite tap:

```sh
brew tap ByteNite2/bytenite-dev-cli https://github.com/ByteNite2/bytenite-dev-cli.git
```

Install the CLI:

```sh
brew install bytenite
```

Update Homebrew:

```sh
brew update
```

Upgrade ByteNite CLI:

```sh
brew upgrade bytenite
```


### Windows

Download and run the latest Windows release from the ByteNite CLI GitHub page.


### Verify Installation

Check the CLI version:

```sh
bytenite version
```


### Authenticate

Authenticate with OAuth2:

```sh
bytenite auth
```

This opens a browser for login. Credentials are stored at:

- Linux: `$HOME/.config/bytenite-cli/auth-prod.json`
- Mac: `/Users/[user]/Library/Application Support/bytenite-cli/auth-prod.json`

---

## Quick Start

Follow these steps to get up and running with your own ByteNite image generation pipeline:

1. **Clone this repository to your own machine:**

   ```sh
   git clone <your-fork-or-this-repo-url> && \
   cd img-gen-diffusers 
   ```
2. **(Optional but recommended) Fork this repo to your own GitHub account.**
3. **Install the ByteNite Developer CLI** (see instructions above for your OS).
4. **Authenticate with ByteNite:**

   ```sh
   bytenite auth
   ```
5. **Push the apps and engines to your ByteNite account:**

   ```sh
   bytenite app push ./img-gen-diffusers-notaai-cpu && \
   bytenite app push ./img-gen-diffusers-flux-gpu && \
   bytenite engine push ./fanout-replica && \
   bytenite engine push ./zipper
   ```
6. **Activate the apps and engines:**

   ```sh
   bytenite app activate img-gen-diffusers-notaai-cpu && \
   bytenite app activate img-gen-diffusers-flux-gpu && \
   bytenite engine activate fanout-replica && \
   bytenite engine activate zipper
   ```
7. **Push the job templates:**

   ```sh
   bytenite template push ./templates/img-gen-diffusers-notaai-cpu-template.json && \
   bytenite template push ./templates/img-gen-diffusers-flux-gpu-template.json
   ```
8. **Launch a job using the API (we have a handy [Postman collection]() ready for you).** ([**GPU** Job Postman collection](https://www.postman.com/bytenite-team/workspace/bytenite-api-demos/collection/42601786-b99aa52d-c026-47a6-a5b6-57d2712dfeb7?action=share&source=copy-link&creator=42601786) or [**CPU** Job Postman collection](https://www.postman.com/bytenite-team/workspace/bytenite-api-demos/collection/36285584-6fe2a402-4559-4165-8b1f-cf11affa119a?action=share&source=copy-link&creator=42601786)) or proceed with the methods described below.

> **Note:** If using Postman, you’ll need to generate an access token with your API key and include it in the `Authorization` header for all requests.
---

## ByteNite Dev CLI: Commands & Usage

Run the help command to see all options:

```sh
bytenite --help
```

**Most users only need these commands in order:**

1. `bytenite app push [app_folder]`
2. `bytenite app activate [app_tag]`
3. `bytenite engine push [engine_folder]`
4. `bytenite engine activate [engine_tag]`
5. `bytenite template push [template_filepath]`
6. `bytenite app status [app_tag]` (to check status)

For more commands, run `bytenite --help` or see the ByteNite documentation.

## App Components

### App (`img-gen-diffusers`)

- Implements image generation using Hugging Face diffusers (Stable Diffusion).
- Accepts a prompt and outputs an image file.
- See `main.py` for implementation details.


### Partitioner (`fanout-replica`)

- Splits the job into multiple parallel tasks (replicas).
- Each replica generates an image from the same or different prompts.
- See `fanout-replica/app/main.py` for logic.


### Templates

- Templates link your app, partitioner, and (optionally) assembler.
- Example (CPU): `templates/img-gen-diffusers-notaai-cpu-template.json`
- Example (GPU): `templates/img-gen-diffusers-flux-gpu-template.json`

## Running Jobs on ByteNite

### Launching via API

You can use our ready-made Postman collections to streamline your job launching ([**GPU** Job Postman collection](https://www.postman.com/bytenite-team/workspace/bytenite-api-demos/collection/42601786-b99aa52d-c026-47a6-a5b6-57d2712dfeb7?action=share&source=copy-link&creator=42601786) or [**CPU** Job Postman collection](https://www.postman.com/bytenite-team/workspace/bytenite-api-demos/collection/36285584-6fe2a402-4559-4165-8b1f-cf11affa119a?action=share&source=copy-link&creator=42601786)) or proceed with the methods described below.

1. **Get an access token:**

    ```python
    import requests
    
    response = requests.post(
        "https://api.bytenite.com/v1/auth/access_token",
        json={"apiKey": "<YOUR_API_KEY>"}
    )
    token = response.json()["token"]
    ```
2. **Create a job:**

    ```python
    response = requests.post(
        "https://api.bytenite.com/v1/customer/jobs",
        headers={"Authorization": token},
        json={"name": "img-gen-job", "templateId": "img-gen-diffusers-flux-gpu-template"}
    )
    jobId = response.json()["job"]["id"]
    ```
3. **Set data source/destination:**

    ```python
    response = requests.patch(
        f"https://api.bytenite.com/v1/customer/jobs/{jobId}/datasource",
        headers={"Authorization": token},
        json={
            "dataSource": {"dataSourceDescriptor": "bypass"},
            "dataDestination": {"dataSourceDescriptor": "bucket"}
        }
    )
    ```
4. **Set parameters:**

    ```python
    response = requests.patch(
        f"https://api.bytenite.com/v1/customer/jobs/{jobId}/params",
        headers={"Authorization": token},
        json={
            "partitioner": {"num_replicas": 5},
            "app": {"prompt": "A photorealistic sunset over the ocean"}
        }
    )
    ```
5. **Run the job:**

    ```python
    response = requests.post(
        f"https://api.bytenite.com/v1/customer/jobs/{jobId}/run",
        headers={"Authorization": token},
        json={"taskTimeout": 3600, "jobTimeout": 86400, "isTestJob": True}
    )
    ```

## Example: Image Generation with Stable Diffusion

- Launch a job with the prompt of your choice and desired number of replicas.
- Each task will generate an image and save it to a temporary output bucket for your retrieval.
- Once your job has completed, you will recieve a downloadable link to your zipped file of outputs
- Results and logs can be accessed via the ByteNite UI or API.

## Troubleshooting & FAQ

- **App fails to start:** Check your container image and manifest.json for correct dependencies and entrypoint.
- **No output images:** Ensure the output path in `main.py` matches ByteNite’s expected results directory.
- **Resource errors:** Increase `min_cpu`/`min_memory` or use the GPU version for heavy workloads.
- **Authentication issues:** Regenerate your API key and access token.
- **See [ByteNite Docs FAQ](https://docs.bytenite.com/faq) for more.**

## References

- [ByteNite Documentation](https://docs.bytenite.com/)
- [ByteNite Dev CLI](https://docs.bytenite.com/sdk/dev-cli)
- [API Reference](https://docs.bytenite.com/api)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index)
- [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5)

---

For questions or support, please open an issue or contact the ByteNite team via the [official docs](https://docs.bytenite.com/).