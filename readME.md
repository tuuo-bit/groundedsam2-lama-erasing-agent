# Eraser Agent

Eraser Agent is a tool designed for automated object removal from images using Grounded SAM 2 for masking and LaMa for inpainting.

## Project Structure

```text
eraser_agent
├── download_singleDinoCkpt.sh   # Script to download DINO checkpoint
├── download_singleSAMCkpt.sh    # Script to download SAM checkpoint
├── pipeline.py                  # Single server for removal pipeline
└── setup.sh                     # Automated setup and installation script
```

## Installation

The project includes a comprehensive `setup.sh` script that automates the entire installation process.

### Prerequisites
* A Linux-based system (Ubuntu recommended).
* Basic terminal access.

### Setup Steps
1. Give execution permissions to the setup script:
   ```bash
   chmod +x setup.sh
   ```
2. Run the setup script:
   ```bash
   ./setup.sh
   ```

### What the Setup Script Does:
* **System Check:** Verifies if `Miniconda` and `git` are installed (installs them if missing).
* **Environment Creation:** Automatically create a conda environment `eraser_env`
* **Dependency Installation:** Installs all required Python packages and libraries for each environment.
* **Repository Setup:** Clones the external repositories for `Grounded-SAM-2` and `LaMa`.
* **Model Management:** Moves the checkpoint scripts to their respective directories and downloads the necessary AI models.

## Usage

Once the setup is complete, you can boot up the server in a terminal.

```bash
conda activate eraser_env
uvicorn pipeline:app --host 0.0.0.0 --port 8188 --reload
```

## Components

* **Masking:** Utilizes Grounded SAM 2 to identify and create masks for objects based on text prompts.
* **Inpainting:** Utilizes the LaMa model to remove the masked areas.

## Further Improvements

* Better detections > Better masks > Find desired targets
* Explore inpainting models for quality