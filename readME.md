# Eraser Agent

Eraser Agent is a tool designed for automated object removal from images using Grounded SAM 2 for masking and LaMa for inpainting.

## Project Structure

```text
eraser_agent
├── download_singleDinoCkpt.sh   # Script to download DINO checkpoint
├── download_singleSAMCkpt.sh    # Script to download SAM checkpoint
├── env_groundedsam2.yaml       # Environment config for Grounded SAM 2
├── env_lama.yaml               # Environment config for LaMa Inpainting
├── env_main.yaml               # Environment config for main server
├── inpaint_server.py           # Server handling inpainting tasks
├── main.py                     # Main entry point for the agent
├── mask_server.py              # Server handling detection and segmentation
└── setup.sh                    # Automated setup and installation script
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
* **Environment Creation:** Automatically creates three specialized Conda environments:
    * `main`
    * `groundedsam2`
    * `lama`
* **Dependency Installation:** Installs all required Python packages and libraries for each environment.
* **Repository Setup:** Clones the external repositories for `Grounded-SAM-2` and `LaMa`.
* **Model Management:** Moves the checkpoint scripts and server files to their respective directories. Downloads the necessary AI models.

## Usage

Once the setup is complete, you can boot up individual servers in separated terminals.

##### Note: Please run mask and inpaint server on the given port numbers only. Else, make sure to make changes to SAM2_URL and LAMA_URL in main.py accordingly.
```bash
conda activate groundedsam2
cd Grounded-SAM-2
uvircorn mask_server:app --host 0.0.0.0 --port 8000 --reload
```
```bash
conda activate lama
cd lama
uvicorn inpaint_server:app --host 0.0.0.0 --port 8001 --reload
```

*Any port number can assigned to main server*

```bash
conda activate main
uvicorn main:app --host 0.0.0.0 --port 8188 --reload
```

## Components

* **Masking:** Utilizes Grounded SAM 2 to identify and create masks for objects based on text prompts.
* **Inpainting:** Utilizes the LaMa model to remove the masked areas.
* **Servers:** The project operates using a server-client architecture where `mask_server.py` and `inpaint_server.py` handle the heavy processing tasks.

## Further Improvements

* Better detections > Better masks > Find desired targets
* Explore inpainting models for quality