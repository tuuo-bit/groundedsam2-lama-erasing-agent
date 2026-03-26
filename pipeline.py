# eraser_agent/pipeline.py

import os
import sys
import traceback
import base64
from pathlib import Path
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_PROJECT_PATH = os.path.join(CURRENT_DIR, "lama")
GROUNDED_SAM_PROJECT_PATH = os.path.join(CURRENT_DIR, "Grounded-SAM-2")
GROUNDED_SAM_CONFIG_PATH = os.path.join(CURRENT_DIR, "Grounded-SAM-2/sam2")

base_dir = Path(__file__).parent
output_dir = base_dir / "testing"
output_dir.mkdir(parents=True, exist_ok=True)
input_path = base_dir / "testing"
input_path.mkdir(parents=True, exist_ok=True)
input_path = input_path / "car.png"   

sys.path.insert(0, LAMA_PROJECT_PATH)
sys.path.insert(0, GROUNDED_SAM_PROJECT_PATH)
sys.path.insert(0, GROUNDED_SAM_CONFIG_PATH)


# --- MUST BE SET BEFORE IMPORTING TORCH/CV2 ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import yaml
import numpy as np
import torch
import uvicorn
from omegaconf import OmegaConf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

# --- LaMa Imports (will work because of path setup) ---
from saicinpainting.training.trainers import load_checkpoint

# --- Grounded-SAM-2 Imports (will work because of path setup) ---
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model as dino_load_model, predict as dino_predict
from grounding_dino.groundingdino.datasets import transforms as T

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LaMa Config
LAMA_MODEL_PATH = os.path.join(LAMA_PROJECT_PATH, "big-lama")

# Grounded-SAM-2 Configs
SAM2_CHECKPOINT = os.path.join(GROUNDED_SAM_PROJECT_PATH, "checkpoints/sam2.1_hiera_large.pt")
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = os.path.join(GROUNDED_SAM_PROJECT_PATH, "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py")
GROUNDING_DINO_CHECKPOINT = os.path.join(GROUNDED_SAM_PROJECT_PATH, "gdino_checkpoints/groundingdino_swinb_cogcoor.pth")

# Hyperparameters
BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.15

# ---------------------------------------------------------
# FastAPI App and Global Models
# ---------------------------------------------------------
app = FastAPI(title="Eraser Agent: GroundedSAM2+LaMa Pipeline")

# Global variables for all models
lama_model = None
sam2_predictor = None
grounding_model = None

class MasterRequest(BaseModel):
    image_path: str = input_path
    text_prompt: str = "word. alphabet. letter. digit. number. watermark. logo. text."
    output_dir: str = output_dir
    save_mask: bool = True

# ---------------------------------------------------------
# Startup Event: Load ALL Models
# ---------------------------------------------------------
@app.on_event("startup")
def load_all_models():
    global lama_model, sam2_predictor, grounding_model
    print(f"[*] Loading all models onto device: {DEVICE}")

    # --- 1. Load LaMa Model ---
    print(f"[*] Loading LaMa model from {LAMA_MODEL_PATH}...")
    if not os.path.exists(LAMA_MODEL_PATH):
        print(f"[!] Error: LaMa model path not found: {LAMA_MODEL_PATH}")
        sys.exit(1)
    try:
        train_config_path = os.path.join(LAMA_MODEL_PATH, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        checkpoint_path = os.path.join(LAMA_MODEL_PATH, 'models', 'best.ckpt')
        
        lama_model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=DEVICE)
        lama_model.freeze()
        lama_model.to(DEVICE)
        lama_model.eval()
        print("[*] LaMa model loaded successfully.")
    except Exception as e:
        print(f"[!] Failed to load LaMa model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 2. Load Grounding DINO Model ---
    print(f"[*] Loading Grounding DINO model...")
    try:
        grounding_model = dino_load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE
        )
        print("[*] Grounding DINO model loaded successfully.")
    except Exception as e:
        print(f"[!] Failed to load Grounding DINO model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 3. Load SAM2 Model ---
    print(f"[*] Loading SAM2 model...")
    try:
        sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        print("[*] SAM2 model loaded successfully.")
    except Exception as e:
        print(f"[!] Failed to load SAM2 model: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    print("\n[*] All models loaded and server is ready!")


# ---------------------------------------------------------
# Helper: Pad image to modulo 8 (Required by LaMa)
# ---------------------------------------------------------
def pad_img_to_modulo(img, modulo=8):
    orig_height, orig_width = img.shape[:2]
    pad_h = (modulo - orig_height % modulo) % modulo
    pad_w = (modulo - orig_width % modulo) % modulo
    
    if len(img.shape) == 3:
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
        
    return img_padded, orig_height, orig_width

# ---------------------------------------------------------
# Main Pipeline Endpoint
# ---------------------------------------------------------
@app.post("/erase")
async def erase_object(request: MasterRequest):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}"
    try:
        text_prompt = request.text_prompt.lower().strip()
        if not text_prompt.endswith("."):
            text_prompt += "."

        print(f"[*] Starting Erase for prompt: '{request.text_prompt}'")

        image = cv2.imread( request.image_path)
        image_rgb = cv2.cvtColor( image, cv2.COLOR_BGR2RGB)

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image = np.array( image_rgb)
        image_tensor, _ = transform(Image.fromarray(image_rgb), None)

        boxes, confidences, labels = dino_predict(
            model=grounding_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )

        h, w, _ = image.shape

        if boxes.shape[0] == 0:

            return JSONResponse(content={
            "status": "warning",
            "message": f"Failed to find '{text_prompt}' in the image. Nothing erased."
            })

        else:
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            sam2_predictor.set_image( image_rgb)
            
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

            if masks.ndim == 4:
                masks = np.squeeze(masks, 1)
            
            masks = np.any(masks, 0)
            raw_mask_img = ( masks * 255).astype(np.uint8)

            kernel_size = int(np.sqrt(h**2 + w**2) // 100)
            kernel_size += 1-kernel_size % 2
            
            final_mask_img = cv2.dilate( raw_mask_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)), iterations=1)

            final_mask_img = cv2.morphologyEx( final_mask_img, cv2.MORPH_CLOSE, np.ones(( kernel_size//2, kernel_size//2), np.uint8))

            final_mask_img = cv2.GaussianBlur( final_mask_img, (kernel_size, kernel_size), 0, 0)
        
        if request.save_mask:
            os.makedirs( os.path.abspath( request.output_dir), exist_ok=True)
            cv2.imwrite( os.path.join( request.output_dir, filename + "_mask.png"), final_mask_img)
        
        print("[*] Starting LaMA Inpainting Process!")

        image_for_lama = image_rgb.copy()

        if image_for_lama.shape[:2] != final_mask_img.shape[:2]:
            final_mask_img = cv2.resize(final_mask_img, (image_for_lama.shape[1], image_for_lama.shape[0]), interpolation=cv2.INTER_NEAREST)

        image_padded, orig_h, orig_w = pad_img_to_modulo(image_for_lama, modulo=8)
        mask_padded, _, _ = pad_img_to_modulo(final_mask_img, modulo=8)

        image_tensor = image_padded.astype('float32') / 255.0
        mask_tensor = mask_padded.astype('float32') / 255.0
        mask_tensor = np.expand_dims(mask_tensor, -1)

        image_tensor = np.transpose(image_tensor, (2, 0, 1))
        mask_tensor = np.transpose(mask_tensor, (2, 0, 1))

        batch = {
            'image': torch.from_numpy(image_tensor).unsqueeze(0).to(DEVICE),
            'mask': torch.from_numpy(mask_tensor).unsqueeze(0).to(DEVICE)
        }
        with torch.no_grad():
            batch['mask'] = (batch['mask'] > 0) * 1.0
            batch = lama_model(batch)
            inpainted_result = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

        inpainted_result = inpainted_result[:orig_h, :orig_w] # Crop back to original size
        inpainted_result = np.clip(inpainted_result * 255, 0, 255).astype('uint8')
        inpainted_result = cv2.cvtColor(inpainted_result, cv2.COLOR_RGB2BGR)

        os.makedirs( os.path.abspath( request.output_dir), exist_ok=True)
        cv2.imwrite( os.path.join( request.output_dir, filename + "_inpaint.png"), inpainted_result)
        
        print("[*] Erase completed successfully!")
        
        return JSONResponse(content={
            "status": "success",
            "message": "Inpainting completed.",
            "num_objects_found": int(boxes.shape[0]),
            "output_path": os.path.join( request.output_dir, filename + ".png")
        })

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))