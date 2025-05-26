#!/usr/bin/env python3
"""
FastAPI-based inference API for the Local-Global Attention Network.

This API provides REST endpoints for twin verification tasks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
from typing import Dict, Any
import logging

from src.models.attention_network import SiameseLocalGlobalNet
from src.inference import TwinVerifier
from src.data.dataset import get_transform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Twin Verification API",
    description="API for identical twin verification using Local-Global Attention Network",
    version="1.0.0"
)

# Global variables for model and verifier
model = None
verifier = None
config = None
transform = None


def load_model(model_path: str, config_path: str):
    """Load the trained model and configuration."""
    global model, verifier, config, transform
    
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create model
        model = SiameseLocalGlobalNet(
            backbone=config['model']['backbone'],
            num_classes=config['model']['num_classes'],
            attention_type=config['model']['attention_type'],
            feature_dim=config['model']['feature_dim'],
            local_regions=config['model']['local_regions'],
            dropout_rate=config['model']['dropout_rate']
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Create verifier
        verifier = TwinVerifier(model, device=device)
        
        # Create transform
        transform = get_transform(config['data']['image_size'], is_training=False)
        
        logger.info(f"Model loaded successfully from {model_path}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_path = os.getenv('MODEL_PATH', 'checkpoints/best_model.pth')
    config_path = os.getenv('CONFIG_PATH', 'configs/train_config.yaml')
    
    if os.path.exists(model_path) and os.path.exists(config_path):
        load_model(model_path, config_path)
    else:
        logger.warning("Model or config file not found. API will not be functional.")


def preprocess_image(image_file: UploadFile) -> torch.Tensor:
    """Preprocess uploaded image."""
    try:
        # Read image
        image_data = image_file.file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image)
        
        return image_tensor
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Twin Verification API",
        "version": "1.0.0",
        "status": "ready" if model is not None else "model not loaded",
        "endpoints": {
            "/verify": "POST - Verify if two images are of the same person",
            "/verify_base64": "POST - Verify using base64 encoded images",
            "/health": "GET - Health check",
            "/model_info": "GET - Model information"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": str(verifier.device) if verifier else "unknown"
    }


@app.get("/model_info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    num_params = sum(p.numel() for p in model.parameters())
    
    return {
        "backbone": config['model']['backbone'],
        "attention_type": config['model']['attention_type'],
        "feature_dim": config['model']['feature_dim'],
        "local_regions": config['model']['local_regions'],
        "num_parameters": num_params,
        "input_size": config['data']['image_size']
    }


@app.post("/verify")
async def verify_images(
    image1: UploadFile = File(..., description="First image file"),
    image2: UploadFile = File(..., description="Second image file"),
    threshold: float = 0.5,
    include_attention: bool = False
):
    """Verify if two uploaded images are of the same person."""
    if verifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess images
        img1_tensor = preprocess_image(image1)
        img2_tensor = preprocess_image(image2)
        
        # Perform verification
        if include_attention:
            similarity, attention_maps = verifier.verify_with_attention(img1_tensor, img2_tensor)
        else:
            similarity = verifier.verify_pair(img1_tensor, img2_tensor)
            attention_maps = None
        
        # Make decision
        is_same_person = similarity > threshold
        confidence = similarity if is_same_person else (1 - similarity)
        
        result = {
            "similarity_score": float(similarity),
            "threshold": threshold,
            "is_same_person": bool(is_same_person),
            "confidence": float(confidence),
            "decision": "SAME" if is_same_person else "DIFFERENT"
        }
        
        # Add attention visualization if requested
        if include_attention and attention_maps:
            # Convert attention maps to base64 for JSON response
            # This is a simplified version - in practice, you'd want to create proper visualizations
            result["attention_available"] = True
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {e}")


@app.post("/verify_base64")
async def verify_base64_images(
    request: Dict[str, Any]
):
    """Verify using base64 encoded images."""
    if verifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract parameters
        image1_b64 = request.get('image1')
        image2_b64 = request.get('image2')
        threshold = request.get('threshold', 0.5)
        include_attention = request.get('include_attention', False)
        
        if not image1_b64 or not image2_b64:
            raise HTTPException(status_code=400, detail="Both image1 and image2 are required")
        
        # Decode base64 images
        img1_data = base64.b64decode(image1_b64)
        img2_data = base64.b64decode(image2_b64)
        
        img1 = Image.open(io.BytesIO(img1_data)).convert('RGB')
        img2 = Image.open(io.BytesIO(img2_data)).convert('RGB')
        
        # Apply transforms
        img1_tensor = transform(img1)
        img2_tensor = transform(img2)
        
        # Perform verification
        if include_attention:
            similarity, attention_maps = verifier.verify_with_attention(img1_tensor, img2_tensor)
        else:
            similarity = verifier.verify_pair(img1_tensor, img2_tensor)
            attention_maps = None
        
        # Make decision
        is_same_person = similarity > threshold
        confidence = similarity if is_same_person else (1 - similarity)
        
        result = {
            "similarity_score": float(similarity),
            "threshold": threshold,
            "is_same_person": bool(is_same_person),
            "confidence": float(confidence),
            "decision": "SAME" if is_same_person else "DIFFERENT"
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Base64 verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {e}")


@app.post("/batch_verify")
async def batch_verify(
    request: Dict[str, Any]
):
    """Batch verification for multiple image pairs."""
    if verifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        pairs = request.get('pairs', [])
        threshold = request.get('threshold', 0.5)
        
        if not pairs:
            raise HTTPException(status_code=400, detail="No image pairs provided")
        
        results = []
        
        for i, pair in enumerate(pairs):
            try:
                # Decode images
                img1_data = base64.b64decode(pair['image1'])
                img2_data = base64.b64decode(pair['image2'])
                
                img1 = Image.open(io.BytesIO(img1_data)).convert('RGB')
                img2 = Image.open(io.BytesIO(img2_data)).convert('RGB')
                
                # Apply transforms
                img1_tensor = transform(img1)
                img2_tensor = transform(img2)
                
                # Perform verification
                similarity = verifier.verify_pair(img1_tensor, img2_tensor)
                is_same_person = similarity > threshold
                confidence = similarity if is_same_person else (1 - similarity)
                
                results.append({
                    "pair_id": i,
                    "similarity_score": float(similarity),
                    "is_same_person": bool(is_same_person),
                    "confidence": float(confidence),
                    "decision": "SAME" if is_same_person else "DIFFERENT"
                })
                
            except Exception as e:
                results.append({
                    "pair_id": i,
                    "error": str(e)
                })
        
        return JSONResponse(content={
            "results": results,
            "total_pairs": len(pairs),
            "successful_pairs": len([r for r in results if 'error' not in r])
        })
    
    except Exception as e:
        logger.error(f"Batch verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch verification failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
