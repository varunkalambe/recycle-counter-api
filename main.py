import os
import io
import json
import torch
import uvicorn
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from PIL import Image
from collections import Counter
from ultralytics import YOLO
import base64
import numpy as np
import cv2

# --- Configuration & Initialization ---

# 1. API Key Setup for Security
API_KEY = os.getenv("API_KEY", "default-fallback-key-for-local-dev")
API_KEY_NAME = "X-API-KEY"  # <-- CORRECTED: This should be the name of the header
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# 2. Firebase/Firestore Initialization
try:
    creds_base64 = os.getenv("GOOGLE_CREDENTIALS_JSON_BASE64")
    if creds_base64:
        creds_json_str = base64.b64decode(creds_base64).decode("utf-8")
        creds_dict = json.loads(creds_json_str)
        cred = credentials.Certificate(creds_dict)
    else:
        cred = credentials.Certificate("serviceAccountKey.json")
    
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firestore initialized successfully.")
except Exception as e:
    print(f"🔥 Firestore initialization failed: {e}")
    db = None

# 3. Initialize FastAPI App
app = FastAPI(title="Bottle Detection API")

# 4. Load Your YOLOv8 Model
try:
    model = YOLO('best.pt')
    print("✅ YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"🔥 Model loading failed: {e}")
    model = None

# --- Helper Functions ---
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# --- API Endpoint ---
@app.post("/process-frame/", dependencies=[Security(get_api_key)])
async def process_image_frame(file: UploadFile = File(...)):
    """
    Receives an image frame, performs object detection, and updates Firestore.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    if not db:
        raise HTTPException(status_code=500, detail="Firestore is not connected")

    # 1. Read and decode the uploaded image
    contents = await file.read()
    try:
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if image is None:
             raise ValueError("Could not decode image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # 2. Perform inference with the model
    results = model(image, verbose=False)

    # 3. Process the results
    detected_brands = []
    
    for r in results:
        if r.boxes.data.shape[0] > 0:
            for box in r.boxes.data:
                class_id = int(box[5].item())
                confidence = float(box[4].item())
                
                if confidence >= 0.25:
                    brand_name = r.names[class_id]
                    detected_brands.append(brand_name)
    
    if not detected_brands:
        return {"status": "success", "message": "No bottles detected in the frame."}

    brand_counts = Counter(detected_brands)

    # 4. Update Firestore database
    if db:
        transaction = db.transaction()
        
        @firestore.transactional
        def update_in_transaction(trans):
            doc_ref = db.collection("bottle_counts").document("live_counts")
            snapshot = doc_ref.get(transaction=trans)
            
            if snapshot.exists:
                current_brands = snapshot.to_dict().get('brands', {})
            else:
                current_brands = {}

            for brand, count in brand_counts.items():
                current_brands[brand] = current_brands.get(brand, 0) + count

            trans.set(doc_ref, {
                'timestamp': firestore.SERVER_TIMESTAMP,
                'total_bottles': sum(current_brands.values()),
                'brands': current_brands
            })
            
        try:
            update_in_transaction(transaction)
            print(f"✅ Processed frame. Detected: {dict(brand_counts)}. Firestore updated.")
        except Exception as e:
            print(f"🔥 Firestore update failed: {e}")
            raise HTTPException(status_code=500, detail=f"Firestore update failed: {e}")

    return {
        "status": "success",
        "detected_brands": dict(brand_counts),
        "message": "Frame processed and counts updated in Firestore."
    }