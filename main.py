import os
import io
import json
import base64
from collections import Counter

import numpy as np
import cv2
from PIL import Image  # (kept if you later need it)

from fastapi import FastAPI, UploadFile, File, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader

import firebase_admin
from firebase_admin import credentials, firestore

# NOTE: We import YOLO only when we actually need it (lazy-load) to speed startup.
# from ultralytics import YOLO  # <-- moved into get_model()

# ---------- Security ----------
API_KEY = os.getenv("API_KEY", "default-fallback-key-for-local-dev")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

# ---------- Firestore ----------
db = None
try:
    creds_base64 = os.getenv("GOOGLE_CREDENTIALS_JSON_BASE64")
    if creds_base64:
        creds_json_str = base64.b64decode(creds_base64).decode("utf-8")
        creds_dict = json.loads(creds_json_str)
        cred = credentials.Certificate(creds_dict)
    else:
        # Falls back to local file if you're running locally
        cred = credentials.Certificate("serviceAccountKey.json")

    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firestore initialized successfully.")
except Exception as e:
    print(f"🔥 Firestore initialization failed: {e}")

# ---------- FastAPI ----------
app = FastAPI(title="Bottle Detection API")

# ---------- YOLO Lazy Loader ----------
_model = None
_model_path = os.getenv("YOLO_WEIGHTS", "best.pt")  # let you override via env if needed

def get_model():
    """
    Lazily load the YOLO model on first use to avoid long cold starts / timeouts.
    """
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            _model = YOLO(_model_path)
            print("✅ YOLOv8 model loaded successfully.")
        except Exception as e:
            print(f"🔥 Model loading failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
    return _model

# ---------- Health & Root (important for Render) ----------
@app.get("/")
def root():
    return {"status": "ok", "service": "recycle-counter-api", "message": "Up and running ✨"}

@app.get("/healthz")
def healthz():
    # Optionally check Firestore connectivity
    try:
        if db:
            _ = db.collection("health").document("ping")  # just touch a ref
        return {"status": "ok"}
    except Exception as e:
        # Still return 200 for Render health check, but include info
        return {"status": "degraded", "detail": str(e)}

# ---------- Main Inference Endpoint ----------
@app.post("/process-frame/")
async def process_image_frame(
    file: UploadFile = File(...),
    _: str = Depends(verify_api_key),  # enforces X-API-KEY on this route
):
    """
    Receives an image frame, performs object detection, and updates Firestore.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore is not connected")

    # 1) Read & decode image
    contents = await file.read()
    try:
        np_image = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # 2) Lazy-load model and run inference
    model = get_model()
    results = model(image, verbose=False)

    # 3) Collect detections
    detected_brands = []
    for r in results:
        # r.boxes.data shape: [N, 6] or similar; indexes: x1, y1, x2, y2, conf, cls
        if hasattr(r, "boxes") and r.boxes is not None and hasattr(r.boxes, "data"):
            boxes = r.boxes.data
            if getattr(boxes, "shape", None) is not None and boxes.shape[0] > 0:
                for box in boxes:
                    confidence = float(box[4].item())
                    class_id = int(box[5].item())
                    if confidence >= 0.25:
                        brand_name = r.names.get(class_id, str(class_id))
                        detected_brands.append(brand_name)

    if not detected_brands:
        return {"status": "success", "message": "No bottles detected in the frame."}

    brand_counts = Counter(detected_brands)

    # 4) Update Firestore (transactional)
    try:
        transaction = db.transaction()

        @firestore.transactional
        def update_in_transaction(trans):
            doc_ref = db.collection("bottle_counts").document("live_counts")
            snapshot = doc_ref.get(transaction=trans)
            current_brands = snapshot.to_dict().get("brands", {}) if snapshot.exists else {}

            for brand, count in brand_counts.items():
                current_brands[brand] = current_brands.get(brand, 0) + count

            trans.set(
                doc_ref,
                {
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "total_bottles": sum(current_brands.values()),
                    "brands": current_brands,
                },
            )

        update_in_transaction(transaction)
        print(f"✅ Processed frame. Detected: {dict(brand_counts)}. Firestore updated.")
    except Exception as e:
        print(f"🔥 Firestore update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Firestore update failed: {e}")

    return {
        "status": "success",
        "detected_brands": dict(brand_counts),
        "message": "Frame processed and counts updated in Firestore.",
    }