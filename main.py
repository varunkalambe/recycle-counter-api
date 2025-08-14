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

# --- Configuration & Initialization ---

# 1. API Key Setup for Security
API_KEY = os.getenv("API_KEY", "default-fallback-key-for-local-dev") # Reads from environment
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# 2. Firebase/Firestore Initialization
try:
    # Get the base64 encoded credentials from the environment variable
    creds_base64 = os.getenv("GOOGLE_CREDENTIALS_JSON_BASE64")
    if creds_base64:
        # Decode the base64 string to a JSON string
        creds_json_str = base64.b64decode(creds_base64).decode("utf-8")
        # Load the JSON string into a dictionary
        creds_dict = json.loads(creds_json_str)
        cred = credentials.Certificate(creds_dict)
    else:
        # Fallback for local development if the environment variable isn't set
        cred = credentials.Certificate("serviceAccountKey.json")
    
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firestore initialized successfully.")
except Exception as e:
    print(f"🔥 Firestore initialization failed: {e}")
    db = None

# 3. Initialize FastAPI App
app = FastAPI(title="Bottle Detection API")

# 4. Load Your YOLOv5 Model (best.pt)
try:
    # Use torch.hub.load to get the YOLOv5 model structure
    # 'ultralytics/yolov5' is the source repo, 'custom' specifies you're using your own weights
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    print("✅ YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"🔥 Model loading failed: {e}")
    model = None

# --- Helper Functions ---

# Function to secure the endpoint with an API key
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
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 2. Perform inference with the model
    results = model(image)

    # 3. Process the results
    # results.pandas().xyxy[0] gives a dataframe of detections
    detections = results.pandas().xyxy[0]
    detected_brands = [row['name'] for index, row in detections.iterrows()]
    
    if not detected_brands:
        return {"status": "success", "message": "No bottles detected in the frame."}

    # Count occurrences of each brand
    brand_counts = Counter(detected_brands)

    # 4. Update Firestore database
    # Use a transaction to safely increment counts
    transaction = db.transaction()
    for brand, count in brand_counts.items():
        doc_ref = db.collection("bottle_counts").document(brand)
        
        # This function will be run atomically by the transaction
        @firestore.transactional
        def update_in_transaction(trans, doc_ref, num_to_add):
            snapshot = doc_ref.get(transaction=trans)
            if snapshot.exists:
                current_count = snapshot.to_dict().get("count", 0)
                trans.update(doc_ref, {"count": current_count + num_to_add})
            else:
                trans.set(doc_ref, {"count": num_to_add})
        
        update_in_transaction(transaction, doc_ref, count)
    
    print(f"✅ Processed frame. Detected: {dict(brand_counts)}")

    return {
        "status": "success",
        "detected_brands": dict(brand_counts)
    }

# --- To Run the API ---
# In your terminal, run: uvicorn main:app --host 0.0.0.0 --port 8000